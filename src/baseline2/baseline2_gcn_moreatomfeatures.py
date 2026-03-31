from __future__ import annotations

import os
import json
import random
import time
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import wandb

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugRes

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

#----------------------------
# PyG imports
#----------------------------
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.nn import BatchNorm


# ----------------------------
# YAML Saver 
# ----------------------------
def save_cfg_yaml(cfg_dict: dict, path: str) -> None:
    lines = []
    for k, v in cfg_dict.items():
        if v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            s = str(v).replace('"', '\\"')
            lines.append(f'{k}: "{s}"')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


#----------------------------
# Config
#----------------------------
@dataclass
class Config:
    # Experiment tagging
    run_tag: str = "gcn_mlp_final"   

    dataset_name: str = "GDSC1"

    # Model dims
    z_dim: int = 128
    drug_hidden: int = 64
    cell_hidden: int = 128 
    dropout: float = 0.5  

    # Training
    seed: int = 42
    batch_size: int = 128
    lr: float = 3e-5      
    weight_decay: float =  1e-2
    epochs: int = 60
    val_size: float = 0.1

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Runtime
    num_workers: int = 4  
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    subset_n: int | None = None

    # Paths 
    cache_dir: str = "cache"
    graph_cache_file: str = "gdsc1_graphs_gcn_atomfeat60.pt" 
    results_dir: str = "results/tdc_drugres_baseline/baseline2" 
    best_ckpt: str = "best_model.pt"


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#----------------------------
# Atom featurization
#----------------------------
def one_hot_with_unknown(value, choices):
    """
    One-hot encoding with extra 'unknown' bucket at the end.
    """
    encoding = [0.0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    else:
        encoding[-1] = 1.0
    return encoding


def atom_to_feature_vector(atom):
    # 1) Atom symbol
    atom_symbols = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]
    symbol_feat = one_hot_with_unknown(atom.GetSymbol(), atom_symbols)

    # 2) Degree
    degree_choices = [0, 1, 2, 3, 4, 5]
    degree_feat = one_hot_with_unknown(atom.GetDegree(), degree_choices)

    # 3) Formal charge
    charge_choices = [-2, -1, 0, 1, 2]
    charge_feat = one_hot_with_unknown(atom.GetFormalCharge(), charge_choices)

    # 4) Hybridization
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hyb_feat = one_hot_with_unknown(atom.GetHybridization(), hyb_choices)

    # 5) Aromatic / Ring
    aromatic_feat = [float(atom.GetIsAromatic())]
    ring_feat = [float(atom.IsInRing())]

    # 6) Number of hydrogens
    num_h_choices = [0, 1, 2, 3, 4]
    num_h_feat = one_hot_with_unknown(atom.GetTotalNumHs(), num_h_choices)

    # 7) Implicit valence
    implicit_val_choices = [0, 1, 2, 3, 4, 5, 6]
    implicit_val_feat = one_hot_with_unknown(atom.GetImplicitValence(), implicit_val_choices)

    # 8) Total valence
    total_val_choices = [0, 1, 2, 3, 4, 5, 6]
    total_val_feat = one_hot_with_unknown(atom.GetTotalValence(), total_val_choices)

    # 9) Chirality
    chiral_choices = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    chiral_feat = one_hot_with_unknown(atom.GetChiralTag(), chiral_choices)

    # 10) Atomic mass (scaled)
    mass_feat = [atom.GetMass() / 100.0]

    features = (
        symbol_feat
        + degree_feat
        + charge_feat
        + hyb_feat
        + aromatic_feat
        + ring_feat
        + num_h_feat
        + implicit_val_feat
        + total_val_feat
        + chiral_feat
        + mass_feat
    )
    return features

# ----------------------------
# Featurizer: SMILES -> graph 
# ----------------------------
def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))

    x = torch.tensor(atom_features, dtype=torch.float32)

    edges_list = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        b_feat = [
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated())
        ]

        edges_list.append([i, j])
        edge_attrs.append(b_feat)

        edges_list.append([j, i])
        edge_attrs.append(b_feat)

    if len(edges_list) > 0:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

#----------------------------
# Dataset
#----------------------------
class DrugResGraphDataset(Dataset):
    def __init__(self, drug_graphs: List[Data], cell_expr: np.ndarray, y: np.ndarray):
        self.drug_graphs = drug_graphs
        self.cell_expr = cell_expr
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        graph = self.drug_graphs[idx]
        x_cell = torch.from_numpy(self.cell_expr[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32).view(1)
        return graph, x_cell, y
    

#----------------------------
# DRUG ENCODER (GCN )
#----------------------------
class DrugGNNEncoder(torch.nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, z_dim: int, dropout: float):
        super().__init__()

        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim * 2)
        self.bn3 = BatchNorm(hidden_dim * 4)

        # Projection Head
        self.fc_g1 = nn.Linear(hidden_dim * 4, 512)
        self.fc_g2 = nn.Linear(512, z_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_graph):
        x, edge_index, batch = drug_graph.x, drug_graph.edge_index, drug_graph.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x) 

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_max_pool(x, batch)

        x = self.fc_g1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        z_drug = self.fc_g2(x)
        z_drug = self.dropout(z_drug)

        return z_drug
        
#----------------------------
# CELL ENCODER (MLP with BatchNorm)
#----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim ),
            nn.BatchNorm1d(hidden_dim ),  # ΚΡΙΣΙΜΟ: Το BatchNorm είναι εδώ!
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim , hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), 
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
#----------------------------
# FULL MODEL
#----------------------------
class DrugResponsePredictor(nn.Module):
    def __init__(self, node_feat_dim: int, cell_in: int, cfg: Config):
        super().__init__()
        
        self.drug_gnn = DrugGNNEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=cfg.drug_hidden,
            z_dim=cfg.z_dim,
            dropout=cfg.dropout,
        )
        
        self.cell_mlp = MLP(
            input_dim=cell_in, 
            hidden_dim=cfg.cell_hidden, 
            output_dim=cfg.z_dim, 
            dropout=cfg.dropout
        )
        
        self.head = nn.Sequential(
            nn.Linear(cfg.z_dim * 2, cfg.z_dim),
            nn.BatchNorm1d(cfg.z_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(cfg.z_dim, cfg.z_dim // 2),
            nn.BatchNorm1d(cfg.z_dim // 2),
            nn.ReLU(),
            
            nn.Linear(cfg.z_dim // 2, 1),

            nn.Sigmoid()
        )

    def forward(self, drug_graph, x_cell: torch.Tensor) -> torch.Tensor:
        z_drug = self.drug_gnn(drug_graph)
        z_cell = self.cell_mlp(x_cell) 

        z = torch.cat([z_drug, z_cell], dim=1)
        return self.head(z)


#----------------------------
# Metrics
#----------------------------
@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).cpu().item())

@torch.no_grad()
def pearsonr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    if pred.std() < 1e-12 or target.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])

#----------------------------
# Graph cache
#----------------------------
def build_or_load_graph_cache(cfg: Config, smiles_list: List[str]) -> List[Data]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
   
    cache_path = os.path.join(cfg.cache_dir, cfg.graph_cache_file)

    if os.path.exists(cache_path):
        cache_dict = torch.load(cache_path, map_location="cpu", weights_only=False)  
        meta = cache_dict["meta"]
        if meta.get("n") == len(smiles_list):
            print(f"Loaded Graph cache: {cache_path}")
            return cache_dict["graphs"]
        print("Cache metadata mismatch -> rebuilding...")

    print("Building Graph cache (CPU)...")
    graphs = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        graphs.append(g)
        if (i + 1) % 5000 == 0:
            print(f"  featurized {i+1}/{len(smiles_list)}")

    meta = {"n": len(smiles_list)}
    torch.save({"graphs": graphs, "meta": meta}, cache_path)
    print(f"Saved Graph cache: {cache_path}")
    return graphs

#----------------------------
# Train / Eval / Predict
#----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for drug_graph, x_cell, y in loader:
        drug_graph = drug_graph.to(device)
        x_cell = x_cell.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        y_hat = model(drug_graph, x_cell)
        loss = loss_fn(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> Tuple[float, float, float]:
    model.eval()
    total = 0.0
    n = 0
    preds, targets = [], []
    for drug_graph, x_cell, y in loader:
        drug_graph = drug_graph.to(device)
        x_cell = x_cell.to(device)
        y = y.to(device)

        y_hat = model(drug_graph, x_cell)
        loss = loss_fn(y_hat, y)

        total += float(loss.item()) * y.size(0)
        n += y.size(0)

        preds.append(y_hat.detach().cpu())
        targets.append(y.detach().cpu())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return total / max(n, 1), rmse(pred, target), pearsonr(pred, target)

@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for drug_graph, x_cell, y in loader:
        drug_graph = drug_graph.to(device)
        x_cell = x_cell.to(device)
        
        y_hat = model(drug_graph, x_cell).detach().cpu().view(-1)
        preds.append(y_hat)
        targets.append(y.detach().cpu().view(-1))
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()

#----------------------------
# Plotting helpers
#----------------------------
def save_learning_plots(history: List[Dict[str, Any]], run_dir: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_rmse = [h["val_rmse"] for h in history]
    val_p = [h["val_pearson"] for h in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_rmse, label="val_RMSE")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "rmse_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_p, label="val_Pearson")
    plt.xlabel("epoch")
    plt.ylabel("Pearson")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pearson_curve.png"), dpi=150)
    plt.close()

def save_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, run_dir: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pred_vs_true.png"), dpi=150)
    plt.close()

#----------------------------
# Main
#----------------------------
def main() -> None:
    cfg = Config()
    cfg.subset_n = None
    seed_everything(cfg.seed)

    os.makedirs(cfg.results_dir, exist_ok=True)
    run_id = f"{cfg.run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    wandb.init(
        project="Machine-learning-for-prediction-of-drug-response-for-drug-cell-pairs", 
        name=run_id,                        
        config=cfg.__dict__      )
               

    device = torch.device(cfg.device)
    print(f"Run dir: {run_dir}")

    print("Loading FULL GDSC1 dataset...")
    data = DrugRes(name=cfg.dataset_name)
    df = data.get_data()

    # LEAVE-DRUG-OUT split 
    idx = np.arange(len(df))
    group_col = "Drug_ID" if "Drug_ID" in df.columns else "Drug"
    groups = df[group_col].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.seed)
    train_idx, val_idx = next(gss.split(idx, groups=groups))

    idx_train, idx_val = idx[train_idx], idx[val_idx]
    
    print(f"[Split] Train samples: {len(idx_train)} | Val samples: {len(idx_val)}")

    # Variance-based gene selection (TRAIN only)
    print("Estimating gene variance using a random TRAIN subset...")
    subset_size = min(10000, len(idx_train))
    subset_indices = np.random.choice(idx_train, size=subset_size, replace=False)
    subset_expr = np.array(df.iloc[subset_indices]["Cell Line"].tolist(), dtype=np.float32)

    gene_variances = np.var(subset_expr, axis=0)
    TOP_K = 1000
    top_indices = np.argsort(gene_variances)[-TOP_K:]
    top_indices = np.sort(top_indices)

    print("Loading full dataset with selected genes...")
    full_expr_list = df["Cell Line"].tolist()
    cell_expr = np.zeros((len(df), TOP_K), dtype=np.float32)

    batch_size = 10000
    for i in range(0, len(df), batch_size):
        end = min(i + batch_size, len(df))
        batch_arr = np.array(full_expr_list[i:end], dtype=np.float32)
        cell_expr[i:end] = batch_arr[:, top_indices]

    import gc
    del full_expr_list, subset_expr
    gc.collect()

    # Graph processing & Standardization
    smiles = df["Drug"].astype(str).tolist()
    y = df["Y"].astype(float).to_numpy()
    
    drug_graphs = build_or_load_graph_cache(cfg, smiles)

    # -----------------------
    # Target normalization (Min-Max Scaling [0, 1])
    # -----------------------
    y_train = y[idx_train]

    y_min = y_train.min()
    y_max = y_train.max()
    
    denominator = max(y_max - y_min, 1e-8) # Ασφάλεια για διαίρεση με το 0

    # Κανονικοποιούμε όλο το y (train & val) ώστε να πέφτει μεταξύ 0 και 1
    y = (y - y_min) / denominator   

    # -----------------------
    # Cell standardization
    # -----------------------
    train_mean = cell_expr[idx_train].mean(axis=0, keepdims=True)
    train_std = cell_expr[idx_train].std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    cell_expr = (cell_expr - train_mean) / train_std

    # -----------------------
    # Create datasets
    # -----------------------
    train_ds = DrugResGraphDataset([drug_graphs[i] for i in idx_train], cell_expr[idx_train], y[idx_train])
    val_ds = DrugResGraphDataset([drug_graphs[i] for i in idx_val], cell_expr[idx_val], y[idx_val])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


    # Model Setup
    sample_graph = next(g for g in drug_graphs if g is not None)
    node_feat_dim = sample_graph.x.shape[1]

    model = DrugResponsePredictor(
        node_feat_dim=node_feat_dim,
        cell_in=TOP_K,
        cfg=cfg
    ).to(device)    
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    # Training Loop
    best_val_rmse = float("inf")
    best_epoch = -1
    history = []
    pat = 0
    best_path = os.path.join(run_dir, cfg.best_ckpt)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_rmse, val_p = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_RMSE={val_rmse:.4f} | val_Pearson={val_p:.4f}")
        
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, 
            "val_rmse": val_rmse, "val_pearson": val_p, "lr": opt.param_groups[0]["lr"]
        })

        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_RMSE": val_rmse,
            "val_Pearson": val_p,
            "learning_rate": opt.param_groups[0]["lr"]
        })

        if val_rmse < best_val_rmse - cfg.min_delta:
            best_val_rmse = val_rmse
            best_epoch = epoch
            pat = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, best_path)
        else:
            pat += 1
            if pat >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_learning_plots(history, run_dir)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    y_pred, y_true = predict(model, val_loader, device)
    save_pred_scatter(y_true, y_pred, run_dir)

# Final Metrics
    best_val_pearson = history[best_epoch - 1]["val_pearson"]
    train_drugs_set = set(df.iloc[idx_train]["Drug"])
    val_drugs_set = set(df.iloc[idx_val]["Drug"])

    metrics = {
        "run_id": run_id,
        "split_type": "leave_drug_out",
        "n_samples": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_unique_drugs": int(df["Drug"].nunique()),
        "n_train_drugs": int(len(train_drugs_set)),
        "n_val_drugs": int(len(val_drugs_set)),
        "n_overlap_drugs": int(len(train_drugs_set.intersection(val_drugs_set))),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "best_val_pearson": float(best_val_pearson),
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    yaml_path = os.path.join(run_dir, "cfg.yaml")
    save_cfg_yaml(cfg.__dict__, yaml_path)
    print(f"Saved config: {yaml_path}")
    print("Done! Results saved to:", run_dir)

    wandb.finish()

if __name__ == "__main__":
    main()