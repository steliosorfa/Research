from __future__ import annotations

from http.client import responses
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



#----------------------------
# new imports
#----------------------------

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool as gap, global_max_pool as gmp

from torch_geometric.nn import BatchNorm
#----------------------------



# ----------------------------
# YAML Saver 
# ----------------------------
def save_cfg_yaml(cfg_dict: dict, path: str) -> None:
    """Saves a flat config dictionary to a basic YAML file without requiring PyYAML."""
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
    run_tag: str = "geognn_3d_ldo_tuned_v1_seed44"
    dataset_name: str = "GDSC1"

    z_dim: int = 128
    drug_hidden: int = 256
    cell_hidden: int = 512
    dropout: float = 0.3

    seed: int = 44
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 80 
    val_size: float = 0.1

    patience: int = 20 
    min_delta: float = 1e-5

    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    subset_n: int | None = None

    cache_dir: str = "cache"
    graph_cache_file: str = "gdsc1_geognn_3d_v1.pt"

    results_dir: str = "results/tdc_drugres_geognn/3d_baseline"
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

# ----------------------------
# Featurizer: SMILES -> 3Dgraph 
# ----------------------------

from scipy.spatial.distance import cdist

def smiles_to_graph_3d(smiles: str):
    """Returns (drug_atom, drug_bond) — two PyG Data objects."""
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # --- 3D conformer generation ---
    try:
        new_mol = Chem.AddHs(mol)
        cids = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        index = np.argmin([x[1] for x in res])
        new_mol = Chem.RemoveHs(new_mol)
        conf = new_mol.GetConformer(id=int(index))
        atom_poses = np.array([list(conf.GetAtomPosition(i)) 
                                for i in range(new_mol.GetNumAtoms())], dtype=np.float32)
    except:
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        new_mol = mol
        atom_poses = np.array([list(conf.GetAtomPosition(i)) 
                                for i in range(mol.GetNumAtoms())], dtype=np.float32)

    # --- Atom features (node features for A2A graph) ---
    atom_features = []
    for atom in new_mol.GetAtoms():
        atom_features.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
            float(int(atom.GetHybridization())),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
            atom.GetMass(),
        ])
    x_atom = torch.tensor(atom_features, dtype=torch.float32)

    # --- Bond features + A2A edge index ---
    edges_list = []
    edge_attrs = []
    bond_lengths = []

    for bond in new_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_len = float(np.linalg.norm(atom_poses[i] - atom_poses[j]))
        b_feat = [
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
            float(str(bond.GetBondType()) == 'SINGLE'),
            float(str(bond.GetBondType()) == 'DOUBLE'),
            float(str(bond.GetBondType()) == 'AROMATIC'),
            b_len,
        ]
        edges_list.append([i, j]); edge_attrs.append(b_feat); bond_lengths.append(b_len)
        edges_list.append([j, i]); edge_attrs.append(b_feat); bond_lengths.append(b_len)

    if len(edges_list) == 0:
        return None, None

    edge_index_atom = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_attr_atom  = torch.tensor(edge_attrs, dtype=torch.float32)

    # --- B2B graph with bond angles ---

    def get_angle(vec1,vec2):
        n1,n2 = np.linalg.norm(vec1),np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
          return 0.0
        return  float(np.arccos(np.clip(np.dot(vec1 / (n1 + 1e-5), vec2 / (n2 + 1e-5)), -1.0, 1.0)))


    edge_pairs = edge_index_atom.t().numpy()  # shape (E, 2)
    E = len(edge_pairs)
    super_edges = []
    bond_angles  = []

    for bond_idx, (src, dst) in enumerate(edge_pairs):
        # find all bonds whose dst == src of current bond
        neighbor_bond_ids = np.where(edge_pairs[:, 1] == src)[0]
        for nb_idx in neighbor_bond_ids:
            if nb_idx == bond_idx:
                continue
            nb_src, nb_dst = edge_pairs[nb_idx]
            super_edges.append([nb_idx, bond_idx])
            vec1 = atom_poses[src]  - atom_poses[dst]
            vec2 = atom_poses[nb_src] - atom_poses[nb_dst]
            bond_angles.append(get_angle(vec1, vec2))

    if len(super_edges) == 0:
        return None, None

    edge_index_bond = torch.tensor(super_edges, dtype=torch.long).t().contiguous()
    edge_attr_bond  = torch.tensor(bond_angles, dtype=torch.float32).unsqueeze(1)
    x_bond          = edge_attr_atom   # bond features become node features in B2B graph

    drug_atom = Data(x=x_atom,  edge_index=edge_index_atom, edge_attr=edge_attr_atom)
    drug_bond = Data(x=x_bond,  edge_index=edge_index_bond, edge_attr=edge_attr_bond)

    return drug_atom, drug_bond
#----------------------------
# Dataset
#----------------------------

class DrugResGraphDataset(Dataset):
    def __init__(self, atom_graphs, bond_graphs, cell_expr, y):
        self.atom_graphs = atom_graphs
        self.bond_graphs = bond_graphs
        self.cell_expr = cell_expr
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.atom_graphs[idx], self.bond_graphs[idx],
                torch.from_numpy(self.cell_expr[idx]),
                torch.tensor(self.y[idx], dtype=torch.float32).view(1))
    

#----------------------------
# Geometric Drug Encoder (A2A + B2B GIN)
#----------------------------

class DrugGeoEncoder(nn.Module):
    def __init__(self, atom_feat_dim: int, bond_feat_dim: int,
                 hidden_dim: int, z_dim: int, dropout: float):
        super().__init__()

        # A2A GIN layers (atom graph)
        self.atom_proj = nn.Linear(atom_feat_dim, hidden_dim)
        self.edge_proj_a = nn.Linear(bond_feat_dim, hidden_dim)

        self.gin1 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))
        self.gin2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))

        # B2B bond-angle GIN layers (bond graph)
        self.bond_node_proj = nn.Linear(bond_feat_dim, hidden_dim)
        self.edge_proj_b    = nn.Linear(1, hidden_dim)  # angle is scalar

        self.gin_b1 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))
        
        self.gin_b2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))
        self.bn_b2 = BatchNorm(hidden_dim)

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn_b = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # project concatenated atom+bond embeddings → z_dim
        self.out_proj = nn.Linear(hidden_dim * 2, z_dim)

    def forward(self, atom_graph, bond_graph):
        # --- A2A stream ---
        xa = self.atom_proj(atom_graph.x)
        ea = self.edge_proj_a(atom_graph.edge_attr)

        xa = self.dropout(F.relu(self.bn1(self.gin1(xa, atom_graph.edge_index, ea))))
        xa = self.dropout(F.relu(self.bn2(self.gin2(xa, atom_graph.edge_index, ea))))
        h_atom = gmp(xa, atom_graph.batch)

        # --- B2B stream ---
        xb = self.bond_node_proj(bond_graph.x)
        eb = self.edge_proj_b(bond_graph.edge_attr)

        xb = self.dropout(F.relu(self.bn_b(self.gin_b1(xb, bond_graph.edge_index, eb))))
        xb = self.dropout(F.relu(self.bn_b2(self.gin_b2(xb, bond_graph.edge_index, eb))))  
        h_bond = gmp(xb, bond_graph.batch)

        # --- fuse ---
        h_geo = F.relu(self.out_proj(torch.cat([h_atom, h_bond], dim=1)))
        return h_geo
    
#----------------------------
# Model
#----------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class DrugResponsePredictor(nn.Module):
    def __init__(self, node_feat_dim: int, cell_in: int, cfg: Config):
        super().__init__()
        
        self.drug_gnn = DrugGeoEncoder(
            atom_feat_dim=9,       # 9 atom features now (added mass)
            bond_feat_dim=6,       # 6 bond features
            hidden_dim=cfg.drug_hidden,
            z_dim=cfg.z_dim,
            dropout=cfg.dropout,
        )
        
        # 2. Instantiate your Cell MLP
        self.cell_mlp = MLP(
            input_dim=cell_in, 
            hidden_dim=cfg.cell_hidden, 
            output_dim=cfg.z_dim, 
            dropout=cfg.dropout
        )
        
        # 3. The Fusion Head
        self.head = nn.Sequential(
            nn.Linear(cfg.z_dim * 2, cfg.z_dim),
            nn.BatchNorm1d(cfg.z_dim),    
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            
            nn.Linear(cfg.z_dim, cfg.z_dim // 2),
            nn.BatchNorm1d(cfg.z_dim // 2),
            nn.ReLU(),
            
            nn.Linear(cfg.z_dim // 2, 1),
            nn.Sigmoid()                  # <--  πρόβλεψη στο [0, 1]
        )

    def forward(self, atom_graph, bond_graph, x_cell: torch.Tensor) -> torch.Tensor:

        z_drug = self.drug_gnn(atom_graph, bond_graph)
        z_cell = self.cell_mlp(x_cell)

        # Fuse and Predict
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

def denormalize(y_scaled: torch.Tensor, y_min: float, y_max: float) -> torch.Tensor:
    return y_scaled * (y_max - y_min) + y_min



#----------------------------
# Graph cache (Replaces ECFP Cache)
#----------------------------
def build_or_load_graph_cache(cfg: Config, smiles_list: List[str]):
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, cfg.graph_cache_file)

    if os.path.exists(cache_path):
        cache_dict = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_dict["meta"].get("n") == len(smiles_list):
            print(f"Loaded Graph cache: {cache_path}")
            return cache_dict["atom_graphs"], cache_dict["bond_graphs"]
        print("Cache mismatch -> rebuilding...")

    print("Building 3D Graph cache...")
    
    # 1. Βρίσκουμε τα μοναδικά φάρμακα (για να μην υπολογίζουμε τα ίδια 3D 1000 φορές!)
    unique_smiles = list(set(smiles_list))
    print(f"Found {len(unique_smiles)} unique drugs. Generating 3D conformers for them...")
    
    smiles_to_graphs = {}
    failed = 0
    
    # 2. Υπολογίζουμε τα 3D γραφήματα ΜΟΝΟ για τα μοναδικά
    for i, smi in enumerate(unique_smiles):
        da, db = smiles_to_graph_3d(smi)
        smiles_to_graphs[smi] = (da, db)
        if da is None:
            failed += 1
        
        # Τυπώνουμε πιο συχνά για να βλέπεις ότι προχωράει!
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(unique_smiles)} unique drugs | failed: {failed}")

    # 3. Αντιστοιχούμε (map) τα έτοιμα γραφήματα σε όλο το dataset των 177k γραμμών
    print("Mapping calculated 3D graphs to the full dataset...")
    atom_graphs, bond_graphs = [], []
    for smi in smiles_list:
        da, db = smiles_to_graphs[smi]
        atom_graphs.append(da)
        bond_graphs.append(db)

    torch.save({"atom_graphs": atom_graphs, "bond_graphs": bond_graphs,
                "meta": {"n": len(smiles_list)}}, cache_path)
    print(f"Saved cache: {cache_path}")
    return atom_graphs, bond_graphs



#----------------------------
# Train / Eval / Predict
#----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for atom_graph, bond_graph, x_cell, y in loader:
        atom_graph = atom_graph.to(device)
        bond_graph = bond_graph.to(device)
        x_cell = x_cell.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        y_hat = model(atom_graph, bond_graph, x_cell)
        loss = loss_fn(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, y_min=None, y_max=None):
    model.eval()
    total = 0.0
    n = 0
    preds, targets = [], []
    for atom_graph, bond_graph, x_cell, y in loader:
        atom_graph = atom_graph.to(device)
        bond_graph  = bond_graph.to(device)
        x_cell = x_cell.to(device)
        y = y.to(device)

        y_hat = model(atom_graph, bond_graph, x_cell)
        loss = loss_fn(y_hat, y)
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
        preds.append(y_hat.detach().cpu())
        targets.append(y.detach().cpu())

    pred   = torch.cat(preds,   dim=0)
    target = torch.cat(targets, dim=0)

    rmse_scaled = rmse(pred, target)
    p           = pearsonr(pred, target)

    # invert min-max scaling → ln(IC50) units
    pred_orig   = denormalize(pred,   y_min, y_max)
    target_orig = denormalize(target, y_min, y_max)
    rmse_orig   = rmse(pred_orig, target_orig)

    return total / max(n, 1), rmse_scaled, rmse_orig, p

@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for atom_graph, bond_graph, x_cell, y in loader:
        atom_graph = atom_graph.to(device)
        bond_graph = bond_graph.to(device)
        x_cell = x_cell.to(device)
        
        y_hat = model(atom_graph, bond_graph, x_cell).detach().cpu().view(-1)
        preds.append(y_hat)
        targets.append(y.detach().cpu().view(-1))
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()



#----------------------------
# Plotting helpers
#----------------------------
def save_learning_plots(history: List[Dict[str, Any]], run_dir: str) -> None:
    epochs     = [h["epoch"]          for h in history]
    train_loss = [h["train_loss"]     for h in history]
    val_loss   = [h["val_loss"]       for h in history]
    val_rmse   = [h["val_rmse_scaled"]   for h in history]  
    val_rmse_orig = [h["val_rmse_lnIC50"] for h in history] 
    val_p      = [h["val_pearson"]    for h in history]

    plt.figure()
    plt.plot(epochs, val_rmse_orig, label="val_RMSE_lnIC50")
    plt.xlabel("epoch")
    plt.ylabel("RMSE (ln IC50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "rmse_lnIC50_curve.png"), dpi=150)
    plt.close()

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
from sklearn.model_selection import GroupShuffleSplit

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
        config=cfg.__dict__      
    )

    device = torch.device(cfg.device)
    print(f"Run dir: {run_dir}")

    # 1. Load Data Frame
    print("Loading FULL GDSC1 dataset...")
    data = DrugRes(name=cfg.dataset_name)
    df = data.get_data()

    # --- ΝΕΟΣ ΚΩΔΙΚΑΣ: Φιλτράρισμα αποτυχημένων 3D φαρμάκων ---
    print("Filtering out drugs that cannot be parsed into 3D graphs...")
    unique_smiles = df["Drug"].unique()
    valid_smiles = set()
    for smi in unique_smiles:
        da, db = smiles_to_graph_3d(smi)
        if da is not None and db is not None:
            valid_smiles.add(smi)
    
    initial_len = len(df)
    df = df[df["Drug"].isin(valid_smiles)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Removed {initial_len - len(df)} pairs containing invalid 3D drugs.")
    print(f"Remaining pairs: {len(df)}")

    # ----------------------------
    # LEAVE-DRUG-OUT split 
    # ----------------------------
    idx = np.arange(len(df))
    group_col = "Drug_ID" if "Drug_ID" in df.columns else "Drug"
    groups = df[group_col].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.seed)
    train_idx, val_idx = next(gss.split(idx, groups=groups))

    idx_train, idx_val = idx[train_idx], idx[val_idx]
    
    print(f"[Split] Train samples: {len(idx_train)} | Val samples: {len(idx_val)}")

    # ----------------------------
    # Variance-based gene selection (TRAIN only)
    # ----------------------------
    print("Estimating gene variance using a random TRAIN subset...")
    subset_size = min(10000, len(idx_train))
    subset_indices = np.random.choice(idx_train, size=subset_size, replace=False)
    subset_expr = np.array(df.iloc[subset_indices]["Cell Line"].tolist(), dtype=np.float32)

    gene_variances = np.var(subset_expr, axis=0)
    TOP_K = 1000
    top_indices = np.argsort(gene_variances)[-TOP_K:]
    top_indices = np.sort(top_indices)

    # Build cell_expr for ALL samples
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

    # ----------------------------
    # Graph processing & Standardization
    # ----------------------------
    smiles = df["Drug"].astype(str).tolist()
    y = df["Y"].astype(float).to_numpy()
    
    # NEW: Build the graph cache!
    atom_graphs, bond_graphs = build_or_load_graph_cache(cfg, smiles)

    # -----------------------
    # Target normalization (Min-Max Scaling [0, 1]) 
    # -----------------------
    y_train = y[idx_train]

    y_min = y_train.min()
    y_max = y_train.max()

    
    denominator = max(y_max - y_min, 1e-8) 

    y = (y - y_min) / denominator   

    # -----------------------
    # Cell Standardization
    # -----------------------

    train_mean = cell_expr[idx_train].mean(axis=0, keepdims=True)
    train_std = cell_expr[idx_train].std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    cell_expr = (cell_expr - train_mean) / train_std

    # Datasets using the List of Graphs
    train_ds = DrugResGraphDataset(
    [atom_graphs[i] for i in idx_train],
    [bond_graphs[i] for i in idx_train],
    cell_expr[idx_train], y[idx_train])

    val_ds = DrugResGraphDataset(
        [atom_graphs[i] for i in idx_val],
        [bond_graphs[i] for i in idx_val],
        cell_expr[idx_val], y[idx_val])
    # PyG DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # ----------------------------
    # Model Setup
    # ----------------------------
    # CRITICAL: node_feat_dim=8 because we extracted 8 features per atom in smiles_to_graph
    model = DrugResponsePredictor(node_feat_dim=8, cell_in=TOP_K, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_val_rmse = float("inf")
    best_epoch = -1
    history = []
    pat = 0
    best_path = os.path.join(run_dir, cfg.best_ckpt)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_rmse, val_rmse_orig, val_p = eval_epoch(model, val_loader, loss_fn, device, y_min, y_max)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{cfg.epochs} "
            f"| train_loss={tr_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_RMSE_scaled={val_rmse:.4f} "
            f"| val_RMSE_lnIC50={val_rmse_orig:.4f} "   
            f"| val_Pearson={val_p:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_rmse_scaled": val_rmse,
            "val_rmse_lnIC50": val_rmse_orig,     
            "val_pearson": val_p,
            "lr": opt.param_groups[0]["lr"]
        })

        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_RMSE_scaled": val_rmse,
            "val_RMSE_lnIC50": val_rmse_orig,      
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

    ckpt = torch.load(best_path, map_location=device, weights_only=False)  # <--- Fixed!
    model.load_state_dict(ckpt["model_state"])

    y_pred, y_true = predict(model, val_loader, device)
    save_pred_scatter(y_true, y_pred, run_dir)

# -------------------------------------------------------
# Final Metrics & Config (Expanded for Git)
# -------------------------------------------------------
    best_val_pearson = history[best_epoch - 1]["val_pearson"]
    
    # Calculate drug sets for logging
    train_drugs_set = set(df.iloc[idx_train]["Drug"])
    val_drugs_set = set(df.iloc[idx_val]["Drug"])

    metrics = {
        "run_id": run_id,
        "split_type": "leave_drug_out",  # <-- Updated to reflect our cold split!
        "n_samples": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_unique_drugs": int(df["Drug"].nunique()),
        "n_train_drugs": int(len(train_drugs_set)),
        "n_val_drugs": int(len(val_drugs_set)),
        "n_overlap_drugs": int(len(train_drugs_set.intersection(val_drugs_set))),
        "best_epoch": int(best_epoch),
         "best_val_rmse_scaled": float(best_val_rmse),
        "best_val_rmse_lnIC50": float(history[best_epoch - 1]["val_rmse_lnIC50"]),
        "best_val_pearson": float(best_val_pearson),
    }

    # Save metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save cfg.yaml
    yaml_path = os.path.join(run_dir, "cfg.yaml")
    save_cfg_yaml(cfg.__dict__, yaml_path)
    print(f"Saved config: {yaml_path}")
    
    print("Done! Results saved to:", run_dir)

    wandb.finish()

if __name__ == "__main__":
    main()