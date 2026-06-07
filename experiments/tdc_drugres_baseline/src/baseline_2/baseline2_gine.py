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

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugRes

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool as gap
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


# ══════════════════════════════════════════
# UNIFIED EXPERIMENTAL SETTINGS — seed=44
# epochs=80, patience=20, split 80/10/10
# DO NOT CHANGE between models
# ══════════════════════════════════════════
@dataclass
class Config:
    run_tag: str = "baseline2_gine"
    dataset_name: str = "GDSC1"

    # ── Split strategy ──────────────────────────────────────────────
    # Options: "random" | "blind_drug" | "blind_cell"
    split_type: str = "random"

    # Ratios: 80 / 10 / 10
    test_size: float = 0.10
    val_size: float = 0.111       # 0.111 * 0.9 ≈ 0.10 overall

    # Model dims
    z_dim: int = 128
    drug_hidden: int = 64
    cell_hidden: int = 512
    dropout: float = 0.5

    # Training — unified across all models
    seed: int = 44
    batch_size: int = 128
    lr: float = 5e-5
    weight_decay: float = 1e-3
    epochs: int = 80
    patience: int = 20
    min_delta: float = 1e-4

    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    subset_n: int | None = None

    # Paths — set dynamically in main() based on __file__
    cache_dir: str = ""
    graph_cache_file: str = "gdsc1_graphs_gine_v2.pt"
    results_dir: str = ""
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
# Featurizer: SMILES -> graph
# ----------------------------
def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetNumRadicalElectrons()),
            float(int(atom.GetHybridization())),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float32)

    edges_list = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = [
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
        ]
        edges_list.append([i, j]); edge_attrs.append(b_feat)
        edges_list.append([j, i]); edge_attrs.append(b_feat)

    if len(edges_list) > 0:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 2), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ----------------------------
# Dataset
# ----------------------------
class DrugResGraphDataset(Dataset):
    def __init__(self, drug_graphs: List[Data], cell_expr: np.ndarray, y: np.ndarray):
        self.drug_graphs = drug_graphs
        self.cell_expr   = cell_expr
        self.y           = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        graph  = self.drug_graphs[idx]
        x_cell = torch.from_numpy(self.cell_expr[idx])
        y      = torch.tensor(self.y[idx], dtype=torch.float32).view(1)
        return graph, x_cell, y


# ----------------------------
# GNN Encoder (GINE)
# ----------------------------
class DrugGNNEncoder(torch.nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int, z_dim: int, dropout: float):
        super().__init__()

        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim))
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, hidden_dim))
        nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                             nn.Linear(hidden_dim, z_dim))

        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)
        self.conv3 = GINEConv(nn3)

        self.bn1     = BatchNorm(hidden_dim)
        self.bn2     = BatchNorm(hidden_dim)
        self.bn3     = BatchNorm(z_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_graph):
        x          = drug_graph.x
        edge_index = drug_graph.edge_index
        edge_attr  = drug_graph.edge_attr
        batch      = drug_graph.batch

        x         = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        x = self.dropout(F.relu(self.bn1(self.conv1(x, edge_index, edge_attr))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, edge_index, edge_attr))))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))

        return gap(x, batch)


# ----------------------------
# Model
# ----------------------------
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
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DrugResponsePredictor(nn.Module):
    def __init__(self, node_feat_dim: int, cell_in: int, cfg: Config):
        super().__init__()
        self.drug_gnn = DrugGNNEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=2,
            hidden_dim=cfg.drug_hidden,
            z_dim=cfg.z_dim,
            dropout=cfg.dropout,
        )
        self.cell_mlp = MLP(cell_in, cfg.cell_hidden, cfg.z_dim, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.z_dim * 2, cfg.z_dim),
            nn.BatchNorm1d(cfg.z_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.z_dim, cfg.z_dim // 2),
            nn.BatchNorm1d(cfg.z_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.z_dim // 2, 1),
            # ← No Sigmoid: direct ln(IC50) prediction
        )

    def forward(self, drug_graph, x_cell):
        z_drug = self.drug_gnn(drug_graph)
        z_cell = self.cell_mlp(x_cell)
        return self.head(torch.cat([z_drug, z_cell], dim=1))


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred.view(-1) - target.view(-1)) ** 2)).item())


@torch.no_grad()
def pearsonr(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.view(-1).cpu().numpy()
    t = target.view(-1).cpu().numpy()
    if p.std() < 1e-12 or t.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


# ----------------------------
# Graph cache
# ----------------------------
def build_or_load_graph_cache(cfg: Config, smiles_list: List[str]) -> List[Data]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, cfg.graph_cache_file)

    if os.path.exists(cache_path):
        cache_dict = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_dict["meta"].get("n") == len(smiles_list):
            print(f"Loaded Graph cache: {cache_path}")
            return cache_dict["graphs"]
        print("Cache metadata mismatch -> rebuilding...")

    print("Building Graph cache (CPU)...")
    graphs = []
    for i, smi in enumerate(smiles_list):
        graphs.append(smiles_to_graph(smi))
        if (i + 1) % 5000 == 0:
            print(f"  featurized {i+1}/{len(smiles_list)}")

    torch.save({"graphs": graphs, "meta": {"n": len(smiles_list)}}, cache_path)
    print(f"Saved Graph cache: {cache_path}")
    return graphs


# ----------------------------
# Unified 80 / 10 / 10 Split
# ----------------------------
def make_splits(df, cfg: Config):
    idx  = np.arange(len(df))
    seed = cfg.seed

    if cfg.split_type == "random":
        dev_idx, test_idx = train_test_split(
            idx, test_size=cfg.test_size, random_state=seed, shuffle=True)
        train_idx, val_idx = train_test_split(
            dev_idx, test_size=cfg.val_size, random_state=seed, shuffle=True)

    elif cfg.split_type in ("blind_drug", "blind_cell"):
        group_col = ("Drug_ID"        if cfg.split_type == "blind_drug" and "Drug_ID" in df.columns
                     else "Drug"      if cfg.split_type == "blind_drug"
                     else "Cell Line_ID" if "Cell Line_ID" in df.columns
                     else "Cell Line")
        groups = df[group_col].astype(str).to_numpy()

        gss_test = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=seed)
        dev_pos, test_pos = next(gss_test.split(idx, groups=groups))
        dev_idx, test_idx = idx[dev_pos], idx[test_pos]

        dev_groups = groups[dev_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=seed)
        tr_pos, va_pos = next(gss_val.split(dev_idx, groups=dev_groups))
        train_idx, val_idx = dev_idx[tr_pos], dev_idx[va_pos]

        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, \
            f"{cfg.split_type}: overlap train/test!"
        assert len(set(groups[val_idx])   & set(groups[test_idx])) == 0, \
            f"{cfg.split_type}: overlap val/test!"
    else:
        raise ValueError(f"Unknown split_type: {cfg.split_type!r}")

    print(f"[Split:{cfg.split_type}]  "
          f"train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    return train_idx, val_idx, test_idx


# ----------------------------
# Train / Eval / Predict
# ----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for drug_graph, x_cell, y in loader:
        drug_graph = drug_graph.to(device)
        x_cell     = x_cell.to(device)
        y          = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(drug_graph, x_cell), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        total += loss.item() * y.size(0); n += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    preds, targets = [], []
    for drug_graph, x_cell, y in loader:
        drug_graph = drug_graph.to(device)
        x_cell     = x_cell.to(device)
        y          = y.to(device)
        yh = model(drug_graph, x_cell)
        total += loss_fn(yh, y).item() * y.size(0); n += y.size(0)
        preds.append(yh.cpu()); targets.append(y.cpu())
    pred   = torch.cat(preds)
    target = torch.cat(targets)
    return total / max(n, 1), rmse(pred, target), pearsonr(pred, target)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for drug_graph, x_cell, y in loader:
        preds.append(model(drug_graph.to(device), x_cell.to(device)).cpu().view(-1))
        targets.append(y.view(-1))
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


# ----------------------------
# Plotting helpers
# ----------------------------
def save_learning_plots(history, run_dir):
    epochs     = [h["epoch"]       for h in history]
    train_loss = [h["train_loss"]  for h in history]
    val_loss   = [h["val_loss"]    for h in history]
    val_rmse   = [h["val_rmse"]    for h in history]
    val_p      = [h["val_pearson"] for h in history]

    for ys, labels, fname in [
        ([train_loss, val_loss], ["train_loss", "val_loss"], "loss_curve.png"),
        ([val_rmse],             ["val_RMSE"],               "rmse_curve.png"),
        ([val_p],                ["val_Pearson"],            "pearson_curve.png"),
    ]:
        plt.figure()
        for y, lbl in zip(ys, labels):
            plt.plot(epochs, y, label=lbl)
        plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=150); plt.close()


def save_pred_scatter(y_true, y_pred, run_dir, tag="test"):
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    plt.xlabel("y_true (ln IC50)"); plt.ylabel("y_pred (ln IC50)")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"pred_vs_true_{tag}.png"), dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    cfg = Config()

    # ── Change this line to switch split strategy ──────────────────
    cfg.split_type = "blind_cell"   # "random" | "blind_drug" | "blind_cell"
    # ──────────────────────────────────────────────────────────────

    # ── Absolute paths relative to project root ────────────────────
    # Layout: <project_root>/src/baseline_2/baseline2_gine.py
    #         <project_root>/results/baseline_2/<run_id>/
    #         <project_root>/cache/
    _this_file = os.path.abspath(__file__)
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))

    cfg.cache_dir   = os.path.join(_proj_root, "cache")
    cfg.results_dir = os.path.join(_proj_root, "results", "baseline_2")
    # ──────────────────────────────────────────────────────────────

    seed_everything(cfg.seed)
    os.makedirs(cfg.cache_dir,   exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    run_id  = f"{cfg.run_tag}_{cfg.split_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    device  = torch.device(cfg.device)

    print(f"Project root : {_proj_root}")
    print(f"Run dir      : {run_dir}")
    print(f"Cache dir    : {cfg.cache_dir}")
    print(f"Split        : {cfg.split_type}")

    # 1. Load data
    print("Loading FULL GDSC1 dataset...")
    data = DrugRes(name=cfg.dataset_name)
    df   = data.get_data()
    print(f"Total pairs: {len(df):,}")

    # 2. Unified split
    train_idx, val_idx, test_idx = make_splits(df, cfg)

    # 3. Gene selection on TRAIN only
    subset_size = min(10_000, len(train_idx))
    subset_idx  = np.random.choice(train_idx, size=subset_size, replace=False)
    subset_expr = np.array(df.iloc[subset_idx]["Cell Line"].tolist(), dtype=np.float32)
    gene_var    = np.var(subset_expr, axis=0)
    TOP_K       = 1_000
    top_indices = np.sort(np.argsort(gene_var)[-TOP_K:])
    del subset_expr

    # 4. Build cell_expr matrix
    full_expr = df["Cell Line"].tolist()
    n         = len(df)
    cell_expr = np.zeros((n, TOP_K), dtype=np.float32)
    BS = 10_000
    for i in range(0, n, BS):
        batch = np.array(full_expr[i:i+BS], dtype=np.float32)
        cell_expr[i:i+BS] = batch[:, top_indices]
    del full_expr
    import gc; gc.collect()

    # 5. Drug graphs
    smiles      = df["Drug"].astype(str).tolist()
    drug_graphs = build_or_load_graph_cache(cfg, smiles)

    # 6. Target — raw ln(IC50), no normalization
    y = df["Y"].astype(float).to_numpy()

    # 7. Cell expression standardisation (TRAIN stats only)
    tr_mean = cell_expr[train_idx].mean(axis=0, keepdims=True)
    tr_std  = cell_expr[train_idx].std(axis=0,  keepdims=True)
    tr_std[tr_std < 1e-8] = 1.0
    cell_expr = (cell_expr - tr_mean) / tr_std

    # 8. Datasets & loaders
    def make_loader(idx, shuffle):
        ds = DrugResGraphDataset(
            [drug_graphs[i] for i in idx], cell_expr[idx], y[idx])
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle, num_workers=cfg.num_workers)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # 9. Model
    model   = DrugResponsePredictor(node_feat_dim=8, cell_in=TOP_K, cfg=cfg).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    # 10. Training loop
    best_val_rmse = float("inf")
    best_epoch    = -1
    history       = []
    pat           = 0
    best_path     = os.path.join(run_dir, cfg.best_ckpt)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss                   = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_rmse, val_p = eval_epoch(model, val_loader, loss_fn, device)
        sched.step(val_loss)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_RMSE={val_rmse:.4f} | val_Pearson={val_p:.4f}")

        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": val_loss, "val_rmse": val_rmse,
                         "val_pearson": val_p, "lr": opt.param_groups[0]["lr"]})

        if val_rmse < best_val_rmse - cfg.min_delta:
            best_val_rmse = val_rmse; best_epoch = epoch; pat = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, best_path)
        else:
            pat += 1
            if pat >= cfg.patience:
                print(f"Early stopping at epoch {epoch}"); break

    # 11. Save history & plots
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    save_learning_plots(history, run_dir)

    # 12. Load best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # 13. Final evaluation on held-out TEST SET
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    _, test_rmse, test_pearson = eval_epoch(model, test_loader, loss_fn, device)
    print(f"TEST RMSE    (ln IC50) : {test_rmse:.4f}")
    print(f"TEST Pearson           : {test_pearson:.4f}")
    print("="*60)

    y_pred_test, y_true_test = predict(model, test_loader, device)
    save_pred_scatter(y_true_test, y_pred_test, run_dir, tag="test")

    # 14. Metrics
    best_val_pearson = history[best_epoch - 1]["val_pearson"]

    group_col = ("Drug_ID"        if cfg.split_type == "blind_drug" and "Drug_ID" in df.columns
                 else "Drug"      if cfg.split_type == "blind_drug"
                 else "Cell Line_ID" if cfg.split_type == "blind_cell" and "Cell Line_ID" in df.columns
                 else "Cell Line" if cfg.split_type == "blind_cell"
                 else None)
    if group_col:
        groups    = df[group_col].astype(str).to_numpy()
        tr_groups = set(groups[train_idx])
        te_groups = set(groups[test_idx])
        overlap   = len(tr_groups & te_groups)
    else:
        tr_groups = te_groups = set(); overlap = "N/A"

    metrics = {
        "run_id"           : run_id,
        "split_type"       : cfg.split_type,
        "seed"             : cfg.seed,
        "n_samples"        : int(len(df)),
        "n_train"          : int(len(train_idx)),
        "n_val"            : int(len(val_idx)),
        "n_test"           : int(len(test_idx)),
        "n_unique_drugs"   : int(df["Drug"].nunique()),
        "n_train_groups"   : int(len(tr_groups)) if tr_groups else "N/A",
        "n_test_groups"    : int(len(te_groups)) if te_groups else "N/A",
        "group_overlap"    : overlap,
        "best_epoch"       : int(best_epoch),
        "best_val_rmse"    : float(best_val_rmse),
        "best_val_pearson" : float(best_val_pearson),
        "test_rmse"        : float(test_rmse),      # ← κύρια μετρική
        "test_pearson"     : float(test_pearson),   # ← κύρια μετρική
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_cfg_yaml(cfg.__dict__, os.path.join(run_dir, "cfg.yaml"))

    print(f"\nResults saved to: {run_dir}")
    print(f"SUMMARY  |  split={cfg.split_type}  "
          f"test_RMSE={test_rmse:.4f}  test_Pearson={test_pearson:.4f}")


if __name__ == "__main__":
    main()