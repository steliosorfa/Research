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
from torch_geometric.nn import GINEConv, global_mean_pool as gap, global_max_pool as gmp
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
    run_tag: str = "final_3d_geognn_tuned"
    dataset_name: str = "GDSC1"

    # ── Split strategy ──────────────────────────────────────────────
    # Options: "random" | "blind_drug" | "blind_cell"
    split_type: str = "random"

    # Ratios: 80 / 10 / 10
    test_size: float = 0.10
    val_size: float = 0.111       # 0.111 * 0.9 ≈ 0.10 overall

    # Model dims
    z_dim: int = 128
    drug_hidden: int = 256
    cell_hidden: int = 512
    dropout: float = 0.2 # ΝΕΟ

    # Training — unified across all models
    seed: int = 44
    batch_size: int = 128 #ΝΕΟ
    lr: float = 1e-4 #ΝΕΟ
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 20
    min_delta: float = 1e-5

    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    subset_n: int | None = None

    # Paths — set dynamically in main() based on __file__
    cache_dir: str = ""
    graph_cache_file: str = "gdsc1_geognn_3d_v1.pt"
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
# Featurizer: SMILES -> 3D graph
# ----------------------------
def smiles_to_graph_3d(smiles: str):
    """Returns (drug_atom, drug_bond) — two PyG Data objects."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    try:
        new_mol = Chem.AddHs(mol)
        cids    = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
        res     = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        index   = np.argmin([x[1] for x in res])
        new_mol = Chem.RemoveHs(new_mol)
        conf    = new_mol.GetConformer(id=int(index))
        atom_poses = np.array([list(conf.GetAtomPosition(i))
                                for i in range(new_mol.GetNumAtoms())], dtype=np.float32)
    except:
        AllChem.Compute2DCoords(mol)
        conf    = mol.GetConformer()
        new_mol = mol
        atom_poses = np.array([list(conf.GetAtomPosition(i))
                                for i in range(mol.GetNumAtoms())], dtype=np.float32)

    # Atom features (A2A nodes)
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

    # Bond features + A2A edges
    edges_list = []
    edge_attrs = []

    for bond in new_mol.GetBonds():
        i     = bond.GetBeginAtomIdx()
        j     = bond.GetEndAtomIdx()
        b_len = float(np.linalg.norm(atom_poses[i] - atom_poses[j]))
        b_feat = [
            float(bond.GetIsAromatic()),
            float(bond.GetIsConjugated()),
            float(str(bond.GetBondType()) == 'SINGLE'),
            float(str(bond.GetBondType()) == 'DOUBLE'),
            float(str(bond.GetBondType()) == 'AROMATIC'),
            b_len,
        ]
        edges_list.append([i, j]); edge_attrs.append(b_feat)
        edges_list.append([j, i]); edge_attrs.append(b_feat)

    if len(edges_list) == 0:
        return None, None

    edge_index_atom = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_attr_atom  = torch.tensor(edge_attrs, dtype=torch.float32)

    # B2B graph with bond angles
    def get_angle(vec1, vec2):
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.arccos(np.clip(
            np.dot(vec1 / (n1 + 1e-5), vec2 / (n2 + 1e-5)), -1.0, 1.0)))

    edge_pairs  = edge_index_atom.t().numpy()
    super_edges = []
    bond_angles = []

    for bond_idx, (src, dst) in enumerate(edge_pairs):
        neighbor_bond_ids = np.where(edge_pairs[:, 1] == src)[0]
        for nb_idx in neighbor_bond_ids:
            if nb_idx == bond_idx:
                continue
            nb_src, nb_dst = edge_pairs[nb_idx]
            super_edges.append([nb_idx, bond_idx])
            vec1 = atom_poses[src]   - atom_poses[dst]
            vec2 = atom_poses[nb_src] - atom_poses[nb_dst]
            bond_angles.append(get_angle(vec1, vec2))

    if len(super_edges) == 0:
        return None, None

    edge_index_bond = torch.tensor(super_edges, dtype=torch.long).t().contiguous()
    edge_attr_bond  = torch.tensor(bond_angles, dtype=torch.float32).unsqueeze(1)
    x_bond          = edge_attr_atom   # bond features as B2B node features

    drug_atom = Data(x=x_atom, edge_index=edge_index_atom, edge_attr=edge_attr_atom)
    drug_bond = Data(x=x_bond, edge_index=edge_index_bond, edge_attr=edge_attr_bond)
    return drug_atom, drug_bond


# ----------------------------
# Dataset
# ----------------------------
class DrugResGraphDataset(Dataset):
    def __init__(self, atom_graphs, bond_graphs, cell_expr, y):
        self.atom_graphs = atom_graphs
        self.bond_graphs = bond_graphs
        self.cell_expr   = cell_expr
        self.y           = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.atom_graphs[idx], self.bond_graphs[idx],
                torch.from_numpy(self.cell_expr[idx]),
                torch.tensor(self.y[idx], dtype=torch.float32).view(1))


# ----------------------------
# Geometric Drug Encoder (A2A + B2B GINE)
# ----------------------------
class DrugGeoEncoder(nn.Module):
    def __init__(self, atom_feat_dim: int, bond_feat_dim: int,
                 hidden_dim: int, z_dim: int, dropout: float):
        super().__init__()

        # A2A stream
        self.atom_proj   = nn.Linear(atom_feat_dim, hidden_dim)
        self.edge_proj_a = nn.Linear(bond_feat_dim, hidden_dim)

        self.gin1 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))
        self.gin2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        # B2B stream
        self.bond_node_proj = nn.Linear(bond_feat_dim, hidden_dim)
        self.edge_proj_b    = nn.Linear(1, hidden_dim)

        self.gin_b1 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))
        self.gin_b2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)))

        self.bn_b  = BatchNorm(hidden_dim)
        self.bn_b2 = BatchNorm(hidden_dim)

        self.dropout  = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim * 2, z_dim)

    def forward(self, atom_graph, bond_graph):
        # A2A
        xa = self.atom_proj(atom_graph.x)
        ea = self.edge_proj_a(atom_graph.edge_attr)
        xa = self.dropout(F.relu(self.bn1(self.gin1(xa, atom_graph.edge_index, ea))))
        xa = self.dropout(F.relu(self.bn2(self.gin2(xa, atom_graph.edge_index, ea))))
        h_atom = gmp(xa, atom_graph.batch)

        # B2B
        xb = self.bond_node_proj(bond_graph.x)
        eb = self.edge_proj_b(bond_graph.edge_attr)
        xb = self.dropout(F.relu(self.bn_b(self.gin_b1(xb, bond_graph.edge_index, eb))))
        xb = self.dropout(F.relu(self.bn_b2(self.gin_b2(xb, bond_graph.edge_index, eb))))
        h_bond = gmp(xb, bond_graph.batch)

        h_geo = F.relu(self.out_proj(torch.cat([h_atom, h_bond], dim=1)))
        return h_geo


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
    def __init__(self, cell_in: int, cfg: Config):
        super().__init__()
        self.drug_gnn = DrugGeoEncoder(
            atom_feat_dim=9, bond_feat_dim=6,
            hidden_dim=cfg.drug_hidden, z_dim=cfg.z_dim, dropout=cfg.dropout)
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

    def forward(self, atom_graph, bond_graph, x_cell):
        z_drug = self.drug_gnn(atom_graph, bond_graph)
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
    unique_smiles  = list(set(smiles_list))
    print(f"Found {len(unique_smiles)} unique drugs. Generating 3D conformers...")
    smiles_to_graphs = {}
    failed = 0

    for i, smi in enumerate(unique_smiles):
        da, db = smiles_to_graph_3d(smi)
        smiles_to_graphs[smi] = (da, db)
        if da is None:
            failed += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(unique_smiles)} | failed: {failed}")

    print("Mapping 3D graphs to full dataset...")
    atom_graphs, bond_graphs = [], []
    for smi in smiles_list:
        da, db = smiles_to_graphs[smi]
        atom_graphs.append(da)
        bond_graphs.append(db)

    torch.save({"atom_graphs": atom_graphs, "bond_graphs": bond_graphs,
                "meta": {"n": len(smiles_list)}}, cache_path)
    print(f"Saved cache: {cache_path}")
    return atom_graphs, bond_graphs


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
    for atom_graph, bond_graph, x_cell, y in loader:
        atom_graph = atom_graph.to(device)
        bond_graph = bond_graph.to(device)
        x_cell     = x_cell.to(device)
        y          = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(atom_graph, bond_graph, x_cell), y)
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
    for atom_graph, bond_graph, x_cell, y in loader:
        atom_graph = atom_graph.to(device)
        bond_graph = bond_graph.to(device)
        x_cell     = x_cell.to(device)
        y          = y.to(device)
        yh = model(atom_graph, bond_graph, x_cell)
        total += loss_fn(yh, y).item() * y.size(0); n += y.size(0)
        preds.append(yh.cpu()); targets.append(y.cpu())
    pred   = torch.cat(preds)
    target = torch.cat(targets)
    return total / max(n, 1), rmse(pred, target), pearsonr(pred, target)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for atom_graph, bond_graph, x_cell, y in loader:
        yh = model(atom_graph.to(device), bond_graph.to(device),
                   x_cell.to(device)).cpu().view(-1)
        preds.append(yh); targets.append(y.view(-1))
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
    # Layout: <project_root>/src/3d_baseline/3d_baseline.py
    #         <project_root>/results/3d_baseline/<run_id>/
    #         <project_root>/cache/
    _this_file = os.path.abspath(__file__)
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))

    cfg.cache_dir   = os.path.join(_proj_root, "cache")
    cfg.results_dir = os.path.join(_proj_root, "results", "3d_baseline")
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

    # 2. Filter drugs that fail 3D generation
    print("Filtering drugs that cannot be parsed into 3D graphs...")
    unique_smiles = df["Drug"].unique()
    valid_smiles  = set()
    for smi in unique_smiles:
        da, db = smiles_to_graph_3d(smi)
        if da is not None and db is not None:
            valid_smiles.add(smi)
    initial_len = len(df)
    df = df[df["Drug"].isin(valid_smiles)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"Removed {initial_len - len(df)} pairs | Remaining: {len(df):,}")

    # 3. Unified split
    train_idx, val_idx, test_idx = make_splits(df, cfg)

    # 4. Gene selection on TRAIN only
    subset_size = min(10_000, len(train_idx))
    subset_idx  = np.random.choice(train_idx, size=subset_size, replace=False)
    subset_expr = np.array(df.iloc[subset_idx]["Cell Line"].tolist(), dtype=np.float32)
    gene_var    = np.var(subset_expr, axis=0)
    TOP_K       = 1_000
    top_indices = np.sort(np.argsort(gene_var)[-TOP_K:])
    del subset_expr

    # 5. Build cell_expr matrix
    full_expr = df["Cell Line"].tolist()
    n         = len(df)
    cell_expr = np.zeros((n, TOP_K), dtype=np.float32)
    BS = 10_000
    for i in range(0, n, BS):
        batch = np.array(full_expr[i:i+BS], dtype=np.float32)
        cell_expr[i:i+BS] = batch[:, top_indices]
    del full_expr
    import gc; gc.collect()

    # 6. Cell standardisation (TRAIN stats only)
    tr_mean = cell_expr[train_idx].mean(axis=0, keepdims=True)
    tr_std  = cell_expr[train_idx].std(axis=0,  keepdims=True)
    tr_std[tr_std < 1e-8] = 1.0
    cell_expr = (cell_expr - tr_mean) / tr_std

    # Save preprocessing meta for cross-dataset evaluation
    torch.save({"top_indices": top_indices, "train_mean": tr_mean, "train_std": tr_std},
               os.path.join(run_dir, "gdsc1_preprocessing_meta.pt"))

    # 7. Drug graphs (3D)
    smiles = df["Drug"].astype(str).tolist()
    atom_graphs, bond_graphs = build_or_load_graph_cache(cfg, smiles)

    # 8. Target — raw ln(IC50), no normalization
    y = df["Y"].astype(float).to_numpy()

    # 9. Datasets & loaders
    def make_loader(idx, shuffle):
        ds = DrugResGraphDataset(
            [atom_graphs[i] for i in idx],
            [bond_graphs[i] for i in idx],
            cell_expr[idx], y[idx])
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle, num_workers=cfg.num_workers)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # 10. Model
    model   = DrugResponsePredictor(cell_in=TOP_K, cfg=cfg).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    # 11. Training loop
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

    # 12. Save history & plots
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    save_learning_plots(history, run_dir)

    # 13. Load best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # 14. Final evaluation on held-out TEST SET
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    _, test_rmse, test_pearson = eval_epoch(model, test_loader, loss_fn, device)
    print(f"TEST RMSE    (ln IC50) : {test_rmse:.4f}")
    print(f"TEST Pearson           : {test_pearson:.4f}")
    print("="*60)

    y_pred_test, y_true_test = predict(model, test_loader, device)
    save_pred_scatter(y_true_test, y_pred_test, run_dir, tag="test")
    np.save(os.path.join(run_dir, "test_predictions.npy"), y_pred_test)
    np.save(os.path.join(run_dir, "test_targets.npy"),     y_true_test)

    # Val scatter
    y_pred_val, y_true_val = predict(model, val_loader, device)
    save_pred_scatter(y_true_val, y_pred_val, run_dir, tag="val")

    # 15. Metrics
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
        "test_rmse"        : float(test_rmse),
        "test_pearson"     : float(test_pearson),
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_cfg_yaml(cfg.__dict__, os.path.join(run_dir, "cfg.yaml"))

    print(f"\nResults saved to: {run_dir}")
    print(f"SUMMARY  |  split={cfg.split_type}  "
          f"test_RMSE={test_rmse:.4f}  test_Pearson={test_pearson:.4f}")


if __name__ == "__main__":
    main()