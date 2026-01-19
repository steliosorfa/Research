"""
Baseline for TDC DrugRes (GDSC1):
Drug SMILES -> ECFP(2048) -> MLP -> z_drug
Cell line gene expression vector -> MLP -> z_cell
Fusion: concat([z_drug, z_cell]) -> head regression -> y_hat

Outputs:
- cache/gdsc1_ecfp2048_radius2.npz  (fingerprint cache)
- results/tdc_drugres_baseline/best_model.pt
- results/tdc_drugres_baseline/metrics.json

Run:
  conda activate tdc-drugres
  python experiments/tdc_drugres_baseline/baseline_gdsc1_mlp.py
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugRes

from sklearn.model_selection import train_test_split


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    dataset_name: str = "GDSC1"

    # Features
    ecfp_bits: int = 2048
    ecfp_radius: int = 2

    # Model dims
    z_dim: int = 256
    drug_hidden: int = 512
    cell_hidden: int = 1024
    dropout: float = 0.1

    # Training
    seed: int = 42
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 15
    val_size: float = 0.1

    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4

    # Runtime
    num_workers: int = 0  # keep 0 on laptop
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (relative to repo root)
    cache_dir: str = "cache"
    fp_cache_file: str = "gdsc1_ecfp2048_radius2.npz"

    results_dir: str = "results/tdc_drugres_baseline"
    best_ckpt: str = "best_model.pt"
    metrics_file: str = "metrics.json"


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Featurizer: SMILES -> ECFP
# ----------------------------
def smiles_to_ecfp(smiles: str, n_bits: int, radius: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


# ----------------------------
# Dataset
# ----------------------------
class DrugResDataset(Dataset):
    def __init__(self, ecfp: np.ndarray, cell_expr: np.ndarray, y: np.ndarray):
        assert ecfp.shape[0] == cell_expr.shape[0] == y.shape[0]
        self.ecfp = ecfp.astype(np.float32)
        self.cell = cell_expr.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_drug = torch.from_numpy(self.ecfp[idx])
        x_cell = torch.from_numpy(self.cell[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32).view(1)
        return x_drug, x_cell, y


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DrugResBaseline(nn.Module):
    def __init__(self, drug_in: int, cell_in: int, cfg: Config):
        super().__init__()
        self.drug_mlp = MLP(drug_in, cfg.drug_hidden, cfg.z_dim, cfg.dropout)
        self.cell_mlp = MLP(cell_in, cfg.cell_hidden, cfg.z_dim, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.z_dim * 2, cfg.z_dim),
            nn.ReLU(),
            nn.Linear(cfg.z_dim, 1),
        )

    def forward(self, x_drug: torch.Tensor, x_cell: torch.Tensor) -> torch.Tensor:
        z_drug = self.drug_mlp(x_drug)
        z_cell = self.cell_mlp(x_cell)
        z = torch.cat([z_drug, z_cell], dim=1)
        return self.head(z)


# ----------------------------
# Metrics
# ----------------------------
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


# ----------------------------
# ECFP cache
# ----------------------------
def build_or_load_ecfp_cache(cfg: Config, smiles_list: List[str]) -> np.ndarray:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, cfg.fp_cache_file)

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        ecfp = data["ecfp"]
        meta = json.loads(data["meta"].tobytes().decode("utf-8"))
        if (
            meta.get("n") == len(smiles_list)
            and meta.get("bits") == cfg.ecfp_bits
            and meta.get("radius") == cfg.ecfp_radius
        ):
            print(f"Loaded ECFP cache: {cache_path}")
            return ecfp
        print("Cache exists but metadata mismatch -> rebuilding...")

    print("Building ECFP cache (CPU)...")
    ecfp = np.zeros((len(smiles_list), cfg.ecfp_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        ecfp[i] = smiles_to_ecfp(smi, cfg.ecfp_bits, cfg.ecfp_radius)
        if (i + 1) % 5000 == 0:
            print(f"  featurized {i+1}/{len(smiles_list)}")

    meta = {"n": len(smiles_list), "bits": cfg.ecfp_bits, "radius": cfg.ecfp_radius}
    np.savez_compressed(cache_path, ecfp=ecfp, meta=json.dumps(meta).encode("utf-8"))
    print(f"Saved ECFP cache: {cache_path}")
    return ecfp


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x_drug, x_cell, y in loader:
        x_drug = x_drug.to(device)
        x_cell = x_cell.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        y_hat = model(x_drug, x_cell)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()

        total += float(loss.item()) * y.size(0)
        n += y.size(0)

    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> Tuple[float, float, float]:
    model.eval()
    total = 0.0
    n = 0
    preds = []
    targets = []
    for x_drug, x_cell, y in loader:
        x_drug = x_drug.to(device)
        x_cell = x_cell.to(device)
        y = y.to(device)

        y_hat = model(x_drug, x_cell)
        loss = loss_fn(y_hat, y)

        total += float(loss.item()) * y.size(0)
        n += y.size(0)

        preds.append(y_hat.detach().cpu())
        targets.append(y.detach().cpu())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return (
    total / max(n, 1),
    rmse(pred, target),
    pearsonr(pred, target),
    )



def main() -> None:
    cfg = Config()
    seed_everything(cfg.seed)

    os.makedirs(cfg.results_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print("Device:", device)

    # Load TDC data
    data = DrugRes(name=cfg.dataset_name)
    df = data.get_data()

    # ---- laptop-safe subset ----
    df = df.sample(n=20000, random_state=cfg.seed).reset_index(drop=True)

    smiles = df["Drug"].astype(str).tolist()
    y = df["Y"].astype(float).to_numpy()

    # Cell Line column is already a list-like expression vector
    cell_expr = np.array(df["Cell Line"].tolist(), dtype=np.float32)
    if cell_expr.ndim != 2:
        raise RuntimeError(f"Cell expr must be 2D; got shape {cell_expr.shape}")

    print("N samples:", len(df))
    print("Cell expr dim:", cell_expr.shape[1])

    # ECFP cache
    ecfp = build_or_load_ecfp_cache(cfg, smiles)

    # Split indices
    idx = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        idx, test_size=cfg.val_size, random_state=cfg.seed, shuffle=True
    )

    # ---- train-only standardization of cell expression (no leakage) ----
    train_mean = cell_expr[idx_train].mean(axis=0, keepdims=True)
    train_std = cell_expr[idx_train].std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0  # avoid div-by-zero

    cell_expr = (cell_expr - train_mean) / train_std

    np.savez_compressed(
    os.path.join(cfg.results_dir, "cell_standardization_stats.npz"),
    mean=train_mean.astype(np.float32),
    std=train_std.astype(np.float32),
)


    train_ds = DrugResDataset(ecfp[idx_train], cell_expr[idx_train], y[idx_train])
    val_ds = DrugResDataset(ecfp[idx_val], cell_expr[idx_val], y[idx_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
    )

    # Model
    model = DrugResBaseline(drug_in=cfg.ecfp_bits, cell_in=cell_expr.shape[1], cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_path = os.path.join(cfg.results_dir, cfg.best_ckpt)

    pat = 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_rmse, val_p = eval_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_RMSE={val_rmse:.4f} | val_Pearson={val_p:.4f}"
        )

        if val_rmse < best_val_rmse - cfg.min_delta:
            best_val_rmse = val_rmse
            pat = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__}, best_path)
        else:
            pat += 1
            if pat >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (no val_RMSE improvement).")
                break

    metrics = {"best_val_rmse": best_val_rmse, "dataset": cfg.dataset_name}
    metrics_path = os.path.join(cfg.results_dir, cfg.metrics_file)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
