"""
Baseline for TDC DrugRes (GDSC1):
Drug SMILES -> ECFP(2048) -> MLP -> z_drug
Cell line gene expression vector -> MLP -> z_cell
Fusion: concat([z_drug, z_cell]) -> head regression -> y_hat

Outputs (per run):
- results/tdc_drugres_baseline/<run_id>/
    - best_model.pt
    - run_metrics.json
    - history.csv
    - cell_standardization_stats.npz
    - loss_curve.png
    - rmse_curve.png
    - pearson_curve.png
    - pred_vs_true.png

Cache:
- cache/gdsc1_ecfp2048_radius2.npz

Run:
  conda activate tdc-drugres
  python experiments/tdc_drugres_baseline/baseline1.py
"""

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
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugRes

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit


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


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # Experiment tagging
    run_tag: str ="helddrugout"

    dataset_name: str = "GDSC1"

    # Features
    ecfp_bits: int = 2048
    ecfp_radius: int = 2

    # Model dims
    z_dim: int = 128
    drug_hidden: int = 256
    cell_hidden: int = 256
    dropout: float = 0.4

    # Training
    seed: int = 42
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    val_size: float = 0.1

    # Early stopping
    patience: int = 8
    min_delta: float = 1e-4

    # Runtime
    num_workers: int = 4  # keep 0 on laptop
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data subset (laptop-safe); set to None for full
    subset_n: int | None = None

    # Paths (relative to repo root)
    cache_dir: str = "cache"
    fp_cache_file: str = "gdsc1_ecfp2048_radius2.npz"

    results_dir: str = "results/tdc_drugres_baseline"
    best_ckpt: str = "best_model.pt"


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
        return int(self.y.shape[0])

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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
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
# Train / Eval / Predict
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


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x_drug, x_cell, y in loader:
        x_drug = x_drug.to(device)
        x_cell = x_cell.to(device)
        y_hat = model(x_drug, x_cell).detach().cpu().view(-1)
        preds.append(y_hat)
        targets.append(y.detach().cpu().view(-1))
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


# ----------------------------
# Plotting helpers
# ----------------------------
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
    path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_rmse, label="val_RMSE")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(run_dir, "rmse_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_p, label="val_Pearson")
    plt.xlabel("epoch")
    plt.ylabel("Pearson")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(run_dir, "pearson_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()


def save_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, run_dir: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    path = os.path.join(run_dir, "pred_vs_true.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    cfg = Config()
    # OVERRIDE: Use full dataset (memory-safe mode)
    cfg.subset_n = None
    seed_everything(cfg.seed)

    os.makedirs(cfg.results_dir, exist_ok=True)
    run_id = f"{cfg.run_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print(f"Run dir: {run_dir}")

    # 1. Load Data Frame (Pandas is efficient)
    print("Loading FULL GDSC1 dataset...")
    data = DrugRes(name=cfg.dataset_name)
    df = data.get_data()
    print(f"Total samples available: {len(df)}")

    print("Columns:", df.columns.tolist())
    print("n_samples:", len(df))
    print("n_unique Drug:", df["Drug"].nunique())
    if "Drug_ID" in df.columns:
        print("n_unique Drug_ID:", df["Drug_ID"].nunique())


    # ----------------------------
    # LEAVE-DRUG-OUT split (blind drugs)
    # ----------------------------
    idx = np.arange(len(df))
    groups = df["Drug_ID"].astype(str).to_numpy()  # αν δεν υπάρχει, χρησιμοποιω df["Drug"]

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.seed)
    train_idx, val_idx = next(gss.split(idx, groups=groups))

    idx_train = idx[train_idx]
    idx_val = idx[val_idx]

    train_drugs = set(groups[idx_train])
    val_drugs = set(groups[idx_val])
    overlap = train_drugs.intersection(val_drugs)
    print(f"[Split] Train samples: {len(idx_train)} | Val samples: {len(idx_val)}")
    print(f"[Split] Unique drugs train: {len(train_drugs)} | val: {len(val_drugs)} | overlap: {len(overlap)}")
    assert len(overlap) == 0 , "Leave-Drug-Out violated: same Drug_ID appears in train and val!"

    # ----------------------------
    # Variance-based gene selection on TRAIN only
    # ----------------------------
    print("Estimating gene variance using a random TRAIN subset (to save RAM)...")

    train_subset_size = 10000
    train_subset_size = min(train_subset_size, len(idx_train))

    subset_indices = np.random.choice(idx_train, size=train_subset_size, replace=False)
    subset_expr = np.array(df.iloc[subset_indices]["Cell Line"].tolist(), dtype=np.float32)

    gene_variances = np.var(subset_expr, axis=0)

    TOP_K = 1000
    top_indices = np.argsort(gene_variances)[-TOP_K:]
    top_indices = np.sort(top_indices)

    print(f"Selected top {TOP_K} genes based on TRAIN variance.")

    # cleanup
    del subset_expr
    import gc
    gc.collect()

    # ----------------------------
    # Build cell_expr for ALL samples using selected genes
    # ----------------------------
    print("Loading full dataset with selected genes...")

    full_expr_list = df["Cell Line"].tolist()
    n_samples = len(df)

    cell_expr = np.zeros((n_samples, TOP_K), dtype=np.float32)

    batch_size = 10000
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_arr = np.array(full_expr_list[i:end], dtype=np.float32)
        cell_expr[i:end] = batch_arr[:, top_indices]

        if i % 50000 == 0:
            print(f"Processed {i}/{n_samples} samples...")

    print(f"Final Cell expr shape: {cell_expr.shape}")

    del full_expr_list
    gc.collect()


    # 4. Standard Setup continues...
    smiles = df["Drug"].astype(str).tolist()
    y = df["Y"].astype(float).to_numpy()
    
    # ECFP cache
    ecfp = build_or_load_ecfp_cache(cfg, smiles)

    # Split
    # idx = np.arange(len(df))
    # idx_train, idx_val = train_test_split(
        #idx, test_size=cfg.val_size, random_state=cfg.seed, shuffle=True
    #)

    # Standardization
    train_mean = cell_expr[idx_train].mean(axis=0, keepdims=True)
    train_std = cell_expr[idx_train].std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0
    cell_expr = (cell_expr - train_mean) / train_std

    np.savez_compressed(
        os.path.join(run_dir, "cell_standardization_stats.npz"),
        mean=train_mean.astype(np.float32),
        std=train_std.astype(np.float32),
    )

    # Datasets
    train_ds = DrugResDataset(ecfp[idx_train], cell_expr[idx_train], y[idx_train])
    val_ds = DrugResDataset(ecfp[idx_val], cell_expr[idx_val], y[idx_val])

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # Model & Optimizer
    # Note: cell_in is automatically 1000 now
    model = DrugResBaseline(drug_in=cfg.ecfp_bits, cell_in=TOP_K, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3
    )
    
    loss_fn = nn.MSELoss()

    # Training Loop
    best_val_rmse = float("inf")
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    pat = 0
    best_path = os.path.join(run_dir, cfg.best_ckpt)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_rmse, val_p = eval_epoch(model, val_loader, loss_fn, device)
        
        # Step scheduler
        scheduler.step(val_loss)

        current_lr = opt.param_groups[0]["lr"]
        
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_RMSE={val_rmse:.4f} | val_Pearson={val_p:.4f}")
        
        history.append({
            "epoch": epoch, 
            "train_loss": tr_loss, 
            "val_loss": val_loss, 
            "val_rmse": val_rmse, 
            "val_pearson": val_p,
            "lr": current_lr
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

    # Save results
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_learning_plots(history, run_dir)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    y_pred, y_true = predict(model, val_loader, device)
    save_pred_scatter(y_true, y_pred, run_dir)
    
# -------------------------------------------------------
# Final Metrics & Config (Expanded for Git)
# -------------------------------------------------------
    # Calculate best Pearson from history (since we tracked it)
    best_val_pearson = history[best_epoch - 1]["val_pearson"]

    metrics = {
        "run_id": run_id,
        "split_type": "random_mixed",
        "n_samples": int(len(df)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_unique_drugs": int(df["Drug"].nunique()),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
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


if __name__ == "__main__":
    main()



