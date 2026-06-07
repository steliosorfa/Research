"""
Baseline 1 for TDC DrugRes (GDSC1) — Unified 80/10/10 Split
Drug SMILES -> ECFP(2048) -> MLP -> z_drug
Cell line gene expression -> MLP -> z_cell
Fusion: concat([z_drug, z_cell]) -> head -> y_hat (ln IC50)

Split strategy (controlled by cfg.split_type):
  - "random"     : standard random 80/10/10
  - "blind_drug" : Leave-Drug-Out  80/10/10  (GroupShuffleSplit by Drug_ID)
  - "blind_cell" : Leave-Cell-Out  80/10/10  (GroupShuffleSplit by Cell Line ID)

All three variants share the same seed, preprocessing pipeline,
and test-set evaluation — results are directly comparable.
"""

from __future__ import annotations

import os
import json
import random
import time
import csv
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc.multi_pred import DrugRes

from sklearn.model_selection import train_test_split, GroupShuffleSplit


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


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    run_tag: str = "baseline1_mlp"
    dataset_name: str = "GDSC1"

    # ── Split strategy ──────────────────────────────────────────────
    # Options: "random" | "blind_drug" | "blind_cell"
    split_type: str = "random"

    # Ratios: 80 / 10 / 10
    test_size: float = 0.10       # held-out test  (set aside first)
    val_size: float = 0.111       # val from remaining dev  (0.111 * 0.9 ≈ 0.10 overall)

    # Features
    ecfp_bits: int = 2048
    ecfp_radius: int = 2

    # Model dims
    z_dim: int = 128
    drug_hidden: int = 256
    cell_hidden: int = 256
    dropout: float = 0.4

    # Training
    seed: int = 44                # unified seed across all experiments
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 20
    min_delta: float = 1e-4

    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths — set dynamically in main() based on __file__
    # These are placeholders; do not edit here.
    cache_dir: str = ""
    fp_cache_file: str = "gdsc1_ecfp2048_radius2.npz"
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
        self.ecfp = ecfp.astype(np.float32)
        self.cell = cell_expr.astype(np.float32)
        self.y    = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.ecfp[idx]),
            torch.from_numpy(self.cell[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32).view(1),
        )


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

    def forward(self, x):
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

    def forward(self, x_drug, x_cell):
        z = torch.cat([self.drug_mlp(x_drug), self.cell_mlp(x_cell)], dim=1)
        return self.head(z)


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
# ECFP cache
# ----------------------------
def build_or_load_ecfp_cache(cfg: Config, smiles_list: List[str]) -> np.ndarray:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, cfg.fp_cache_file)

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        meta = json.loads(data["meta"].tobytes().decode("utf-8"))
        if (meta.get("n") == len(smiles_list)
                and meta.get("bits") == cfg.ecfp_bits
                and meta.get("radius") == cfg.ecfp_radius):
            print(f"Loaded ECFP cache: {cache_path}")
            return data["ecfp"]

    print("Building ECFP cache...")
    ecfp = np.zeros((len(smiles_list), cfg.ecfp_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        ecfp[i] = smiles_to_ecfp(smi, cfg.ecfp_bits, cfg.ecfp_radius)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(smiles_list)}")
    meta = {"n": len(smiles_list), "bits": cfg.ecfp_bits, "radius": cfg.ecfp_radius}
    np.savez_compressed(cache_path, ecfp=ecfp,
                        meta=json.dumps(meta).encode("utf-8"))
    print(f"Saved ECFP cache: {cache_path}")
    return ecfp


# ----------------------------
# Unified 80 / 10 / 10 Split
# ----------------------------
def make_splits(df, cfg: Config):
    """
    Returns (train_idx, val_idx, test_idx) as numpy arrays of integer positions
    into df, using the strategy specified by cfg.split_type.

    Ratios: ~80% train | ~10% val | ~10% test
    Seed  : cfg.seed  (same for all experiments → directly comparable)
    """
    idx    = np.arange(len(df))
    seed   = cfg.seed

    if cfg.split_type == "random":
        # ── Random 80/10/10 ────────────────────────────────────────
        dev_idx, test_idx = train_test_split(
            idx, test_size=cfg.test_size, random_state=seed, shuffle=True
        )
        train_idx, val_idx = train_test_split(
            dev_idx, test_size=cfg.val_size, random_state=seed, shuffle=True
        )

    elif cfg.split_type in ("blind_drug", "blind_cell"):
        # ── Group-based 80/10/10 ───────────────────────────────────
        if cfg.split_type == "blind_drug":
            # group by drug → drugs in test never seen during training
            group_col = "Drug_ID" if "Drug_ID" in df.columns else "Drug"
        else:
            # group by cell line → cell lines in test never seen during training
            group_col = "Cell Line_ID" if "Cell Line_ID" in df.columns else "Cell Line"

        groups = df[group_col].astype(str).to_numpy()

        # Step 1: carve out held-out test set (10%)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size,
                                     random_state=seed)
        dev_pos, test_pos = next(gss_test.split(idx, groups=groups))
        dev_idx  = idx[dev_pos]
        test_idx = idx[test_pos]

        # Step 2: split dev → train (≈80%) / val (≈10%)
        dev_groups = groups[dev_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size,
                                    random_state=seed)
        tr_pos, va_pos = next(gss_val.split(dev_idx, groups=dev_groups))
        train_idx = dev_idx[tr_pos]
        val_idx   = dev_idx[va_pos]

        # Sanity-check: zero overlap between splits
        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, \
            f"{cfg.split_type}: overlap between train and test groups!"
        assert len(set(groups[val_idx])   & set(groups[test_idx])) == 0, \
            f"{cfg.split_type}: overlap between val and test groups!"

    else:
        raise ValueError(f"Unknown split_type: {cfg.split_type!r}. "
                         f"Choose 'random', 'blind_drug', or 'blind_cell'.")

    print(f"[Split:{cfg.split_type}]  "
          f"train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    return train_idx, val_idx, test_idx


# ----------------------------
# Train / Eval / Predict
# ----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x_drug, x_cell, y in loader:
        x_drug, x_cell, y = x_drug.to(device), x_cell.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_drug, x_cell), y)
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
        n     += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    preds, targets = [], []
    for x_drug, x_cell, y in loader:
        x_drug, x_cell, y = x_drug.to(device), x_cell.to(device), y.to(device)
        yh = model(x_drug, x_cell)
        total += loss_fn(yh, y).item() * y.size(0)
        n     += y.size(0)
        preds.append(yh.cpu()); targets.append(y.cpu())
    pred   = torch.cat(preds)
    target = torch.cat(targets)
    return total / max(n, 1), rmse(pred, target), pearsonr(pred, target)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x_drug, x_cell, y in loader:
        preds.append(model(x_drug.to(device), x_cell.to(device)).cpu().view(-1))
        targets.append(y.view(-1))
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


# ----------------------------
# Plotting helpers
# ----------------------------
def save_learning_plots(history, run_dir):
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    val_rmse   = [h["val_rmse"]   for h in history]
    val_p      = [h["val_pearson"]for h in history]

    for (ys, label, fname) in [
        ([train_loss, val_loss], ["train_loss","val_loss"], "loss_curve.png"),
        ([val_rmse],             ["val_RMSE"],              "rmse_curve.png"),
        ([val_p],                ["val_Pearson"],           "pearson_curve.png"),
    ]:
        plt.figure()
        for y, lbl in zip(ys, label):
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

    #cfg.split_type = "random"      
    #cfg.split_type = "blind_drug"  
    cfg.split_type = "blind_cell" 

    # ── Absolute paths relative to project root ────────────────────
    # Layout:  <project_root>/src/baseline_1/baseline1.py
    #          <project_root>/results/baseline_1/<run_id>/
    #          <project_root>/cache/
    #
    # os.path.abspath(__file__)  → .../src/baseline_1/baseline1.py
    # dirname × 1               → .../src/baseline_1/
    # dirname × 2               → .../src/
    # dirname × 3               → .../tdc_drugres_baseline/  (project root)
    _this_file  = os.path.abspath(__file__)
    _proj_root  = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))

    cfg.cache_dir   = os.path.join(_proj_root, "cache")
    cfg.results_dir = os.path.join(_proj_root, "results", "baseline_1")
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
    print("Loading GDSC1...")
    df = DrugRes(name=cfg.dataset_name).get_data()
    print(f"Total pairs: {len(df):,}")

    # 2. Unified split  ← single call, same seed everywhere
    train_idx, val_idx, test_idx = make_splits(df, cfg)

    # 3. Gene selection on TRAIN only (prevents leakage)
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

    # 5. ECFP fingerprints
    smiles = df["Drug"].astype(str).tolist()
    ecfp   = build_or_load_ecfp_cache(cfg, smiles)

    # 6. Target vector
    y = df["Y"].astype(float).to_numpy()

    # 7. Standardise cell expression (stats from TRAIN only)
    tr_mean = cell_expr[train_idx].mean(axis=0, keepdims=True)
    tr_std  = cell_expr[train_idx].std(axis=0,  keepdims=True)
    tr_std[tr_std < 1e-8] = 1.0
    cell_expr = (cell_expr - tr_mean) / tr_std

    # 8. Datasets & loaders
    def make_loader(idx, shuffle):
        ds = DrugResDataset(ecfp[idx], cell_expr[idx], y[idx])
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle, num_workers=cfg.num_workers)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # 9. Model
    model    = DrugResBaseline(cfg.ecfp_bits, TOP_K, cfg).to(device)
    opt      = torch.optim.Adam(model.parameters(),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=3)
    loss_fn  = nn.MSELoss()

    # 10. Training loop with early stopping (monitored on VAL)
    best_val_rmse = float("inf")
    best_epoch    = -1
    history       = []
    pat           = 0
    best_path     = os.path.join(run_dir, cfg.best_ckpt)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss              = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_rmse, val_p = eval_epoch(model, val_loader, loss_fn, device)
        sched.step(val_loss)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_RMSE={val_rmse:.4f} | val_Pearson={val_p:.4f}")

        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": val_loss, "val_rmse": val_rmse,
                         "val_pearson": val_p, "lr": opt.param_groups[0]["lr"]})

        if val_rmse < best_val_rmse - cfg.min_delta:
            best_val_rmse = val_rmse
            best_epoch    = epoch
            pat           = 0
            torch.save({"model_state": model.state_dict(),
                        "cfg": cfg.__dict__}, best_path)
        else:
            pat += 1
            if pat >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 11. Save training history & plots
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    save_learning_plots(history, run_dir)

    # 12. Load best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # 13. ── FINAL EVALUATION ON HELD-OUT TEST SET ──────────────────
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

    # Drug/cell overlap stats for logging
    group_col = ("Drug_ID"      if cfg.split_type == "blind_drug"  else
                 "Cell Line_ID" if cfg.split_type == "blind_cell"  else None)
    if group_col and group_col in df.columns:
        groups     = df[group_col].astype(str).to_numpy()
        tr_groups  = set(groups[train_idx])
        te_groups  = set(groups[test_idx])
        overlap    = len(tr_groups & te_groups)
    else:
        tr_groups = te_groups = set(); overlap = "N/A"

    metrics = {
        "run_id"          : run_id,
        "split_type"      : cfg.split_type,
        "seed"            : cfg.seed,
        "n_samples"       : int(len(df)),
        "n_train"         : int(len(train_idx)),
        "n_val"           : int(len(val_idx)),
        "n_test"          : int(len(test_idx)),
        "n_unique_drugs"  : int(df["Drug"].nunique()),
        "n_train_groups"  : int(len(tr_groups)) if tr_groups else "N/A",
        "n_test_groups"   : int(len(te_groups)) if te_groups else "N/A",
        "group_overlap"   : overlap,
        "best_epoch"      : int(best_epoch),
        "best_val_rmse"   : float(best_val_rmse),
        "best_val_pearson": float(best_val_pearson),
        "test_rmse"       : float(test_rmse),        # ← κύρια μετρική
        "test_pearson"    : float(test_pearson),      # ← κύρια μετρική
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_cfg_yaml(cfg.__dict__, os.path.join(run_dir, "cfg.yaml"))

    print(f"\nResults saved to: {run_dir}")
    print(f"SUMMARY  |  split={cfg.split_type}  "
          f"test_RMSE={test_rmse:.4f}  test_Pearson={test_pearson:.4f}")


if __name__ == "__main__":
    main()