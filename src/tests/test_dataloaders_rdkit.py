
from dataloaders import get_gdsc_loaders
import torch

print("=== GDSC1 Colab DataLoader Test (RDKit ECFP + Cell One-Hot) ===\n")

# ---- Loaders ----
train_loader, valid_loader, test_loader, input_dim = get_gdsc_loaders(batch_size=32)

print(f"Input feature dimension (ECFP drug-fp + cell-onehot): {input_dim}")

# ---- Get one batch ----
x_batch, y_batch = next(iter(train_loader))
print(f"\nBatch X shape: {x_batch.shape}   (batch_size, feature_dim)")
print(f"Batch Y shape: {y_batch.shape}   (batch_size)")

# ---- Inspect first sample ----
x0 = x_batch[0]
y0 = y_batch[0]

print("\n=== First sample (x[0]) ===")
print(f"x[0] shape: {x0.shape}")
print(f"y[0] value: {y0.item():.4f}")

# ---- Explain structure ----
train_ds = train_loader.dataset
drug_dim = train_ds.drug_fps.shape[1]        # μέγεθος ECFP fingerprint
cell_dim = train_ds.cell_onehots.shape[1]    # μέγεθος one-hot για cell line

print("\n=== Feature Structure ===")
print(f"Drug feature vector (RDKit ECFP) size: {drug_dim}")
print(f"Cell line one-hot vector size:         {cell_dim}")
print(f"Total concatenated feature dim:        {drug_dim + cell_dim}")

# ---- Show the first 10 values of x[0] ----
print("\nx[0][:10] =", x0[:10])

# ---- Dataset sizes ----
print("\n=== Dataset Sizes ===")
print(f"Train size: {len(train_loader.dataset)}")
print(f"Valid size: {len(valid_loader.dataset)}")
print(f"Test size:  {len(test_loader.dataset)}")

print("\n=== RDKit DataLoader Test Completed Successfully ===")
