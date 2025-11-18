# test_dataloader.py
import torch
from dataloaders import get_gdsc_loaders

print("=== GDSC1 Local DataLoader Test (Bag-of-Chars + Cell One-Hot) ===\n")

# ---- Loaders ----
train_loader, valid_loader, test_loader, input_dim = get_gdsc_loaders(batch_size=32)

print(f"Input feature dimension (drug-vec + cell-onehot): {input_dim}")

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
drug_dim = len(train_loader.dataset.char2idx)    # number of SMILES characters
cell_dim = len(train_loader.dataset.cell2idx)    # number of unique cell lines

print("\n=== Feature Structure ===")
print(f"Drug feature vector (bag-of-chars) size: {drug_dim}")
print(f"Cell line one-hot vector size:            {cell_dim}")
print(f"Total concatenated feature dim:           {drug_dim + cell_dim}")

# ---- Show the first 10 values of x[0] ----
print("\nx[0][:10] =", x0[:10])

# ---- Dataset sizes ----
print("\n=== Dataset Sizes ===")
print(f"Train size: {len(train_loader.dataset)}")
print(f"Valid size: {len(valid_loader.dataset)}")
print(f"Test size:  {len(test_loader.dataset)}")

print("\n=== Test Completed Successfully ===")
