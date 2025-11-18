# dataloaders.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tdc.multi_pred import DrugRes

# ---- Drug encoding: bag-of-characters over SMILES ----

def build_char_vocab(smiles_list):
    chars = sorted({ch for s in smiles_list for ch in s})
    char2idx = {ch: i for i, ch in enumerate(chars)}
    return char2idx

def encode_drug_bag_of_chars(smiles, char2idx, vocab_size):
    vec = np.zeros(vocab_size, dtype=np.float32)
    for ch in smiles:
        if ch in char2idx:
            vec[char2idx[ch]] += 1.0
    if len(smiles) > 0:
        vec /= len(smiles)
    return vec

# ---- Cell encoding: one-hot over "Cell Line" ----

def build_cell_lookup(train_df, cell_col="Cell Line_ID"):
    all_cells = sorted(train_df[cell_col].unique())
    cell2idx = {c: i for i, c in enumerate(all_cells)}
    return cell2idx, len(all_cells)

def encode_cell_one_hot(cell_id, cell2idx, num_cells):
    vec = np.zeros(num_cells, dtype=np.float32)
    idx = cell2idx[cell_id]
    vec[idx] = 1.0
    return vec

# ---- PyTorch Dataset ----

class GDSCDataset(Dataset):
    def __init__(
        self,
        df,
        char2idx,
        cell2idx,
        drug_col="Drug",
        cell_col="Cell Line_ID",
        y_col="Y",
    ):
        self.df = df.reset_index(drop=True)
        self.char2idx = char2idx
        self.vocab_size = len(char2idx)
        self.cell2idx = cell2idx
        self.num_cells = len(cell2idx)
        self.drug_col = drug_col
        self.cell_col = cell_col
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = row[self.drug_col]
        cell_name = row[self.cell_col]
        y = float(row[self.y_col])

        drug_vec = encode_drug_bag_of_chars(smiles, self.char2idx, self.vocab_size)
        cell_vec = encode_cell_one_hot(cell_name, self.cell2idx, self.num_cells)

        x = np.concatenate([drug_vec, cell_vec]).astype(np.float32)

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor

# ---- Helper: build loaders ----

def get_gdsc_loaders(name="GDSC1", batch_size=64):
    data = DrugRes(name=name)
    split = data.get_split(method="random")

    train_df = split["train"]
    valid_df = split["valid"]
    test_df  = split["test"]

    # Build vocab / lookups based on train set
    char2idx = build_char_vocab(train_df["Drug"].tolist())
    cell2idx, num_cells = build_cell_lookup(train_df, cell_col="Cell Line_ID")

    train_ds = GDSCDataset(train_df, char2idx, cell2idx)
    valid_ds = GDSCDataset(valid_df, char2idx, cell2idx)
    test_ds  = GDSCDataset(test_df,  char2idx, cell2idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[0]

    return train_loader, valid_loader, test_loader, input_dim
