import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tdc.multi_pred import DrugRes

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.preprocessing import OneHotEncoder


def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bit_vect, arr)
    return arr.astype(np.float32)


class GDSCDataset(Dataset):
    def __init__(
        self,
        df,
        drug_fp_cache,
        onehot_encoder,
        drug_col="Drug",
        cell_col="Cell Line_ID",
        y_col="Y",
    ):
        self.df = df.reset_index(drop=True)
        self.drug_col = drug_col
        self.cell_col = cell_col
        self.y_col = y_col

        smiles_list = self.df[drug_col].tolist()
        cell_list   = self.df[cell_col].tolist()

        drug_fps = [drug_fp_cache[s] for s in smiles_list]
        self.drug_fps = np.stack(drug_fps, axis=0)

        cell_array = np.array(cell_list).reshape(-1, 1)
        self.cell_onehots = onehot_encoder.transform(cell_array)

        self.X = np.concatenate(
            [self.drug_fps, self.cell_onehots], axis=1
        ).astype(np.float32)
        self.Y = self.df[y_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def get_gdsc_loaders(
    name="GDSC1",
    batch_size=64,
    radius=2,
    n_bits=2048,
):
    data = DrugRes(name=name)
    split = data.get_split(method="random")

    train_df = split["train"]
    valid_df = split["valid"]
    test_df  = split["test"]

    all_smiles = np.unique(
        np.concatenate([
            train_df["Drug"].values,
            valid_df["Drug"].values,
            test_df["Drug"].values,
        ])
    )

    drug_fp_cache = {
        s: smiles_to_ecfp(s, radius=radius, n_bits=n_bits)
        for s in all_smiles
    }

    cell_col = "Cell Line_ID"
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    onehot_encoder.fit(train_df[[cell_col]])

    train_ds = GDSCDataset(train_df, drug_fp_cache, onehot_encoder)
    valid_ds = GDSCDataset(valid_df, drug_fp_cache, onehot_encoder)
    test_ds  = GDSCDataset(test_df,  drug_fp_cache, onehot_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[0]

    return train_loader, valid_loader, test_loader, input_dim
