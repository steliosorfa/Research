import torch
from rdkit import Chem
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

def smiles_to_graph(smiles: str):
    # 1. Initialize Molecule
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    # 2. Extract Atom Features (Nodes)
    # We'll use: [Atomic Number, Degree, IsAromatic]
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            float(atom.GetIsAromatic())
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)

    # 3. Extract Connectivity (Edges)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for an undirected graph
        edges.append([i, j])
        edges.append([j, i])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 4. Wrap in PyTorch Geometric Data object
    return Data(x=x, edge_index=edge_index)

# --- EXECUTION & PRINTING ---
test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
graph_data = smiles_to_graph(test_smiles)

print(f"SMILES: {test_smiles}")
print("-" * 30)
print(f"Graph Representation:")
print(f"Number of Atoms (Nodes): {graph_data.num_nodes}")
print(f"Number of Bonds (Edges): {graph_data.num_edges // 2}") # Divided by 2 because we store both directions
print(f"Node Feature Matrix (X):\n{graph_data.x}")
print(f"Edge Index (Connectivity):\n{graph_data.edge_index}")