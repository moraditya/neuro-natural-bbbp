# src/featurization.py
from rdkit import Chem
import torch
from torch_geometric.data import Data

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
DEGREE_LIST = [0, 1, 2, 3, 4, 5]

def _one_hot(x, allowable):
    return [int(x == a) for a in allowable]

def atom_to_feature_vector(atom):
    # simple MVP feature vector
    atom_type = _one_hot(atom.GetAtomicNum(), ATOM_LIST) + [int(atom.GetAtomicNum() not in ATOM_LIST)]
    degree = _one_hot(atom.GetDegree(), DEGREE_LIST) + [int(atom.GetDegree() > DEGREE_LIST[-1])]
    formal_charge = atom.GetFormalCharge()
    is_aromatic = int(atom.GetIsAromatic())
    return torch.tensor(atom_type + degree + [formal_charge, is_aromatic], dtype=torch.float)

def smiles_to_data(smiles: str, label: int, idx: int | None = None) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.stack(x_list, dim=0)

    # Edges (undirected)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    if len(edge_index) == 0:
        return None  # skip molecules with no bonds

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.smiles = smiles
    if idx is not None:
        data.idx = idx
    return data

def df_to_pyg_lists(df, smiles_col="smiles", label_col="permeable", split_col="split"):
    train_graphs, test_graphs = [], []

    for idx, row in df.iterrows():
        d = smiles_to_data(row[smiles_col], int(row[label_col]), idx=idx)
        if d is None:
            continue
        if split_col in df.columns and row[split_col] == "test":
            test_graphs.append(d)
        else:
            train_graphs.append(d)

    return train_graphs, test_graphs
