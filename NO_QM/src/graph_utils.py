from rdkit import Chem
from torch_geometric.data import Data
import torch

# One-hot encoders with fallback
def one_hot_encoding(x, allowable_set):
    return [int(x == s) for s in allowable_set]

def atom_features(atom):
    # Hybridization: one-hot for [SP, SP2, SP3, SP3D, SP3D2]
    hyb = one_hot_encoding(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])

    # Chirality: one-hot for [R, S, unspecified]
    chirality = one_hot_encoding(atom.GetChiralTag(), [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
    ])

    features = [
        atom.GetAtomicNum(),                     # atomic number
        atom.GetDegree(),                        # number of bonds
        atom.GetFormalCharge(),                  # formal charge
        atom.GetNumExplicitHs(),                 # explicit Hs
        int(atom.GetIsAromatic()),               # aromatic
        int(atom.IsInRing())                     # ring
    ] + hyb + chirality                          # add one-hot vectors

    return torch.tensor(features, dtype=torch.float)

def bond_features(bond):
    return torch.tensor([
        int(bond.GetBondTypeAsDouble()),       # bond type (1, 2, 3)
        int(bond.GetIsConjugated()),           # conjugated
        int(bond.IsInRing())                   # in ring
    ], dtype=torch.float)

def mol_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # Edge index and features
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bf = bond_features(bond)
        edge_attr += [bf, bf]  # add both directions

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

