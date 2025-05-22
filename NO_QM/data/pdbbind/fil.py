from rdkit import Chem
import pandas as pd

def has_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumBonds() > 0

df = pd.read_csv("pdbbind_ki_with_deltaG.csv")
print("Before filtering:", len(df))
df = df[df["smiles"].apply(has_bonds)]
print("After filtering:", len(df))
df.to_csv("bindingdb_ki_filtered.csv", index=False)
df = pd.read_csv("bindingdb_ki_filtered.csv")
print(df[df["smiles"] == "[Cl-]"])


