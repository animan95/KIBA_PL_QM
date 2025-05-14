import pandas as pd
import pickle
import json
import ast

def extract_sequence(seq_str):
    try:
        seq_dict = ast.literal_eval(seq_str)
        return list(seq_dict.values())[0]  # Take the only value
    except:
        return None
# Load SMILES and sequences
with open("/users/PAS0291/aniketmandal95/DeepDTA/data/kiba/ligands_iso.txt") as f:
    raw = f.read().strip()
    #smiles = [line.strip() for line in f]
# Add braces to turn it into valid JSON

# Parse it
ligand_dict = ast.literal_eval(raw)

# Extract SMILES in order
smiles= list(ligand_dict.values())

print(f"✅ Parsed {len(smiles)} ligands.")
print("First 3 SMILES:", smiles[:3])

with open("/users/PAS0291/aniketmandal95/DeepDTA/data/kiba/proteins.txt") as f:
    sequences = [line.strip() for line in f if line.strip()]

# Load KIBA scores matrix
#affinity = pd.read_csv("/users/PAS0291/aniketmandal95/DeepDTA/data/kiba/Y", sep='\t', header=None)
with open("/users/PAS0291/aniketmandal95/DeepDTA/data/kiba/Y", "rb") as f:
        affinity = pickle.load(f, encoding='latin1')
# Build long-form dataframe
num_ligands, num_proteins = affinity.shape

data = []
for i in range(num_ligands):
    for j in range(num_proteins):
        val = affinity[i, j]
        if isinstance(val, str) and val == 'null':
            continue
        if i >= len(smiles) or j >= len(sequences):
            continue  # skip if data is missing
        data.append({
            "SMILES": smiles[i],
            "Sequence": sequences[j],
            "KIBA_Score": float(val)
        })

df = pd.DataFrame(data)

# Apply fix
df["Sequence"] = df["Sequence"].apply(extract_sequence)

# Drop any failed parses
df = df.dropna(subset=["Sequence"])

# Normalize (min-max)
df['AffinityNorm'] = (df['KIBA_Score'] - df['KIBA_Score'].min()) / (df['KIBA_Score'].max() - df['KIBA_Score'].min())

# Binarize: Label = 1 if strong binder
#df['Label'] = df['KIBA_Score'].apply(lambda x: 1 if x < 12.1 else 0)
# New — allows all numeric values (float64, etc.)
df = df[pd.to_numeric(df["KIBA_Score"], errors='coerce').notna()]
print(df.shape)
print(df.head())
# Save the cleaned dataset
df.to_csv("kiba_processed.csv", index=False)
print("Saved cleaned KIBA dataset as kiba_processed.csv")


