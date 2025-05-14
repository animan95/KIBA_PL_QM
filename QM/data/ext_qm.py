import pandas as pd
import tarfile
from rdkit import Chem
from rdkit.Chem import inchi
from io import BytesIO
from tqdm import tqdm

# --- Canonicalize SMILES using InChIKey ---
def to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return inchi.MolToInchiKey(mol) if mol else None
    except:
        return None

# --- Load KIBA data ---
kiba_df = pd.read_csv("kiba_processed.csv")
kiba_df["inchikey"] = kiba_df["SMILES"].apply(to_inchikey)
kiba_keys = set(kiba_df["inchikey"].dropna())

print(f"✔ Loaded {len(kiba_keys)} unique InChIKeys from KIBA")

# --- Extract QM data from QMugs ---
qm_data = []

with tarfile.open("structures.tar.gz", "r:gz") as tar:
    sdf_members = [m for m in tar.getmembers() if m.name.endswith(".sdf")]

    for member in tqdm(sdf_members, desc="Scanning QMugs"):
        try:
            f = tar.extractfile(member)
            if f is None:
                continue
            sdf_bytes = f.read()
            suppl = Chem.ForwardSDMolSupplier(BytesIO(sdf_bytes), removeHs=False)
            for mol in suppl:
                if mol is None:
                    continue
                try:
                    ikey = inchi.MolToInchiKey(mol)
                    if ikey in kiba_keys:
                        props = mol.GetPropsAsDict()
                        gap = props.get("DFT:GAP")
                        # Validate that gap is a float
                        if gap is not None:
                            qm_data.append({
                                "inchikey": ikey,
                                "conformer_id": member.name.split("/")[-1].replace(".sdf", ""),
                                "homo": props.get("DFT:HOMO_ENERGY"),
                                "lumo": props.get("DFT:LUMO_ENERGY"),
                                "gap": float(gap),
                                "dipole": props.get("DFT:DIPOLE")
                            })
                except Exception as e:
                    continue
        except:
            continue

qm_df_all = pd.DataFrame(qm_data)
print(f"✅ Extracted QM properties for {len(qm_df_all)} conformers")

# --- Filter to lowest HOMO–LUMO gap per ligand (inchikey) ---
qm_df_filtered = qm_df_all.sort_values("gap").drop_duplicates(subset="inchikey", keep="first")
qm_df_filtered.to_csv("kiba_with_qm.csv", index=False)
print(f"✅ Saved {len(qm_df_filtered)} ligands with lowest-gap conformer")

