import pandas as pd
import numpy as np
from tdc.multi_pred import DTI

# Constants for ΔG conversion
R = 1.987e-3  # kcal/mol·K
T = 298.15    # K

# Load dataset
data = DTI(name='BindingDB_Ki').get_data()
#data.harmonize_affinities(mode = 'max_affinity')
# Check the structure
print(data.head())
# Drop outliers with extremely weak binding
data = data[data['Y'] > 3]   # pKi > 3  → Ki < 1 mM
data = data[data['Y'] < 12]  # pKi < 12 → Ki > 1 pM

# 'Drug' = SMILES, 'Target' = protein sequence, 'Y' = pKi

data['K_M'] = 10 ** data['Y']       # Convert pKi to Ki (M)
data['deltaG'] = -R * T * np.log(data['K_M'])  # ΔG = -RT ln K
data = data[np.isfinite(data['deltaG'])]
# Rename columns for clarity
data.rename(columns={
    'Drug': 'smiles',
    'Target': 'sequence',
    'Y': 'pKi'
}, inplace=True)

# Save to CSV
output_csv = 'pdbbind_ki_with_deltaG.csv'
data[['smiles', 'sequence', 'pKi', 'deltaG']].to_csv(output_csv, index=False)

print(f"✅ Saved dataset: {output_csv}")
print(f"Total samples: {len(data)}")

