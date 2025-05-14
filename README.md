# Protein–Ligand Binding Affinity Prediction with Quantum Features

This project implements a deep learning pipeline to predict binding affinities between ligands and proteins using both molecular graph information and quantum mechanical descriptors.

---

## 🧪 Dataset

We use the **KIBA** dataset, enriched with quantum mechanical (QM) descriptors from **QMugs** or similar sources.

### Dataset Features

- **SMILES** strings for ligands
- **Amino acid sequences** for proteins
- **KIBA scores** as regression targets
- **Quantum features**:
  - `homo`: HOMO energy
  - `lumo`: LUMO energy
  - `gap`: Computed as `lumo - homo`
  - `dipole_x`, `dipole_y`, `dipole_z`: Dipole vector components
  - `dipole_total`: Dipole moment magnitude

---

## 🧠 Model Architecture

This project uses three modules:

### 1. Ligand Encoder (`LigandGCN`)
- Graph Neural Network (GINConv)
- Converts ligand SMILES to molecular graphs
- Output: 128-dimensional embedding

### 2. Protein Encoder (`ProteinCNN`)
- 1D Convolutional Neural Network
- Converts amino acid sequences into embeddings
- Output: 128-dimensional embedding

### 3. Affinity Predictor (`AffinityPredictor`)
- Multilayer perceptron
- Input: `[ligand_vec || QM features || protein_vec]`
- Output: Scalar prediction (binding affinity)

---

## ⚙️ Features

- ✅ Graph-based ligand representation using PyTorch Geometric
- ✅ Sequence-based protein encoding using CNNs
- ✅ Quantum descriptors integrated as additional features
- ✅ Supports custom datasets with SMILES, sequences, and QM features
- ✅ End-to-end PyTorch training with evaluation on RMSE and R²
- ✅ Automatic hyperparameter tuning with Optuna

---

## 🔍 Hyperparameter Tuning (Optuna)

We tune the following parameters:
- `learning_rate`: sampled logarithmically
- `dropout`: between 0.2 and 0.5
- `hidden_dim`: {128, 256, 512}
- `batch_size`: {32, 64, 128}

### Run tuning:
```bash
python optuna_tuning_kiba.py
