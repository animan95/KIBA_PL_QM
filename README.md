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
- Output: 135-dimensional embedding

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

## 🧬 Applications

This model and pipeline can be used for:

- **Drug–target interaction prediction**  
  Predicting bioactivity scores for protein–ligand pairs using structure and QM-informed descriptors.

- **Quantum-informed virtual screening**  
  Leveraging HOMO–LUMO gaps and dipole moments to bias ML models toward chemically plausible ligands.

- **Cheminformatics research**  
  Exploring the effects of electronic structure features on binding affinity, enabling hybrid DFT + ML approaches.

- **Benchmarking ligand embeddings**  
  Testing different graph neural network architectures on chemically relevant regression tasks.

- **Modeling electronic interactions in drug discovery**  
  Going beyond topological fingerprints by including quantum-derived fields such as polarity and orbital energies.

## 📚 Citation

If you use this pipeline or its components, please cite the following:

1. **KIBA Dataset**  
   Tang, J., Szwajda, A., Shakyawar, S. et al.  
   *Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets: A Comparative and Integrative Analysis*  
   J. Chem. Inf. Model. 54, 735–743 (2014).  
   [https://doi.org/10.1021/ci400709d](https://doi.org/10.1021/ci400709d)

2. **QMugs Dataset**  
   Isert, C., Atz, K., Jiménez-Luna, J., Schneider, G.  
   *QMugs: Quantum Mechanical Properties of Drug-like Molecules*  
   Mach. Learn.: Sci. Technol. 12, 015004 (2022).  
   [https://doi.org/10.1088/2632-2153/ac58b9](https://doi.org/10.1088/2632-2153/ac58b9)

