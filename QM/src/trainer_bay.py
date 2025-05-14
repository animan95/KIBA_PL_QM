import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch as GraphBatch
from sklearn.metrics import mean_squared_error
from math import sqrt

# --- Models ---
from lig_gnn import LigandGCN
from prot_cnn import ProteinCNN
from aff_pred import AffinityPredictor

# --- Data utils ---
from dataset import KIBADataset
from graph_utils import mol_to_graph
from seq_utils import encode_sequence

def collate_fn(batch):
    graphs, sequences, labels, qms = zip(*batch)
    graph_batch = GraphBatch.from_data_list(graphs)
    sequence_batch = pad_sequence(sequences, batch_first=True)
    label_batch = torch.cat(labels).unsqueeze(1)
    qm_batch = torch.stack(qms)
    return graph_batch, sequence_batch, label_batch, qm_batch

def evaluate(model_ligand, model_protein, model_predictor, loader):
    model_ligand.eval()
    model_protein.eval()
    model_predictor.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for graph_batch, seq_batch, targets, qm_batch in loader:
            ligand_vecs = model_ligand(graph_batch)
            protein_vecs = model_protein(seq_batch)
            ligand_vecs = torch.cat([ligand_vecs, qm_batch.to(ligand_vecs.device)], dim=1)
            preds = model_predictor(ligand_vecs, protein_vecs)
            all_preds.append(preds.squeeze().cpu())
            all_targets.append(targets.squeeze().cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def objective(trial):
    dataset = KIBADataset("/users/PAS0291/aniketmandal95/prot-lig-QM/data/kiba_with_qm.csv")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model_ligand = LigandGCN()
    model_protein = ProteinCNN()
    model_predictor = AffinityPredictor(ligand_dim=135, protein_dim=128, dropout=dropout)

    params = list(model_ligand.parameters()) + list(model_protein.parameters()) + list(model_predictor.parameters())
    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(10):  # fewer epochs for faster tuning
        model_ligand.train()
        model_protein.train()
        model_predictor.train()
        for graph_batch, seq_batch, targets, qm_batch in train_loader:
            optimizer.zero_grad()
            ligand_vecs = model_ligand(graph_batch)
            protein_vecs = model_protein(seq_batch)
            ligand_vecs = torch.cat([ligand_vecs, qm_batch.to(ligand_vecs.device)], dim=1)
            preds = model_predictor(ligand_vecs, protein_vecs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

    val_rmse = evaluate(model_ligand, model_protein, model_predictor, val_loader)
    return val_rmse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best parameters:", study.best_params)
# After tuning is complete
best_params = study.best_params
print("Best parameters found:", best_params)

# Rebuild the final model with best params
batch_size = best_params["batch_size"]
hidden_dim = best_params["hidden_dim"]
dropout = best_params["dropout"]
lr = best_params["lr"]

# Full training with best config
dataset = KIBADataset("/users/PAS0291/aniketmandal95/prot-lig-QM/data/kiba_with_qm.csv")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

model_ligand = LigandGCN()
model_protein = ProteinCNN()
model_predictor = AffinityPredictor(ligand_dim=135, protein_dim=128, dropout=dropout)

params = list(model_ligand.parameters()) + list(model_protein.parameters()) + list(model_predictor.parameters())
optimizer = optim.Adam(params, lr=lr)
loss_fn = nn.MSELoss()

for epoch in range(30):  # full training for more epochs
    model_ligand.train()
    model_protein.train()
    model_predictor.train()
    for graph_batch, seq_batch, targets, qm_batch in train_loader:
        optimizer.zero_grad()
        ligand_vecs = model_ligand(graph_batch)
        protein_vecs = model_protein(seq_batch)
        ligand_vecs = torch.cat([ligand_vecs, qm_batch.to(ligand_vecs.device)], dim=1)
        preds = model_predictor(ligand_vecs, protein_vecs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

# Evaluate and save final model
val_rmse, val_r2 = evaluate(model_ligand, model_protein, model_predictor, val_loader)
print(f"Final model RMSE on validation: {val_rmse:.4f}|R2 val: {val_r2:.4f}")

torch.save({
    'ligand_model': model_ligand.state_dict(),
    'protein_model': model_protein.state_dict(),
    'predictor': model_predictor.state_dict(),
    'best_params': best_params
}, "kiba_final_model.pt")

