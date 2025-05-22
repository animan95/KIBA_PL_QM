import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch as GraphBatch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# --- Models ---
from lig_gnn import LigandGCN
from prot_cnn import ProteinCNN
from aff_pred_bay import AffinityPredictor

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

# --- Training ---
def train(model_ligand, model_protein, model_predictor, loader, optimizer, loss_fn):
    model_ligand.train()
    model_protein.train()
    model_predictor.train()
    total_loss = 0
    for graph_batch, seq_batch, targets, qm_batch in loader:
        optimizer.zero_grad()
        ligand_vecs = model_ligand(graph_batch)
        protein_vecs = model_protein(seq_batch)
        ligand_vecs = torch.cat([ligand_vecs, qm_batch.to(ligand_vecs.device)], dim=1)
        preds = model_predictor(ligand_vecs, protein_vecs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.wait = 0
        self.should_stop = False

    def __call__(self, current_score):
        # lower is better (we track RMSE)
        score = -current_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.wait = 0

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
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    #print("True:", y_true[:10])
    #print("Pred:", y_pred[:10])
    return rmse, r2

def train(model_ligand, model_protein, model_predictor, loader, optimizer, loss_fn):
    model_ligand.train()
    model_protein.train()
    model_predictor.train()
    total_loss = 0.0
    for graph_batch, seq_batch, targets, qm_batch in loader:
        optimizer.zero_grad()
        # forward pass
        ligand_vecs = model_ligand(graph_batch)
        protein_vecs = model_protein(seq_batch)
        # concatenate QM features
        ligand_vecs = torch.cat([ligand_vecs, qm_batch.to(ligand_vecs.device)], dim=1)
        preds = model_predictor(ligand_vecs, protein_vecs)

        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# --- Main script ---
def main():
    dataset = KIBADataset("/users/PAS0291/aniketmandal95/prot-lig-QM/data/kiba_qm_cleaned.csv")
    print(f"Total dataset size: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model_ligand = LigandGCN(dropout=0.23350345267318412)
    model_protein = ProteinCNN()
    model_predictor = AffinityPredictor(ligand_dim=135, protein_dim=128, dropout=0.23350345267318412)

    params = list(model_ligand.parameters()) + list(model_protein.parameters()) + list(model_predictor.parameters())
    optimizer = optim.Adam(params, lr=0.00019268808868664345)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=1e-4,
    ) 
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=10, min_delta=0.001)

    best_rmse = float('inf')
    best_path = "kiba_best_qm.pt"

    for epoch in range(1, 101):
       avg_train_loss = train(
            model_ligand, model_protein, model_predictor,
            train_loader, optimizer, loss_fn
          )
       val_rmse, val_r2 = evaluate(
            model_ligand, model_protein, model_predictor,
            val_loader
        )

       print(f"Epoch {epoch:02d} | " f"Train Loss: {avg_train_loss:.4f} | "f"Val RMSE: {val_rmse:.4f} | "f"Val R²: {val_r2:.4f}")

        # step the LR scheduler on validation RMSE
       scheduler.step(val_rmse)

        # checkpoint best model
       if val_rmse < best_rmse:
           best_rmse = val_rmse
           torch.save({
               'ligand_model':  model_ligand.state_dict(),
               'protein_model': model_protein.state_dict(),
               'predictor':     model_predictor.state_dict()
           }, best_path)
       print(f"→ Saved new best model (RMSE {best_rmse:.4f})")


       early_stopper(val_rmse)
       if early_stopper.should_stop:
           print(f"Stopping early at epoch {epoch}")
           break
    #torch.save({
    #    'ligand_model': model_ligand.state_dict(),
    #    'protein_model': model_protein.state_dict(),
    #    'predictor': model_predictor.state_dict()
    #}, "kiba_model_final.pt")

if __name__ == "__main__":
    main()
