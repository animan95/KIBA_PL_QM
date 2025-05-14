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
from aff_pred import AffinityPredictor

# --- Data utils ---
from dataset import KIBADataset
from graph_utils import mol_to_graph
from seq_utils import encode_sequence

def collate_fn(batch):
    graphs, sequences, labels = zip(*batch)
    graph_batch = GraphBatch.from_data_list(graphs)
    sequence_batch = pad_sequence(sequences, batch_first=True)
    label_batch = torch.cat(labels).unsqueeze(1)
    return graph_batch, sequence_batch, label_batch

# --- Training ---
def train(model_ligand, model_protein, model_predictor, loader, optimizer, loss_fn):
    model_ligand.train()
    model_protein.train()
    model_predictor.train()
    total_loss = 0
    for graph_batch, seq_batch, targets in loader:
        optimizer.zero_grad()
        ligand_vecs = model_ligand(graph_batch)
        protein_vecs = model_protein(seq_batch)
        preds = model_predictor(ligand_vecs, protein_vecs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model_ligand, model_protein, model_predictor, loader):
    model_ligand.eval()
    model_protein.eval()
    model_predictor.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for graph_batch, seq_batch, targets in loader:
            ligand_vecs = model_ligand(graph_batch)
            protein_vecs = model_protein(seq_batch)
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

# --- Main script ---
def main():
    dataset = KIBADataset("/users/PAS0291/aniketmandal95/prot-lig/data/kiba_processed.csv")
    print(f"Total dataset size: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model_ligand = LigandGCN()
    model_protein = ProteinCNN()
    model_predictor = AffinityPredictor()

    params = list(model_ligand.parameters()) + list(model_protein.parameters()) + list(model_predictor.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 21):
          model_ligand.train()
          model_protein.train()
          model_predictor.train()

          total_loss = 0
          for graph_batch, seq_batch, targets in train_loader:
              optimizer.zero_grad()
              ligand_vecs = model_ligand(graph_batch)
              protein_vecs = model_protein(seq_batch)
              preds = model_predictor(ligand_vecs, protein_vecs)
              loss = loss_fn(preds, targets)
              loss.backward()
              optimizer.step()
              total_loss += loss.item()

          avg_train_loss = total_loss / len(train_loader)
          val_rmse, val_r2 = evaluate(model_ligand, model_protein, model_predictor, val_loader)

          print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
    torch.save({
        'ligand_model': model_ligand.state_dict(),
        'protein_model': model_protein.state_dict(),
        'predictor': model_predictor.state_dict()
    }, "kiba_model.pt")

if __name__ == "__main__":
    main()
