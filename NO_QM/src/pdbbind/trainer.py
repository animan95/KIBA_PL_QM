import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch as GraphBatch
from torch.utils.tensorboard import SummaryWriter
import csv
import os
from sklearn.model_selection import train_test_split

# --- Your modules ---
from lig_gnn import LigandGCN
from prot_cnn import ProteinCNN
from aff_pred import AffinityPredictor
from graph_utils import mol_to_graph
from seq_utils import encode_sequence

# --- Dataset ---
class BindingDBDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, mol_to_graph, encode_sequence):
        self.df = pd.read_csv(csv_path)
        self.mol_to_graph = mol_to_graph
        self.encode_sequence = encode_sequence

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph = self.mol_to_graph(row["smiles"])
        sequence = self.encode_sequence(row["sequence"])
        label = torch.tensor([row["deltaG"]], dtype=torch.float)
        return graph, sequence, label

# --- Collate function ---
def collate_fn(batch):
    graphs, sequences, labels = zip(*batch)
    graph_batch = GraphBatch.from_data_list(graphs)
    seq_batch = pad_sequence(sequences, batch_first=True)
    y_batch = torch.cat(labels)
    return graph_batch, seq_batch, y_batch

# --- Training ---
def train_one_epoch(model_lig, model_prot, model_pred, loader, optimizer, lossf, device):
    model_lig.train(); model_prot.train(); model_pred.train()
    total_loss = 0.0
    for g, p, y in loader:
        g, p, y = g.to(device), p.to(device), y.to(device)
        l_out = model_lig(g)
        p_out = model_prot(p)
        pred = model_pred(l_out, p_out)
        loss = lossf(pred.squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)

# --- Evaluation ---
def evaluate(model_lig, model_prot, model_pred, loader, device):
    model_lig.eval(); model_prot.eval(); model_pred.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for g, p, y in loader:
            g, p = g.to(device), p.to(device)
            l_out = model_lig(g)
            p_out = model_prot(p)
            pred = model_pred(l_out, p_out)
            all_preds.append(pred.squeeze().cpu())
            all_labels.append(y)
    pred_all = torch.cat(all_preds)
    true_all = torch.cat(all_labels)
    rmse = torch.sqrt(torch.mean((pred_all - true_all)**2)).item()
    return rmse

from rdkit import Chem
# --- Main ---
def main():
    device = torch.device("cpu")
    # Split dataset if needed
    if not all(os.path.exists(f) for f in ["train.csv", "val.csv", "test.csv"]):
        df = pd.read_csv("../../data/pdbbind/bindingdb_ki_filtered.csv")
        train_val, test = train_test_split(df, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1, random_state=42)
        train.to_csv("train.csv", index=False)
        val.to_csv("val.csv", index=False)
        test.to_csv("test.csv", index=False)

    # Load pretrained model
    ligand_model = LigandGCN().to(device)
    protein_model = ProteinCNN().to(device)
    predictor = AffinityPredictor(
        ligand_dim=128,
        protein_dim=128).to(device)

    checkpoint = torch.load("kiba_model.pt", map_location=device)
    ligand_model.load_state_dict(checkpoint["ligand_model"])
    protein_model.load_state_dict(checkpoint["protein_model"])
    predictor.load_state_dict(checkpoint["predictor"])

    # Optional: freeze all but predictor
    freeze_all = True
    if freeze_all:
        for param in ligand_model.parameters(): param.requires_grad = False
        for param in protein_model.parameters(): param.requires_grad = False
        optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(
            list(ligand_model.parameters()) +
            list(protein_model.parameters()) +
            list(predictor.parameters()), lr=1e-4)

    # Data loaders
    train_ds = BindingDBDataset("train.csv", mol_to_graph, encode_sequence)
    val_ds   = BindingDBDataset("val.csv", mol_to_graph, encode_sequence)
    test_ds  = BindingDBDataset("test.csv", mol_to_graph, encode_sequence)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    lossf = nn.MSELoss()

    # Logging
    log_dir = "logs_bindingdb"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    csv_log_path = os.path.join(log_dir, "training_log.csv")
    with open(csv_log_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "train_loss", "val_rmse"])

    for epoch in range(1, 21):
        loss = train_one_epoch(ligand_model, protein_model, predictor, train_loader, optimizer, lossf, device)
        val_rmse = evaluate(ligand_model, protein_model, predictor, val_loader, device)

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("RMSE/val", val_rmse, epoch)

        with open(csv_log_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, loss, val_rmse])

        print(f"[Epoch {epoch}] Train Loss: {loss:.4f} | Val RMSE: {val_rmse:.3f}")

    # Final test evaluation
    test_rmse = evaluate(ligand_model, protein_model, predictor, test_loader, device)
    print(f"\n✅ Test RMSE: {test_rmse:.3f} kcal/mol")
    writer.add_scalar("RMSE/test", test_rmse, 0)

    # Save model
    torch.save({
        "ligand_model": ligand_model.state_dict(),
        "protein_model": protein_model.state_dict(),
        "predictor": predictor.state_dict()
    }, "bindingdb_finetuned.pt")

if __name__ == "__main__":
    main()

