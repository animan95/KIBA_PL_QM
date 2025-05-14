import torch.nn as nn
import torch

class AffinityPredictor(nn.Module):
    def __init__(self, ligand_dim=128, protein_dim=128, dropout=0.3):
        super(AffinityPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ligand_dim + protein_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, ligand_vec, protein_vec):
        combined = torch.cat((ligand_vec, protein_vec), dim=1)
        return self.fc(combined)

