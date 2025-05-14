import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

class LigandGCN(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, dropout=0.2):
        super(LigandGCN, self).__init__()

        # Feedforward MLPs for GINConv
        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Two GIN convolution layers
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Pool node features into graph embedding
        x = global_mean_pool(x, batch)

        x = self.dropout(x)
        return self.out(x)

