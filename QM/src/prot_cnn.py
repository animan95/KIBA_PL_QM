import torch.nn as nn
import torch

class ProteinCNN(nn.Module):
    def __init__(self, vocab_size=26, embed_dim=50, hidden_dim=128):
        super(ProteinCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, seqs):
        x = self.embedding(seqs).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return x

