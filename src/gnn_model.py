# src/gnn_model.py
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

class BBBP_GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)  # 2 classes: non-permeable (0), permeable (1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        out = self.lin(x)               # logits
        return out
