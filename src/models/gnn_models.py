"""GNN model definitions for Phase 4 static baselines.

All models share the same interface:
    forward(x: Tensor[N, F], edge_index: Tensor[2, E]) -> Tensor[N, 1]  (logits)

Architecture:
    GCN       : GCNConv  → ReLU → Dropout → GCNConv  → ReLU → Dropout → Linear
    GraphSAGE : SAGEConv → ReLU → Dropout → SAGEConv → ReLU → Dropout → Linear
    GAT       : GATConv  → ELU  → Dropout → GATConv  → ELU  → Dropout → Linear
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 heads: int = 2, dropout: float = 0.3):
        super().__init__()
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        head_dim = hidden_channels // heads
        self.conv1 = GATConv(in_channels, head_dim, heads=heads,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels, head_dim, heads=heads,
                             dropout=dropout, concat=True)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


MODEL_REGISTRY: dict[str, type] = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
}


def get_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
