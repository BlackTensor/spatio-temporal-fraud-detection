"""Phase 6: Heterogeneous GNN model definitions.

Heterogeneity is constructed from Elliptic's directed edges by splitting them
into two typed relations:

  'sends'    : A → B  (A funded B; B receives info from predecessors)
  'receives' : B → A  (reverse; A receives info from successors)

Each relation gets its own convolution with independent parameters, giving the
model asymmetric inductive bias for the directed fraud graph.  The two
per-direction outputs are **summed** before the activation, keeping the output
dimension identical to the homogeneous baselines (Phase 4/5) for a fair
parameter-count comparison.

Models
------
HeteroSAGE        : 2× SAGEConv per layer (sends + receives), summed  — Phase 6.1a
HeteroGAT (HGAT)  : 2× GATConv per layer (sends + receives), summed  — Phase 6.1
HeteroTemporalGNN : HeteroSAGE encoder + GRU temporal context         — Phase 6.2

All static models expose:
    forward(x, edge_index) → Tensor[N, 1]   (logits)

The temporal model exposes:
    init_state(device) → h
    forward_snapshot(x, edge_index, h) → (logits, h_new)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


def _rev(edge_index: torch.Tensor) -> torch.Tensor:
    """Return the reversed edge index (swap source ↔ destination rows)."""
    return edge_index[[1, 0]]


# ── Phase 6.1a: HeteroSAGE ───────────────────────────────────────────────────

class HeteroSAGE(nn.Module):
    """Static: separate SAGEConv per edge direction, summed per layer.

    Direct heterogeneous analog of GraphSAGE (Phase 4.3).
    Parameter count ≈ 2× GraphSAGE.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.conv1_fwd = SAGEConv(in_channels,      hidden_channels, aggr="mean")
        self.conv1_rev = SAGEConv(in_channels,      hidden_channels, aggr="mean")
        self.conv2_fwd = SAGEConv(hidden_channels,  hidden_channels, aggr="mean")
        self.conv2_rev = SAGEConv(hidden_channels,  hidden_channels, aggr="mean")
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        rev = _rev(edge_index)
        z = F.relu(self.conv1_fwd(x, edge_index) + self.conv1_rev(x, rev))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.conv2_fwd(z, edge_index) + self.conv2_rev(z, rev))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z)


# ── Phase 6.1: HeteroGAT (HGAT) ─────────────────────────────────────────────

class HeteroGAT(nn.Module):
    """Static: separate GATConv per edge direction, summed per layer (HGAT).

    Direct heterogeneous analog of GAT (Phase 4.4).
    Parameter count ≈ 2× GAT.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        head_dim = hidden_channels // heads
        self.dropout = dropout

        self.conv1_fwd = GATConv(in_channels,     head_dim, heads=heads,
                                  dropout=dropout, concat=True)
        self.conv1_rev = GATConv(in_channels,     head_dim, heads=heads,
                                  dropout=dropout, concat=True)
        self.conv2_fwd = GATConv(hidden_channels, head_dim, heads=heads,
                                  dropout=dropout, concat=True)
        self.conv2_rev = GATConv(hidden_channels, head_dim, heads=heads,
                                  dropout=dropout, concat=True)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        rev = _rev(edge_index)
        z = F.elu(self.conv1_fwd(x, edge_index) + self.conv1_rev(x, rev))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.elu(self.conv2_fwd(z, edge_index) + self.conv2_rev(z, rev))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z)


# ── Phase 6.2: HeteroTemporalGNN (HTGN) ─────────────────────────────────────

class HeteroTemporalGNN(nn.Module):
    """Phase 6.2: HeteroSAGE encoder + GRU temporal context (HTGN).

    Combines:
      - Direction-typed message passing (heterogeneous, from Phase 6.1)
      - Graph-level GRU temporal state (same as TemporalSnapshotGNN, Phase 5.1)

    State: h — Tensor[1, hidden_channels]  (initialised to zeros each epoch)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden = hidden_channels
        self.dropout = dropout

        self.conv1_fwd = SAGEConv(in_channels,     hidden_channels, aggr="mean")
        self.conv1_rev = SAGEConv(in_channels,     hidden_channels, aggr="mean")
        self.conv2_fwd = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.conv2_rev = SAGEConv(hidden_channels, hidden_channels, aggr="mean")

        self.gru = nn.GRUCell(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels * 2, 1)

    def init_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, self.hidden, device=device)

    def forward_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one temporal snapshot.

        Returns
        -------
        logits : [N, 1]
        h_new  : [1, hidden]  — detach before next snapshot (TBPTT-1)
        """
        rev = _rev(edge_index)

        z = F.relu(self.conv1_fwd(x, edge_index) + self.conv1_rev(x, rev))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.conv2_fwd(z, edge_index) + self.conv2_rev(z, rev))
        z = F.dropout(z, p=self.dropout, training=self.training)

        g     = z.mean(dim=0, keepdim=True)           # [1, hidden]
        h_new = self.gru(g, h)                         # [1, hidden]

        ctx    = h_new.expand(z.size(0), -1)           # [N, hidden]
        logits = self.classifier(torch.cat([z, ctx], dim=-1))  # [N, 1]
        return logits, h_new


# ── Registry ─────────────────────────────────────────────────────────────────

HETERO_MODEL_REGISTRY: dict[str, type] = {
    "hetero_sage": HeteroSAGE,
    "hgat":        HeteroGAT,
    "htgn":        HeteroTemporalGNN,
}

STATIC_MODELS   = {"hetero_sage", "hgat"}
TEMPORAL_MODELS = {"htgn"}
