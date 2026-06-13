"""Phase 5: Temporal GNN model definitions.

Phase 5.1 — TemporalSnapshotGNN
    GraphSAGE encoder + GRU-based temporal context injection.
    For each snapshot t:
        node_emb_t  = SAGEConv(x_t, edge_index_t)          # [N_t, hidden]
        graph_emb_t = mean_pool(node_emb_t)                 # [hidden]
        h_t         = GRUCell(graph_emb_t, h_{t-1})         # [hidden]
        logits_t    = classifier([node_emb_t | h_t.expand]) # [N_t, 1]

Phase 5.2 — EvolveGCN (EvolveGCN-O style)
    Input projection (169→hidden) followed by two GCN-style layers whose
    weight matrices are evolved by GRUCell at each snapshot:
        W1_t = GRUCell(W1_{t-1}, W1_{t-1})   (W is both input and hidden)
        W2_t = GRUCell(W2_{t-1}, W2_{t-1})
        node_emb_t = EvolveConv2(EvolveConv1(x_t, W1_t), W2_t)
        logits_t   = classifier(node_emb_t)

Both models expose a `forward_snapshot(x, edge_index, state)` interface that
returns `(logits, new_state)`.  State is a simple tuple so the trainer owns
the lifecycle (init, detach, pass-through).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing


# ── EvolveGCN helper ─────────────────────────────────────────────────────────

class _EvolveConv(MessagePassing):
    """GCN-style layer with an externally-supplied weight matrix.

    Uses mean neighbourhood aggregation + self-connection (GraphSAGE-like),
    which avoids the degree-normalisation issues of symmetric GCN on directed
    graphs.
    """

    def __init__(self) -> None:
        super().__init__(aggr="mean")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        W: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        x          : [N, in_channels]
        edge_index : [2, E]
        W          : [out_channels, in_channels]
        bias       : [out_channels]
        returns    : [N, out_channels]
        """
        x_proj = F.linear(x, W, bias)          # [N, out_channels]
        agg    = self.propagate(edge_index, x=x_proj)  # mean of neighbours
        return x_proj + agg                     # self + neighbour

    def message(self, x_j: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x_j


# ── Phase 5.1 ─────────────────────────────────────────────────────────────────

class TemporalSnapshotGNN(nn.Module):
    """Phase 5.1: GraphSAGE + GRU temporal context.

    State: h  — Tensor[1, hidden_channels]
    State is initialised to zeros at the start of each epoch / inference pass
    and evolved by the GRUCell after each snapshot.
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

        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")

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
        """Process one snapshot.

        Returns
        -------
        logits : [N, 1]
        h_new  : [1, hidden]  — new GRU state (detach before next snapshot)
        """
        # GNN encoding
        z = F.relu(self.conv1(x, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.relu(self.conv2(z, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)

        # Graph-level summary for GRU input
        g = z.mean(dim=0, keepdim=True)         # [1, hidden]
        h_new = self.gru(g, h)                   # [1, hidden]

        # Augment node embeddings with temporal context
        ctx = h_new.expand(z.size(0), -1)        # [N, hidden]
        logits = self.classifier(torch.cat([z, ctx], dim=-1))  # [N, 1]

        return logits, h_new


# ── Phase 5.2 ─────────────────────────────────────────────────────────────────

class EvolveGCN(nn.Module):
    """Phase 5.2: EvolveGCN-O — GRU-evolved weight matrices.

    State: (W1, W2)  — both Tensor[hidden, hidden]
    W_t = GRUCell(W_{t-1}, W_{t-1})  (EvolveGCN-O: W is input AND hidden)

    An input projection from in_channels → hidden is applied first so that
    the evolved weight matrices stay at a fixed [hidden, hidden] size,
    keeping parameter count manageable.
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

        self.input_proj = nn.Linear(in_channels, hidden_channels, bias=False)

        self.conv1 = _EvolveConv()
        self.conv2 = _EvolveConv()

        # Learnable initial weight matrices (also used as W_0)
        self.W1_init = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.W2_init = nn.Parameter(torch.empty(hidden_channels, hidden_channels))
        self.b1 = nn.Parameter(torch.zeros(hidden_channels))
        self.b2 = nn.Parameter(torch.zeros(hidden_channels))
        nn.init.xavier_uniform_(self.W1_init)
        nn.init.xavier_uniform_(self.W2_init)

        # GRU cells — each row of W treated as a hidden unit
        # GRUCell(input_size=hidden, hidden_size=hidden)
        self.rnn1 = nn.GRUCell(hidden_channels, hidden_channels)
        self.rnn2 = nn.GRUCell(hidden_channels, hidden_channels)

        self.classifier = nn.Linear(hidden_channels, 1)

    def init_state(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        W1 = self.W1_init.clone()   # keeps requires_grad from Parameter
        W2 = self.W2_init.clone()
        return W1, W2

    def _evolve(
        self,
        W: torch.Tensor,
        rnn: nn.GRUCell,
    ) -> torch.Tensor:
        """Evolve weight matrix W via GRU (EvolveGCN-O: W as both input and hidden)."""
        # W : [hidden, hidden]  → treated as hidden rows, each of size hidden
        return rnn(W, W)   # [hidden, hidden]

    def forward_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process one snapshot.

        Returns
        -------
        logits   : [N, 1]
        new_state: (W1_new, W2_new)
        """
        W1, W2 = state

        # Evolve weight matrices
        W1_new = self._evolve(W1, self.rnn1)
        W2_new = self._evolve(W2, self.rnn2)

        # Input projection
        z = F.relu(self.input_proj(x))          # [N, hidden]

        # Layer 1
        z = F.relu(self.conv1(z, edge_index, W1_new, self.b1))
        z = F.dropout(z, p=self.dropout, training=self.training)

        # Layer 2
        z = F.relu(self.conv2(z, edge_index, W2_new, self.b2))
        z = F.dropout(z, p=self.dropout, training=self.training)

        logits = self.classifier(z)

        return logits, (W1_new, W2_new)


# ── Registry ─────────────────────────────────────────────────────────────────

TEMPORAL_MODEL_REGISTRY: dict[str, type] = {
    "temporal_snapshot_gnn": TemporalSnapshotGNN,
    "evolve_gcn":            EvolveGCN,
}
