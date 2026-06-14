"""Phase 3.1: Build PyG Data object from processed tensors.

Elliptic is a homogeneous transaction graph (single node type).
The original project plan referenced user/account/device/IP node types, which
belong to the IEEE-CIS template.  Elliptic has one node type (Bitcoin
transactions), so we build a PyG ``Data`` (homogeneous) object here.
HeteroData with multiple edge types will be introduced in Phase 6.

Outputs
-------
data/processed/graph.pt     PyG Data object — full transductive graph
data/processed/graph_info.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_graph")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> int:
    logger.info("Loading processed tensors …")
    x = torch.load(PROCESSED_DIR / "node_features.pt", weights_only=True)
    edge_index = torch.load(PROCESSED_DIR / "edge_index.pt", weights_only=True)
    y = torch.load(PROCESSED_DIR / "node_labels.pt", weights_only=True)
    time_step = torch.load(PROCESSED_DIR / "node_times.pt", weights_only=True)

    logger.info("  x          : %s  dtype=%s", tuple(x.shape), x.dtype)
    logger.info("  edge_index : %s  dtype=%s", tuple(edge_index.shape), edge_index.dtype)
    logger.info("  y          : %s  dtype=%s", tuple(y.shape), y.dtype)
    logger.info("  time_step  : %s  dtype=%s", tuple(time_step.shape), time_step.dtype)

    # Load split masks (6 masks: train/val/test node masks + labeled sub-masks)
    splits_pt = torch.load(PROCESSED_DIR / "splits.pt", weights_only=True)
    # splits.pt keys: train_mask, val_mask, test_mask,
    #                 train_labeled_mask, val_labeled_mask, test_labeled_mask
    if isinstance(splits_pt, dict):
        train_mask = splits_pt["train_mask"]
        val_mask = splits_pt["val_mask"]
        test_mask = splits_pt["test_mask"]
        train_labeled_mask = splits_pt["train_labeled_mask"]
        val_labeled_mask = splits_pt["val_labeled_mask"]
        test_labeled_mask = splits_pt["test_labeled_mask"]
    else:
        # Fallback: splits.pt saved as a list in the order above
        keys = ["train_mask", "val_mask", "test_mask",
                "train_labeled_mask", "val_labeled_mask", "test_labeled_mask"]
        train_mask, val_mask, test_mask, \
            train_labeled_mask, val_labeled_mask, test_labeled_mask = splits_pt

    logger.info("Building PyG Data object …")
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=time_step,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        train_labeled_mask=train_labeled_mask,
        val_labeled_mask=val_labeled_mask,
        test_labeled_mask=test_labeled_mask,
        num_nodes=x.shape[0],
    )

    logger.info("Validating graph …")
    assert data.num_nodes == x.shape[0], "Node count mismatch"
    assert data.num_edges == edge_index.shape[1], "Edge count mismatch"
    assert (data.edge_index[0].max() < data.num_nodes), "src out of range"
    assert (data.edge_index[1].max() < data.num_nodes), "dst out of range"
    n_isolated = int((torch.zeros(data.num_nodes, dtype=torch.long)
                      .scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
                      .scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
                      == 0).sum())
    logger.info("  isolated nodes: %d", n_isolated)

    out_path = PROCESSED_DIR / "graph.pt"
    torch.save(data, out_path)
    logger.info("Saved graph.pt")

    info = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": x.shape[1],
        "num_time_steps": int(time_step.unique().numel()),
        "time_step_range": [int(time_step.min()), int(time_step.max())],
        "labels": {
            "illicit": int((y == 1).sum()),
            "licit": int((y == 0).sum()),
            "unknown": int((y == -1).sum()),
        },
        "split_masks": {
            "train_nodes": int(train_mask.sum()),
            "val_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "train_labeled": int(train_labeled_mask.sum()),
            "val_labeled": int(val_labeled_mask.sum()),
            "test_labeled": int(test_labeled_mask.sum()),
        },
        "isolated_nodes": n_isolated,
        "node_type": "transaction (homogeneous — single type)",
        "edge_type": "bitcoin_flow (directed)",
        "note": (
            "Elliptic has one node type (Bitcoin transactions). "
            "HeteroData with multiple edge types is introduced in Phase 6."
        ),
    }
    with open(PROCESSED_DIR / "graph_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n=== Phase 3.1 Graph Build Summary ===")
    print(f"  Nodes       : {data.num_nodes:,}")
    print(f"  Edges       : {data.num_edges:,}")
    print(f"  Features    : {x.shape[1]}")
    print(f"  Time steps  : {info['num_time_steps']}  "
          f"(ts {info['time_step_range'][0]}–{info['time_step_range'][1]})")
    print(f"  Isolated    : {n_isolated}")
    print(f"  Saved       : {out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
