"""Phase 3.3: Build per-time-step temporal snapshots.

Each snapshot is a PyG Data object containing only the nodes active at that
time step and the edges whose *both* endpoints are in the same time step.
Node indices are re-mapped to 0-based within each snapshot; the original
global indices are stored as ``global_node_ids``.

This produces the standard Elliptic-style temporal snapshot sequence used
in the original paper and is the input format expected by the snapshot GNN
in Phase 5.

Outputs
-------
data/processed/temporal_snapshots.pt   list of 49 PyG Data objects (ts 1-49)
data/processed/temporal_snapshots_info.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("temporal_snapshots")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def build_snapshot(
    ts: int,
    node_mask: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    time_step: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    train_labeled_mask: torch.Tensor,
    val_labeled_mask: torch.Tensor,
    test_labeled_mask: torch.Tensor,
) -> Data:
    """Build a re-indexed snapshot for time step ts."""
    global_ids = torch.where(node_mask)[0]          # [n_local] global indices
    n_local = global_ids.shape[0]

    # Map global index → local index for edge filtering
    global_to_local = torch.full((x.shape[0],), -1, dtype=torch.long)
    global_to_local[global_ids] = torch.arange(n_local, dtype=torch.long)

    src_global, dst_global = edge_index
    local_src = global_to_local[src_global]
    local_dst = global_to_local[dst_global]
    edge_valid = (local_src >= 0) & (local_dst >= 0)

    local_edge_index = torch.stack([local_src[edge_valid], local_dst[edge_valid]], dim=0)

    return Data(
        x=x[global_ids],
        edge_index=local_edge_index,
        y=y[global_ids],
        time_step=ts,
        global_node_ids=global_ids,
        train_mask=train_mask[global_ids],
        val_mask=val_mask[global_ids],
        test_mask=test_mask[global_ids],
        train_labeled_mask=train_labeled_mask[global_ids],
        val_labeled_mask=val_labeled_mask[global_ids],
        test_labeled_mask=test_labeled_mask[global_ids],
        num_nodes=n_local,
    )


def main() -> int:
    logger.info("Loading graph.pt …")
    data = torch.load(PROCESSED_DIR / "graph.pt", weights_only=False)

    x = data.x
    edge_index = data.edge_index
    y = data.y
    time_step = data.time_step

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    train_labeled_mask = data.train_labeled_mask
    val_labeled_mask = data.val_labeled_mask
    test_labeled_mask = data.test_labeled_mask

    steps = sorted(time_step.unique().tolist())
    logger.info("Building %d snapshots …", len(steps))

    snapshots: list[Data] = []
    info_records: list[dict] = []

    for ts in steps:
        node_mask = (time_step == ts)
        snap = build_snapshot(
            ts, node_mask, x, edge_index, y, time_step,
            train_mask, val_mask, test_mask,
            train_labeled_mask, val_labeled_mask, test_labeled_mask,
        )
        snapshots.append(snap)

        n_illicit = int((snap.y == 1).sum())
        n_licit = int((snap.y == 0).sum())
        info_records.append({
            "time_step": int(ts),
            "n_nodes": snap.num_nodes,
            "n_edges": int(snap.edge_index.shape[1]),
            "n_illicit": n_illicit,
            "n_licit": n_licit,
            "n_unknown": snap.num_nodes - n_illicit - n_licit,
            "in_train": int(snap.train_mask.sum()),
            "in_val": int(snap.val_mask.sum()),
            "in_test": int(snap.test_mask.sum()),
        })

        if int(ts) % 10 == 1:
            logger.info(
                "  ts=%d  nodes=%d  edges=%d  illicit=%d",
                ts, snap.num_nodes, snap.edge_index.shape[1], n_illicit,
            )

    out_path = PROCESSED_DIR / "temporal_snapshots.pt"
    torch.save(snapshots, out_path)
    logger.info("Saved temporal_snapshots.pt  (%d snapshots)", len(snapshots))

    info = {
        "num_snapshots": len(snapshots),
        "time_steps": [int(s) for s in steps],
        "total_nodes_across_snapshots": sum(r["n_nodes"] for r in info_records),
        "total_edges_across_snapshots": sum(r["n_edges"] for r in info_records),
        "snapshots": info_records,
    }
    info_path = PROCESSED_DIR / "temporal_snapshots_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    node_counts = [r["n_nodes"] for r in info_records]
    edge_counts = [r["n_edges"] for r in info_records]

    print("\n=== Phase 3.3 Temporal Snapshots ===")
    print(f"  Snapshots       : {len(snapshots)}")
    print(f"  Nodes / snap    : min={min(node_counts)}  max={max(node_counts)}  "
          f"mean={sum(node_counts)//len(node_counts)}")
    print(f"  Edges / snap    : min={min(edge_counts)}  max={max(edge_counts)}  "
          f"mean={sum(edge_counts)//len(edge_counts)}")
    print(f"  Saved           : {out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
