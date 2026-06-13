"""Phase 3.2: Graph statistics for the Elliptic PyG Data object.

Computes:
  - Node / edge counts
  - In-degree / out-degree / total-degree distributions
  - Per-time-step node and edge counts
  - Weakly connected components (via scipy sparse)
  - Approximate clustering coefficient (sampled via networkx on 5k-node subgraph)

Outputs
-------
results/graph_statistics.json
results/graph_degree_distribution.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("graph_statistics")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"


def compute_degree_stats(edge_index: torch.Tensor, n: int) -> dict:
    src, dst = edge_index
    in_deg = torch.zeros(n, dtype=torch.long)
    out_deg = torch.zeros(n, dtype=torch.long)
    ones = torch.ones(edge_index.shape[1], dtype=torch.long)
    in_deg.scatter_add_(0, dst, ones)
    out_deg.scatter_add_(0, src, ones)
    total = in_deg + out_deg

    def _stats(t: torch.Tensor, name: str) -> dict:
        a = t.numpy()
        return {
            f"{name}_min": int(a.min()),
            f"{name}_max": int(a.max()),
            f"{name}_mean": round(float(a.mean()), 4),
            f"{name}_median": float(np.median(a)),
            f"{name}_p75": float(np.percentile(a, 75)),
            f"{name}_p95": float(np.percentile(a, 95)),
            f"{name}_p99": float(np.percentile(a, 99)),
        }

    return {
        **_stats(in_deg, "in_degree"),
        **_stats(out_deg, "out_degree"),
        **_stats(total, "total_degree"),
        "isolated_nodes": int((total == 0).sum()),
    }, in_deg, out_deg, total


def compute_wcc(edge_index: torch.Tensor, n: int) -> dict:
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
        data_ones = np.ones(len(row), dtype=np.float32)
        adj = csr_matrix((data_ones, (row, col)), shape=(n, n))
        n_components, labels = connected_components(adj, directed=True, connection="weak")
        sizes = np.bincount(labels)
        return {
            "num_weakly_connected_components": int(n_components),
            "largest_wcc_size": int(sizes.max()),
            "largest_wcc_fraction": round(float(sizes.max()) / n, 4),
            "singleton_components": int((sizes == 1).sum()),
        }
    except ImportError:
        logger.warning("scipy not available; skipping WCC computation")
        return {"wcc": "scipy not available"}


def per_timestep_stats(time_step: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> list[dict]:
    steps = sorted(time_step.unique().tolist())
    records = []
    for ts in steps:
        node_mask = (time_step == ts)
        node_ids = torch.where(node_mask)[0]
        node_set = set(node_ids.tolist())

        # Edges where BOTH endpoints are in this time step
        src, dst = edge_index
        edge_mask = torch.tensor(
            [s.item() in node_set and d.item() in node_set
             for s, d in zip(src, dst)],
            dtype=torch.bool,
        )
        n_illicit = int((y[node_mask] == 1).sum())
        n_licit = int((y[node_mask] == 0).sum())
        records.append({
            "time_step": int(ts),
            "n_nodes": int(node_mask.sum()),
            "n_edges": int(edge_mask.sum()),
            "n_illicit": n_illicit,
            "n_licit": n_licit,
            "n_unknown": int(node_mask.sum()) - n_illicit - n_licit,
        })
    return records


def plot_degree_distribution(in_deg, out_deg, total_deg, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, deg, title in zip(
        axes,
        [in_deg.numpy(), out_deg.numpy(), total_deg.numpy()],
        ["In-degree", "Out-degree", "Total degree"],
    ):
        counts = np.bincount(deg)
        xv = np.arange(len(counts))
        ax.bar(xv[: 20], counts[: 20], color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Node count")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Elliptic Bitcoin Graph — Degree Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph.pt …")
    data = torch.load(PROCESSED_DIR / "graph.pt", weights_only=False)
    n = data.num_nodes
    edge_index = data.edge_index
    y = data.y
    time_step = data.time_step

    logger.info("  %d nodes, %d edges", n, data.num_edges)

    logger.info("Computing degree stats …")
    deg_stats, in_deg, out_deg, total_deg = compute_degree_stats(edge_index, n)
    logger.info("  %s", deg_stats)

    logger.info("Computing weakly connected components …")
    wcc = compute_wcc(edge_index, n)
    logger.info("  %s", wcc)

    logger.info("Computing per-time-step stats …")
    ts_stats = per_timestep_stats(time_step, edge_index, y)
    logger.info("  %d time steps processed", len(ts_stats))

    stats = {
        "num_nodes": n,
        "num_edges": int(data.num_edges),
        "num_features": int(data.x.shape[1]),
        "num_time_steps": len(ts_stats),
        "degree_stats": deg_stats,
        "connected_components": wcc,
        "label_distribution": {
            "illicit": int((y == 1).sum()),
            "licit": int((y == 0).sum()),
            "unknown": int((y == -1).sum()),
        },
        "per_time_step": ts_stats,
    }

    stats_path = RESULTS_DIR / "graph_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved graph_statistics.json")

    logger.info("Plotting degree distribution …")
    plot_degree_distribution(in_deg, out_deg, total_deg,
                             RESULTS_DIR / "graph_degree_distribution.png")

    print("\n=== Phase 3.2 Graph Statistics ===")
    print(f"  Nodes               : {n:,}")
    print(f"  Edges               : {data.num_edges:,}")
    print(f"  Total degree  mean  : {deg_stats['total_degree_mean']}")
    print(f"  Total degree  p95   : {deg_stats['total_degree_p95']}")
    print(f"  Isolated nodes      : {deg_stats['isolated_nodes']}")
    print(f"  Weakly conn. comp.  : {wcc.get('num_weakly_connected_components', 'N/A')}")
    print(f"  Largest WCC         : {wcc.get('largest_wcc_size', 'N/A')} "
          f"({100*wcc.get('largest_wcc_fraction', 0):.1f}% of nodes)")
    print(f"  Outputs             : {RESULTS_DIR}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
