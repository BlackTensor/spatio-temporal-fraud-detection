"""Phase 11.1: Scalability testing.

Goal
----
Measure how inference latency and memory footprint scale as the graph grows
from 1x to ~10x the real Elliptic size, and identify the bottleneck (does cost
grow with the number of nodes, the number of edges, or feature width?).

Why synthetic tiling
--------------------
Elliptic is a fixed dataset, so we cannot download a "10x Elliptic". To create a
larger graph with the *same structural statistics* (degree distribution, feature
distribution, intra-time-step connectivity) we tile k disjoint copies of the real
graph: copy i's node ids are offset by i * num_nodes, and its edges by the same
offset. The result has exactly k x nodes and k x edges and preserves the local
neighbourhood structure each GNN layer sees — which is what governs message-passing
cost. This is the standard way to stress-test full-batch GNN inference.

Reference model
---------------
GraphSAGE (Phase 4 best, test AUC=0.777) — the production scoring model. Its
mean-aggregation forward pass (2x SAGEConv + Linear) is representative of the
message-passing cost shared by every GNN in this repo.

Measurement notes
-----------------
- Latency: mean +/- std of `n_runs` full-graph forward passes after 1 warm-up.
- Memory: we report the *analytic input footprint* (feature tensor +
  edge_index), which dominates and is deterministic, plus the incremental Python
  allocation seen by tracemalloc. PyTorch allocates tensors via its own caching
  allocator (not visible to tracemalloc), so the analytic footprint — not
  tracemalloc — is the figure to trust for working-set planning (matches the
  Phase 8 memory note).
- Bottleneck: we fit latency ~ a + b*nodes and latency ~ a + b*edges and report
  which scales (nodes and edges grow together under tiling, so we also report
  ms-per-1k-nodes stability as the practical headline).

Usage
-----
    python -m src.evaluation.scalability
    python -m src.evaluation.scalability --scales 1 2 5 10 --n-runs 3
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.gnn_models import GraphSAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scalability")

REPO_ROOT     = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR   = REPO_ROOT / "results"

SEED         = 42
GRAPHSAGE_HP = dict(in_channels=169, hidden_channels=64, dropout=0.3)
BYTES_PER_MB = 1024 * 1024


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_graphsage() -> GraphSAGE:
    model = GraphSAGE(**GRAPHSAGE_HP)
    state = torch.load(RESULTS_DIR / "graphsage_model.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def tile_graph(x: torch.Tensor, edge_index: torch.Tensor, k: int):
    """Return (x_k, edge_index_k) = k disjoint copies of the graph.

    k copies => k * N nodes and k * E edges, preserving each node's local
    neighbourhood (so per-node message-passing cost is unchanged).
    """
    if k == 1:
        return x, edge_index
    n = x.size(0)
    x_k = x.repeat(k, 1)                                   # [k*N, F]
    offsets = (torch.arange(k) * n).repeat_interleave(edge_index.size(1))
    edge_k = edge_index.repeat(1, k) + offsets             # [2, k*E]
    return x_k, edge_k


def analytic_footprint_mb(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Input tensor memory (features float32 + edge_index int64), in MB."""
    feat_bytes = x.numel() * x.element_size()
    edge_bytes = edge_index.numel() * edge_index.element_size()
    return (feat_bytes + edge_bytes) / BYTES_PER_MB


@torch.no_grad()
def time_forward(model: GraphSAGE, x: torch.Tensor, edge_index: torch.Tensor,
                 n_runs: int) -> tuple[float, float]:
    model.eval()
    _ = model(x, edge_index)          # warm-up (caches, lazy init)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(x, edge_index)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def measure_scale(model, x_base, edge_base, k: int, n_runs: int) -> dict | None:
    """Build the k-tiled graph, measure latency + memory, then free it."""
    try:
        x_k, edge_k = tile_graph(x_base, edge_base, k)
    except (RuntimeError, MemoryError) as e:           # OOM building the graph
        logger.warning("  scale %dx: could not allocate (%s)", k, e)
        return None

    n_nodes, n_edges = x_k.size(0), edge_k.size(1)
    input_mb = analytic_footprint_mb(x_k, edge_k)

    try:
        tracemalloc.start()
        mean_ms, std_ms = time_forward(model, x_k, edge_k, n_runs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except (RuntimeError, MemoryError) as e:           # OOM during forward
        tracemalloc.stop()
        logger.warning("  scale %dx: forward pass OOM (%s)", k, e)
        del x_k, edge_k
        gc.collect()
        return None

    row = {
        "scale":             k,
        "n_nodes":           int(n_nodes),
        "n_edges":           int(n_edges),
        "input_footprint_mb": round(input_mb, 1),
        "latency_ms_mean":   round(mean_ms, 2),
        "latency_ms_std":    round(std_ms, 2),
        "ms_per_1k_nodes":   round(mean_ms / (n_nodes / 1000.0), 4),
        "tracemalloc_peak_mb": round(peak / BYTES_PER_MB, 2),
    }
    logger.info("  %2dx | %8d nodes | %8d edges | %7.1f MB in | "
                "%8.1f ms (+/- %.1f) | %.3f ms/1k nodes",
                k, n_nodes, n_edges, input_mb, mean_ms, std_ms, row["ms_per_1k_nodes"])

    del x_k, edge_k
    gc.collect()
    return row


def identify_bottleneck(rows: list[dict]) -> dict:
    """Linear fits of latency vs nodes/edges + ms/1k-node drift => bottleneck."""
    if len(rows) < 2:
        return {"verdict": "insufficient data (need >=2 scales)"}

    nodes = np.array([r["n_nodes"] for r in rows], dtype=float)
    edges = np.array([r["n_edges"] for r in rows], dtype=float)
    lat   = np.array([r["latency_ms_mean"] for r in rows], dtype=float)

    # slope (ms per element) and R^2 for each predictor
    def fit(xv):
        A = np.vstack([xv, np.ones_like(xv)]).T
        coef, *_ = np.linalg.lstsq(A, lat, rcond=None)
        pred = A @ coef
        ss_res = float(((lat - pred) ** 2).sum())
        ss_tot = float(((lat - lat.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(coef[0]), float(coef[1]), r2

    node_slope, node_int, node_r2 = fit(nodes)
    edge_slope, edge_int, edge_r2 = fit(edges)

    ms1k = np.array([r["ms_per_1k_nodes"] for r in rows])
    ms1k_drift = float((ms1k.max() - ms1k.min()) / ms1k.mean()) if ms1k.mean() else 0.0

    # Under disjoint tiling nodes and edges scale identically, so the practical
    # statement is whether per-node cost stays flat (linear / well-behaved) or
    # grows (super-linear => memory-bandwidth or allocation bottleneck).
    if ms1k_drift < 0.25:
        scaling = "approximately linear in graph size (cost/node is stable)"
    elif ms1k.argmax() > ms1k.argmin():
        scaling = "super-linear at large scale (cost/node rises => memory-bandwidth bound)"
    else:
        scaling = "sub-linear (fixed overhead amortised at large scale)"

    return {
        "latency_vs_nodes": {"ms_per_node": node_slope, "intercept_ms": node_int, "r2": round(node_r2, 4)},
        "latency_vs_edges": {"ms_per_edge": edge_slope, "intercept_ms": edge_int, "r2": round(edge_r2, 4)},
        "ms_per_1k_nodes_relative_drift": round(ms1k_drift, 4),
        "scaling_verdict": scaling,
        "dominant_memory_term": "node feature tensor (N x 169 float32)",
        "note": ("Full-batch transductive inference holds the entire graph in memory; "
                 "feature tensor dominates footprint and grows linearly with N. "
                 "Message-passing compute grows with E. Under disjoint tiling N and E "
                 "scale together, matching Elliptic's mean degree ~2.3."),
    }


def plot_scaling(rows: list[dict], out_png: Path) -> None:
    nodes = [r["n_nodes"] for r in rows]
    lat   = [r["latency_ms_mean"] for r in rows]
    err   = [r["latency_ms_std"] for r in rows]
    mem   = [r["input_footprint_mb"] for r in rows]
    ms1k  = [r["ms_per_1k_nodes"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].errorbar(nodes, lat, yerr=err, marker="o", capsize=3, color="#1f77b4")
    axes[0].set_xlabel("nodes"); axes[0].set_ylabel("forward latency (ms)")
    axes[0].set_title("Inference latency vs graph size"); axes[0].grid(alpha=0.3)

    axes[1].plot(nodes, mem, marker="s", color="#d62728")
    axes[1].set_xlabel("nodes"); axes[1].set_ylabel("input footprint (MB)")
    axes[1].set_title("Memory footprint vs graph size"); axes[1].grid(alpha=0.3)

    axes[2].plot(nodes, ms1k, marker="^", color="#2ca02c")
    axes[2].set_xlabel("nodes"); axes[2].set_ylabel("ms per 1k nodes")
    axes[2].set_title("Per-node cost (flat = linear scaling)"); axes[2].grid(alpha=0.3)

    fig.suptitle("Phase 11.1 — GraphSAGE full-batch inference scalability (CPU)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_png.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=[1, 2, 5, 10],
                        help="graph-size multipliers to test (disjoint tiling)")
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    torch.set_num_threads(torch.get_num_threads())   # use default thread count

    logger.info("Loading graph.pt …")
    data = torch.load(PROCESSED_DIR / "graph.pt", weights_only=False)
    x_base, edge_base = data.x, data.edge_index
    logger.info("  base graph: %d nodes  %d edges  %d features",
                x_base.size(0), edge_base.size(1), x_base.size(1))

    model = load_graphsage()
    logger.info("Loaded GraphSAGE (Phase 4 best). Threads=%d", torch.get_num_threads())
    logger.info("Measuring scalability over scales %s (%d runs each) …",
                args.scales, args.n_runs)

    rows = []
    for k in sorted(set(args.scales)):
        row = measure_scale(model, x_base, edge_base, k, args.n_runs)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.error("No scale completed — out of memory at every multiplier.")
        return 1

    bottleneck = identify_bottleneck(rows)

    out = {
        "phase": "11.1 scalability",
        "model": "GraphSAGE",
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "n_runs_per_scale": args.n_runs,
        "method": "disjoint k-tiling of the real Elliptic graph (same degree/feature stats)",
        "base_graph": {"n_nodes": int(x_base.size(0)),
                       "n_edges": int(edge_base.size(1)),
                       "n_features": int(x_base.size(1))},
        "measurements": rows,
        "bottleneck": bottleneck,
    }
    out_json = RESULTS_DIR / "scalability_analysis.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved %s", out_json.name)

    plot_scaling(rows, RESULTS_DIR / "scalability.png")

    print("\n=== Phase 11.1 Scalability — summary ===")
    for r in rows:
        print(f"  {r['scale']:>2}x  {r['n_nodes']:>9,} nodes  "
              f"{r['latency_ms_mean']:>9.1f} ms  "
              f"{r['ms_per_1k_nodes']:.3f} ms/1k  "
              f"{r['input_footprint_mb']:>7.1f} MB")
    print(f"  Verdict: {bottleneck.get('scaling_verdict', 'n/a')}")
    print(f"  Output: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
