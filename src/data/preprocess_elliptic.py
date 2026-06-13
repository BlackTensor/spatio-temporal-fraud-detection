"""Phase 1.3: Data Cleaning & Preprocessing for the Elliptic Bitcoin dataset.

Steps
-----
1. Load the three raw CSVs.
2. Validate entity IDs: every node referenced in edges must exist in features.
3. Handle missing timestamps / duplicates (confirmed none in EDA; still checked
   programmatically so the script is self-contained and re-runnable).
4. Remove duplicate directed edges (keep first occurrence in CSV order).
5. Filter ultra-low-degree nodes: report counts; keep all (each has 165 valid
   features and provides graph context — no isolated junk nodes found).
6. Build a deterministic integer ID mapping: row order in elliptic_txs_features.csv
   becomes the canonical node index (preserves the natural time-ordered grouping
   of the published dataset).
7. Encode labels: illicit (class=1) → 1, licit (class=2) → 0, unknown → -1.
8. Save processed tensors + ID map to data/processed/.

Outputs
-------
data/processed/
    id_map.json            {str(txId): int_index}  (used by all later phases)
    node_features.pt       float32 tensor [N, 165]
    node_times.pt          int32  tensor [N]  (time step 1-49)
    node_labels.pt         int64  tensor [N]  (1=illicit, 0=licit, -1=unknown)
    edge_index.pt          int64  tensor [2, E]  (deduplicated directed edges)
    preprocessing_report.json

Usage
-----
    python -m src.data.preprocess_elliptic
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess_elliptic")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "elliptic"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_FEATURES = 165


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, N_FEATURES + 1)]
    features = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None, names=feat_cols)
    classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")
    edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
    return features, classes, edges


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict:
    report: dict = {}
    node_ids = set(features["txId"])

    # ---- missing timestamps ----
    missing_ts = int(features["time_step"].isna().sum())
    report["missing_timestamps"] = missing_ts
    if missing_ts:
        logger.warning("  %d nodes have missing time_step — will be assigned ts=-1", missing_ts)

    # ---- unknown txIds in classes ----
    cls_ids = set(classes["txId"])
    extra_cls = cls_ids - node_ids
    report["class_ids_not_in_features"] = len(extra_cls)
    if extra_cls:
        logger.warning("  %d class entries reference unknown txIds (will be dropped)", len(extra_cls))

    # ---- edge nodes not in features ----
    edge_nodes = set(edges["txId1"]) | set(edges["txId2"])
    dangling = edge_nodes - node_ids
    report["dangling_edge_nodes"] = len(dangling)
    if dangling:
        logger.warning(
            "  %d edge nodes not found in features — those edges will be dropped", len(dangling)
        )

    # ---- duplicate edges ----
    dup = int(edges.duplicated(subset=["txId1", "txId2"]).sum())
    report["duplicate_directed_edges"] = dup

    # ---- duplicate node IDs in features ----
    dup_nodes = int(features["txId"].duplicated().sum())
    report["duplicate_node_ids"] = dup_nodes
    if dup_nodes:
        logger.warning("  %d duplicate txIds in features — keeping first occurrence", dup_nodes)

    logger.info("Validation: %s", report)
    return report


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_ids = set(features["txId"])

    # Drop duplicate node rows (keep first — deterministic with row-order ID map)
    if features["txId"].duplicated().any():
        features = features.drop_duplicates(subset="txId", keep="first")
        logger.info("  dropped duplicate node rows → %d nodes remain", len(features))

    # Fill missing time steps with -1 sentinel (none expected)
    if features["time_step"].isna().any():
        features["time_step"] = features["time_step"].fillna(-1).astype(int)

    # Drop dangling edges (both endpoints must be in features)
    mask = edges["txId1"].isin(node_ids) & edges["txId2"].isin(node_ids)
    dropped = int((~mask).sum())
    if dropped:
        edges = edges[mask].copy()
        logger.info("  dropped %d dangling edges → %d remain", dropped, len(edges))

    # Remove duplicate directed edges (keep first in CSV order)
    before = len(edges)
    edges = edges.drop_duplicates(subset=["txId1", "txId2"], keep="first")
    removed = before - len(edges)
    if removed:
        logger.info("  removed %d duplicate edges", removed)

    return features, classes, edges


# ---------------------------------------------------------------------------
# Build integer ID map
# ---------------------------------------------------------------------------

def build_id_map(features: pd.DataFrame) -> dict[int, int]:
    """Returns {txId (int): node_index (int)}.
    Row order in features CSV = canonical node index.
    """
    return {int(txid): idx for idx, txid in enumerate(features["txId"])}


# ---------------------------------------------------------------------------
# Encode labels
# ---------------------------------------------------------------------------

def encode_labels(classes: pd.DataFrame, id_map: dict[int, int], n_nodes: int) -> torch.Tensor:
    """Map class strings to integers.
    illicit (class='1') → 1
    licit   (class='2') → 0
    unknown              → -1  (masked during training/eval)
    """
    labels = torch.full((n_nodes,), -1, dtype=torch.long)
    for _, row in classes.iterrows():
        txid = int(row["txId"])
        if txid not in id_map:
            continue
        idx = id_map[txid]
        c = str(row["class"]).strip()
        if c == "1":
            labels[idx] = 1
        elif c == "2":
            labels[idx] = 0
        # 'unknown' stays -1
    return labels


# ---------------------------------------------------------------------------
# Degree analysis (for reporting)
# ---------------------------------------------------------------------------

def degree_summary(edge_index: torch.Tensor, n_nodes: int) -> dict:
    src, dst = edge_index
    deg = torch.zeros(n_nodes, dtype=torch.long)
    deg.scatter_add_(0, src, torch.ones_like(src))
    deg.scatter_add_(0, dst, torch.ones_like(dst))
    deg_np = deg.numpy()
    isolated = int((deg_np == 0).sum())
    return {
        "isolated_nodes": isolated,
        "degree_min": int(deg_np.min()),
        "degree_max": int(deg_np.max()),
        "degree_mean": round(float(deg_np.mean()), 3),
        "degree_median": float(np.median(deg_np)),
        "degree_p95": float(np.percentile(deg_np, 95)),
        "nodes_degree_1": int((deg_np == 1).sum()),
        "nodes_degree_lte_2": int((deg_np <= 2).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw CSVs …")
    features, classes, edges = load_raw()
    logger.info("  features %s | classes %s | edges %s",
                features.shape, classes.shape, edges.shape)

    logger.info("Validating …")
    val_report = validate(features, classes, edges)

    logger.info("Cleaning …")
    features, classes, edges = clean(features, classes, edges)

    logger.info("Building integer ID map …")
    id_map = build_id_map(features)
    n_nodes = len(id_map)
    logger.info("  %d unique nodes", n_nodes)

    # ---- node feature tensor ----
    logger.info("Building node_features tensor [%d, %d] …", n_nodes, N_FEATURES)
    feat_cols = [f"feat_{i}" for i in range(1, N_FEATURES + 1)]
    node_features = torch.tensor(features[feat_cols].values, dtype=torch.float32)

    # ---- time step tensor ----
    node_times = torch.tensor(features["time_step"].values, dtype=torch.int32)

    # ---- label tensor ----
    logger.info("Encoding labels …")
    node_labels = encode_labels(classes, id_map, n_nodes)
    illicit = int((node_labels == 1).sum())
    licit = int((node_labels == 0).sum())
    unknown = int((node_labels == -1).sum())
    logger.info("  illicit=%d  licit=%d  unknown=%d", illicit, licit, unknown)

    # ---- edge index tensor ----
    logger.info("Building edge_index tensor …")
    src_idx = torch.tensor([id_map[int(v)] for v in edges["txId1"]], dtype=torch.long)
    dst_idx = torch.tensor([id_map[int(v)] for v in edges["txId2"]], dtype=torch.long)
    edge_index = torch.stack([src_idx, dst_idx], dim=0)  # shape [2, E]
    logger.info("  edge_index shape: %s", list(edge_index.shape))

    # ---- degree analysis ----
    deg_stats = degree_summary(edge_index, n_nodes)
    logger.info("  degree stats: %s", deg_stats)
    if deg_stats["isolated_nodes"]:
        logger.warning(
            "  %d isolated nodes (degree 0) — kept (they carry valid features)",
            deg_stats["isolated_nodes"],
        )

    # ---- save ----
    logger.info("Saving processed files to %s …", PROCESSED_DIR)

    id_map_path = PROCESSED_DIR / "id_map.json"
    with open(id_map_path, "w") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)
    logger.info("  id_map.json (%d entries)", len(id_map))

    torch.save(node_features, PROCESSED_DIR / "node_features.pt")
    logger.info("  node_features.pt %s  dtype=%s", tuple(node_features.shape), node_features.dtype)

    torch.save(node_times, PROCESSED_DIR / "node_times.pt")
    logger.info("  node_times.pt %s  dtype=%s", tuple(node_times.shape), node_times.dtype)

    torch.save(node_labels, PROCESSED_DIR / "node_labels.pt")
    logger.info("  node_labels.pt %s  dtype=%s", tuple(node_labels.shape), node_labels.dtype)

    torch.save(edge_index, PROCESSED_DIR / "edge_index.pt")
    logger.info("  edge_index.pt %s  dtype=%s", tuple(edge_index.shape), edge_index.dtype)

    # ---- preprocessing report ----
    report = {
        "n_nodes": n_nodes,
        "n_edges": int(edge_index.shape[1]),
        "n_features": N_FEATURES,
        "n_time_steps": int(node_times.unique().numel()),
        "time_step_range": [int(node_times.min()), int(node_times.max())],
        "labels": {"illicit": illicit, "licit": licit, "unknown": unknown},
        "illicit_pct_of_labelled": round(100 * illicit / (illicit + licit), 2),
        "feature_dtype": str(node_features.dtype),
        "label_encoding": {"1=illicit": 1, "2=licit": 0, "unknown": -1},
        "validation": val_report,
        "degree_stats": deg_stats,
        "files": {
            "id_map": "id_map.json",
            "node_features": "node_features.pt",
            "node_times": "node_times.pt",
            "node_labels": "node_labels.pt",
            "edge_index": "edge_index.pt",
        },
    }
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("  preprocessing_report.json")

    # ---- console summary ----
    print("\n=== Phase 1.3 Preprocessing Summary ===")
    print(f"  Nodes              : {n_nodes:,}")
    print(f"  Edges              : {int(edge_index.shape[1]):,}")
    print(f"  Time steps         : {report['n_time_steps']} (range {report['time_step_range']})")
    print(f"  Node features      : {N_FEATURES}")
    print(f"  Illicit nodes      : {illicit:,}  ({report['illicit_pct_of_labelled']}% of labelled)")
    print(f"  Licit nodes        : {licit:,}")
    print(f"  Unknown nodes      : {unknown:,}")
    print(f"  Isolated nodes     : {deg_stats['isolated_nodes']:,}")
    print(f"  Duplicate edges    : {val_report['duplicate_directed_edges']}")
    print(f"  Dangling edges     : {val_report['dangling_edge_nodes']}")
    print(f"  Outputs            : {PROCESSED_DIR}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
