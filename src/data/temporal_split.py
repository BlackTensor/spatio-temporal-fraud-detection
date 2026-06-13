"""Phase 1.5: Temporal train / val / test split — no label leakage.

Split strategy
--------------
The Elliptic dataset covers 49 time steps in chronological order.
Following the convention established in the original Elliptic paper
(Weber et al., 2019), time steps 1–34 form the training period.
We further subdivide the remaining steps into a validation window
(ts 35–42) and a test window (ts 43–49).

    train  ts  1–34   (~67 % of nodes)   earliest
    val    ts 35–42   (~19 % of nodes)
    test   ts 43–49   (~14 % of nodes)   latest

No-leakage guarantee
--------------------
* Node features (graph structure, raw features) for all 203 k nodes are
  always available — this is the standard transductive GNN setting.
* Only **training node labels** are exposed during model training.
* Validation labels are used solely for early stopping / hyperparameter
  selection and are never back-propagated.
* Test labels are withheld until final evaluation.

Outputs
-------
data/processed/splits.json      human-readable summary + index lists
data/processed/splits.pt        {mask tensors} for direct PyG use

splits.pt keys
--------------
train_mask          bool [N]  — all training nodes
val_mask            bool [N]  — all validation nodes
test_mask           bool [N]  — all test nodes
train_labeled_mask  bool [N]  — training nodes with known label (≠ -1)
val_labeled_mask    bool [N]  — val nodes with known label
test_labeled_mask   bool [N]  — test nodes with known label

Usage
-----
    python -m src.data.temporal_split
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("temporal_split")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TRAIN_TS = (1, 34)
VAL_TS   = (35, 42)
TEST_TS  = (43, 49)


def make_masks(
    times: torch.Tensor,
    labels: torch.Tensor,
    ts_range: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (split_mask, labeled_mask) for a given time-step range."""
    lo, hi = ts_range
    split_mask   = (times >= lo) & (times <= hi)
    labeled_mask = split_mask & (labels != -1)
    return split_mask, labeled_mask


def split_stats(
    name: str,
    ts_range: tuple[int, int],
    split_mask: torch.Tensor,
    labeled_mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    n_total   = int(split_mask.sum())
    n_labeled = int(labeled_mask.sum())
    n_illicit = int((labels[labeled_mask] == 1).sum())
    n_licit   = int((labels[labeled_mask] == 0).sum())
    n_unknown = n_total - n_labeled
    illicit_pct = round(100 * n_illicit / n_labeled, 2) if n_labeled else 0.0
    return {
        "name": name,
        "time_step_range": list(ts_range),
        "n_nodes": n_total,
        "n_labeled": n_labeled,
        "n_illicit": n_illicit,
        "n_licit": n_licit,
        "n_unknown": n_unknown,
        "illicit_pct_of_labeled": illicit_pct,
    }


def main() -> int:
    logger.info("Loading processed tensors …")
    times  = torch.load(PROCESSED_DIR / "node_times.pt",  weights_only=True)
    labels = torch.load(PROCESSED_DIR / "node_labels.pt", weights_only=True)
    n_nodes = times.shape[0]
    logger.info("  %d nodes, time steps %d–%d", n_nodes, int(times.min()), int(times.max()))

    # ---- build masks ----
    logger.info("Building split masks …")
    train_mask, train_labeled_mask = make_masks(times, labels, TRAIN_TS)
    val_mask,   val_labeled_mask   = make_masks(times, labels, VAL_TS)
    test_mask,  test_labeled_mask  = make_masks(times, labels, TEST_TS)

    # sanity: every node belongs to exactly one split
    overlap = (train_mask & val_mask).any() or (val_mask & test_mask).any() or (train_mask & test_mask).any()
    if overlap:
        raise RuntimeError("Split masks overlap — check time-step boundaries.")
    coverage = int(train_mask.sum()) + int(val_mask.sum()) + int(test_mask.sum())
    if coverage != n_nodes:
        raise RuntimeError(f"Split coverage {coverage} ≠ {n_nodes} nodes — check boundaries.")
    logger.info("  Overlap check: OK | Coverage check: OK (%d / %d)", coverage, n_nodes)

    # ---- per-split stats ----
    splits_info = [
        split_stats("train", TRAIN_TS, train_mask, train_labeled_mask, labels),
        split_stats("val",   VAL_TS,   val_mask,   val_labeled_mask,   labels),
        split_stats("test",  TEST_TS,  test_mask,  test_labeled_mask,  labels),
    ]
    for s in splits_info:
        logger.info(
            "  %-5s  ts=%s  nodes=%6d  labeled=%5d  illicit=%4d  licit=%5d  unknown=%6d  illicit%%=%.1f",
            s["name"], s["time_step_range"], s["n_nodes"], s["n_labeled"],
            s["n_illicit"], s["n_licit"], s["n_unknown"], s["illicit_pct_of_labeled"],
        )

    # ---- save splits.pt ----
    masks_dict = {
        "train_mask":         train_mask,
        "val_mask":           val_mask,
        "test_mask":          test_mask,
        "train_labeled_mask": train_labeled_mask,
        "val_labeled_mask":   val_labeled_mask,
        "test_labeled_mask":  test_labeled_mask,
    }
    splits_pt_path = PROCESSED_DIR / "splits.pt"
    torch.save(masks_dict, splits_pt_path)
    logger.info("Saved splits.pt (%d mask tensors)", len(masks_dict))

    # ---- save splits.json ----
    # Store indices as lists so JSON is human-readable and usable without torch
    json_payload = {
        "split_strategy": "temporal — no label leakage",
        "time_step_ranges": {
            "train": list(TRAIN_TS),
            "val":   list(VAL_TS),
            "test":  list(TEST_TS),
        },
        "splits": {s["name"]: s for s in splits_info},
        "note": (
            "train_labeled_mask / val_labeled_mask / test_labeled_mask exclude unknown-label nodes. "
            "Full graph structure (all 203 k nodes + edges) is visible during training in the "
            "transductive GNN setting. Only training labels are used for gradient updates."
        ),
    }
    splits_json_path = PROCESSED_DIR / "splits.json"
    with open(splits_json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    logger.info("Saved splits.json")

    # ---- console summary ----
    total_labeled = sum(s["n_labeled"] for s in splits_info)
    print("\n=== Phase 1.5 Temporal Split Summary ===")
    print(f"{'Split':<6} {'TS range':<10} {'Nodes':>8} {'Labeled':>8} {'Illicit':>8} "
          f"{'Licit':>8} {'Unknown':>8} {'Illicit%':>9}")
    print("-" * 72)
    for s in splits_info:
        ts = f"{s['time_step_range'][0]}–{s['time_step_range'][1]}"
        print(f"{s['name']:<6} {ts:<10} {s['n_nodes']:>8,} {s['n_labeled']:>8,} "
              f"{s['n_illicit']:>8,} {s['n_licit']:>8,} {s['n_unknown']:>8,} "
              f"{s['illicit_pct_of_labeled']:>8.1f}%")
    print("-" * 72)
    print(f"{'Total':<6} {'1–49':<10} {n_nodes:>8,} {total_labeled:>8,}")
    print(f"\n  Saved: splits.pt + splits.json -> {PROCESSED_DIR}")
    print("  No-leakage guarantee: only train labels exposed during training.")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
