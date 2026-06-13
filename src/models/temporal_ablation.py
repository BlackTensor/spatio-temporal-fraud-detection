"""Phase 5.3: Temporal ablation — compare static (Phase 4) vs temporal (Phase 5) models.

Loads training JSON files for all models and builds:
  results/temporal_ablation.md   — Markdown table
  results/temporal_ablation.csv  — CSV table
  results/temporal_ablation_roc.png — overlay ROC curves for all 5 models

Usage
-----
    python -m src.models.temporal_ablation
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("temporal_ablation")

REPO_ROOT    = Path(__file__).resolve().parents[2]
RESULTS_DIR  = REPO_ROOT / "results"

# Models in display order
MODEL_FILES = [
    # (display_name, training_json, has_prob_arrays)
    ("XGBoost",            "baseline_metrics.json",              False),
    ("GCN",                "gcn_training.json",                  True),
    ("GraphSAGE",          "graphsage_training.json",            True),
    ("GAT",                "gat_training.json",                  True),
    ("SnapshotGNN (5.1)",  "temporal_snapshot_gnn_training.json", True),
    ("EvolveGCN (5.2)",    "evolve_gcn_training.json",           True),
]

COLORS = {
    "XGBoost":           "#95a5a6",
    "GCN":               "#e74c3c",
    "GraphSAGE":         "#2ecc71",
    "GAT":               "#9b59b6",
    "SnapshotGNN (5.1)": "#e67e22",
    "EvolveGCN (5.2)":   "#3498db",
}


def load_results() -> list[dict]:
    rows = []

    # XGBoost — different JSON structure
    xgb_metrics_path = RESULTS_DIR / "baseline_metrics.json"
    xgb_eval_path    = RESULTS_DIR / "baseline_evaluation.json"
    if xgb_metrics_path.exists() and xgb_eval_path.exists():
        with open(xgb_metrics_path) as f:
            bm = json.load(f)
        with open(xgb_eval_path) as f:
            be = json.load(f)
        rows.append({
            "name":       "XGBoost",
            "n_params":   "N/A",
            "val_f1":     round(bm["val_f1"], 4),
            "val_auc":    round(bm["val_roc_auc"], 4),
            "test_f1":    round(be["test_f1"], 4),
            "test_auc":   round(be["test_roc_auc"], 4),
            "train_s":    "N/A",
            "best_epoch": "N/A",
            "phase":      "Phase 2",
            "val_probs":  None,
            "val_labels": None,
            "test_probs":  None,
            "test_labels": None,
        })
    else:
        logger.warning("XGBoost results not found — skipping")

    # GNN models
    gnn_files = [
        ("GCN",               "gcn_training.json"),
        ("GraphSAGE",         "graphsage_training.json"),
        ("GAT",               "gat_training.json"),
        ("SnapshotGNN (5.1)", "temporal_snapshot_gnn_training.json"),
        ("EvolveGCN (5.2)",   "evolve_gcn_training.json"),
    ]

    for display_name, fname in gnn_files:
        path = RESULTS_DIR / fname
        if not path.exists():
            logger.warning("  %s not found — skipping", fname)
            continue
        with open(path) as f:
            d = json.load(f)

        # Prob arrays may be inline (Phase 5) or in a sidecar _probs.json (Phase 4).
        val_probs   = d.get("val_probs")
        val_labels  = d.get("val_labels")
        test_probs  = d.get("test_probs")
        test_labels = d.get("test_labels")

        if val_probs is None:
            model_key = fname.replace("_training.json", "")
            sidecar = RESULTS_DIR / f"{model_key}_probs.json"
            if sidecar.exists():
                with open(sidecar) as f:
                    pd = json.load(f)
                val_probs   = pd.get("val_probs")
                val_labels  = pd.get("val_labels")
                test_probs  = pd.get("test_probs")
                test_labels = pd.get("test_labels")

        phase = "Phase 4" if display_name in ("GCN", "GraphSAGE", "GAT") else "Phase 5"
        rows.append({
            "name":       display_name,
            "n_params":   d.get("n_params", "N/A"),
            "val_f1":     round(d["val_metrics"]["f1"], 4),
            "val_auc":    round(d["val_metrics"]["roc_auc"], 4),
            "test_f1":    round(d["test_metrics"]["f1"], 4),
            "test_auc":   round(d["test_metrics"]["roc_auc"], 4),
            "train_s":    round(d.get("training_time_s", 0), 1),
            "best_epoch": d.get("best_epoch", "N/A"),
            "phase":      phase,
            "val_probs":  val_probs,
            "val_labels": val_labels,
            "test_probs":  test_probs,
            "test_labels": test_labels,
        })

    return rows


def build_table(rows: list[dict]) -> None:
    cols = ["Model", "Phase", "Params", "Val F1", "Val AUC", "Test F1", "Test AUC",
            "Train (s)", "Best Ep"]

    table_rows = []
    for r in rows:
        table_rows.append({
            "Model":      r["name"],
            "Phase":      r["phase"],
            "Params":     r["n_params"],
            "Val F1":     r["val_f1"],
            "Val AUC":    r["val_auc"],
            "Test F1":    r["test_f1"],
            "Test AUC":   r["test_auc"],
            "Train (s)":  r["train_s"],
            "Best Ep":    r["best_epoch"],
        })

    # ── CSV ──
    csv_path = RESULTS_DIR / "temporal_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table_rows)
    logger.info("Saved temporal_ablation.csv")

    # ── Markdown ──
    def _sep(c: str) -> str:
        return "-" * max(len(c), 8)

    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(_sep(c) for c in cols) + " |"
    body   = "\n".join(
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for r in table_rows
    )

    note = (
        "\n\n## Key Findings\n\n"
        "| Comparison | Test AUC delta |\n"
        "|------------|----------------|\n"
    )

    # Compute deltas
    auc_by_name = {r["name"]: r["test_auc"] for r in rows}
    baselines = [
        ("SnapshotGNN vs GraphSAGE (best static)",
         auc_by_name.get("SnapshotGNN (5.1)", 0) - auc_by_name.get("GraphSAGE", 0)),
        ("EvolveGCN vs GraphSAGE",
         auc_by_name.get("EvolveGCN (5.2)", 0) - auc_by_name.get("GraphSAGE", 0)),
        ("Best temporal vs XGBoost",
         max(
             auc_by_name.get("SnapshotGNN (5.1)", 0),
             auc_by_name.get("EvolveGCN (5.2)",  0),
         ) - auc_by_name.get("XGBoost", 0)),
    ]
    for label, delta in baselines:
        sign = "+" if delta >= 0 else ""
        note += f"| {label} | {sign}{delta:.4f} |\n"

    note += (
        "\n> **Note on concept drift**: Illicit prevalence drops 11.6% (train) → "
        "9.2% (val) → 2.5% (test). Test F1 is low for all models due to this "
        "distribution shift. ROC-AUC is the reliable cross-time metric.\n"
    )

    md = f"# Phase 5: Temporal GNN Ablation — Elliptic Bitcoin Dataset\n\n{header}\n{sep}\n{body}\n{note}\n"
    md_path = RESULTS_DIR / "temporal_ablation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Saved temporal_ablation.md")

    print("\n" + header)
    print(sep)
    print(body)


def plot_roc(rows: list[dict]) -> None:
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(15, 6))

    for ax, split, title in [
        (ax_val,  "val",  "Validation (ts 35-42)"),
        (ax_test, "test", "Test (ts 43-49)"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        for r in rows:
            probs  = r.get(f"{split}_probs")
            labels = r.get(f"{split}_labels")
            if probs is None or labels is None:
                continue
            probs  = np.array(probs)
            labels = np.array(labels)
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = r[f"{split}_auc"]
            phase_marker = "★" if r["phase"] == "Phase 5" else " "
            ax.plot(fpr, tpr, lw=2.0 if r["phase"] == "Phase 5" else 1.2,
                    color=COLORS.get(r["name"], "gray"),
                    linestyle="-" if r["phase"] == "Phase 5" else "--",
                    label=f"{phase_marker}{r['name']} AUC={auc:.3f}")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Phase 5 Temporal Ablation: Static vs Temporal GNNs — Elliptic",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = RESULTS_DIR / "temporal_ablation_roc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved temporal_ablation_roc.png")


def main() -> int:
    rows = load_results()
    if not rows:
        logger.error("No model results found in %s", RESULTS_DIR)
        return 1

    logger.info("Found %d models to compare", len(rows))
    build_table(rows)
    plot_roc(rows)

    print("\n=== Phase 5.3 Ablation complete ===")
    print(f"  Outputs: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
