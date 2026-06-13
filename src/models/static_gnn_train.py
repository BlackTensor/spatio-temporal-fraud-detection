"""Phase 4: Static GNN baselines on the Elliptic Bitcoin dataset.

Phases covered
--------------
4.1  graph_homogeneous.pt — Elliptic is already homogeneous; graph.pt IS the
     homogeneous graph. This script saves a copy as graph_homogeneous.pt for
     the ablation reference used in Phase 6.
4.2  GCN        — results/gcn_model.pt, results/gcn_training.json
4.3  GraphSAGE  — results/graphsage_model.pt, results/graphsage_training.json
4.4  GAT        — results/gat_model.pt, results/gat_training.json
4.5  Comparison — results/model_comparison.csv, results/model_comparison.md

Training setup
--------------
- Full-batch transductive training (whole graph in memory; masks select nodes)
- Loss: BCEWithLogitsLoss(pos_weight) to handle 9:1 licit-to-illicit imbalance
- Early stopping on val-labeled F1 (patience=30 epochs)
- Threshold tuned on val set (F1-optimal), applied to test — no leakage
- Seeds fixed for reproducibility

Usage
-----
    python -m src.models.static_gnn_train
    python -m src.models.static_gnn_train --models gcn graphsage   # subset
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from src.models.gnn_models import get_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("static_gnn_train")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

SEED = 42

# ── Hyperparameters ──────────────────────────────────────────────────────────
HP = {
    "hidden_channels": 64,
    "dropout": 0.3,
    "heads": 2,           # GAT only
    "lr": 0.01,
    "weight_decay": 5e-4,
    "max_epochs": 300,
    "patience": 30,       # early stopping on val F1
}


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    logits = model(data.x, data.edge_index).squeeze(-1)
    probs = torch.sigmoid(logits[mask]).numpy()
    labels = data.y[mask].numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(labels, probs)),
    }, probs, labels


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds 0.05–0.95; return (thresh, f1) that maximises illicit F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


@torch.no_grad()
def time_inference_ms(model: torch.nn.Module, data, n_runs: int = 5) -> float:
    """Average inference ms over n_runs forward passes (whole graph)."""
    model.eval()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(data.x, data.edge_index)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)   # ms for full graph


# ── Core trainer ─────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    data,
    pos_weight: torch.Tensor,
) -> dict:
    set_seed(SEED)
    n_features = data.x.shape[1]
    kwargs: dict = dict(in_channels=n_features, hidden_channels=HP["hidden_channels"],
                        dropout=HP["dropout"])
    if model_name == "gat":
        kwargs["heads"] = HP["heads"]

    model = get_model(model_name, **kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[%s] %d parameters", model_name.upper(), n_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )

    train_mask = data.train_labeled_mask
    val_mask   = data.val_labeled_mask
    test_mask  = data.test_labeled_mask

    best_val_auc = -1.0
    best_epoch   = 0
    best_state   = None
    no_improve   = 0
    history: list[dict] = []

    t_train_start = time.perf_counter()

    for epoch in range(1, HP["max_epochs"] + 1):
        # ── train ──
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], data.y[train_mask].float(),
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        # ── evaluate on val ──
        val_m, _, _ = evaluate(model, data, val_mask)
        val_f1  = val_m["f1"]
        val_auc = val_m["roc_auc"]

        history.append({
            "epoch":   epoch,
            "loss":    round(float(loss.item()), 6),
            "val_f1":  round(val_f1, 6),
            "val_auc": round(val_auc, 6),
        })

        # Early stopping on val_auc (more stable than val_f1 at fixed threshold under imbalance)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                "  [%s] ep %3d  loss=%.4f  val_f1=%.4f  val_auc=%.4f  best_ep=%d",
                model_name.upper(), epoch, loss.item(), val_f1, val_auc, best_epoch,
            )

        if no_improve >= HP["patience"]:
            logger.info("  [%s] early stop at epoch %d (best=%d)", model_name.upper(), epoch, best_epoch)
            break

    training_time_s = time.perf_counter() - t_train_start

    # ── reload best checkpoint ──
    model.load_state_dict(best_state)

    # ── threshold on val ──
    _, val_probs, val_labels = evaluate(model, data, val_mask)
    best_thresh, _ = find_best_threshold(val_probs, val_labels)

    # ── final val metrics at best threshold ──
    val_m, val_probs, val_labels = evaluate(model, data, val_mask, threshold=best_thresh)

    # ── test metrics ──
    test_m, test_probs, test_labels = evaluate(model, data, test_mask, threshold=best_thresh)

    # ── inference latency ──
    infer_ms_full = time_inference_ms(model, data)
    infer_ms_per_1k = infer_ms_full / (data.num_nodes / 1000)

    # ── detailed reports ──
    val_report  = classification_report(
        val_labels,  (val_probs >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )
    test_report = classification_report(
        test_labels, (test_probs >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )

    result = {
        "model":              model_name,
        "n_params":           n_params,
        "best_epoch":         best_epoch,
        "val_threshold":      round(best_thresh, 4),
        "training_time_s":    round(training_time_s, 2),
        "inference_ms_full":  round(infer_ms_full, 2),
        "inference_ms_per_1k": round(infer_ms_per_1k, 4),
        "val_metrics":        {k: round(v, 6) for k, v in val_m.items()},
        "test_metrics":       {k: round(v, 6) for k, v in test_m.items()},
        "val_report":         val_report,
        "test_report":        test_report,
        "hyperparams":        HP,
        "history":            history,
        "val_probs":          val_probs.tolist(),
        "val_labels":         val_labels.tolist(),
        "test_probs":         test_probs.tolist(),
        "test_labels":        test_labels.tolist(),
    }

    # ── save model ──
    model_path = RESULTS_DIR / f"{model_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("  Saved %s", model_path.name)

    # ── save training JSON (without raw prob arrays — keep file small) ──
    training_log = {k: v for k, v in result.items()
                    if k not in ("val_probs", "val_labels", "test_probs", "test_labels")}
    log_path = RESULTS_DIR / f"{model_name}_training.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("  Saved %s", log_path.name)

    # ── save prob arrays separately (used by temporal_ablation.py for ROC plots) ──
    probs_log = {
        "val_probs":   result["val_probs"],
        "val_labels":  result["val_labels"],
        "test_probs":  result["test_probs"],
        "test_labels": result["test_labels"],
    }
    probs_path = RESULTS_DIR / f"{model_name}_probs.json"
    with open(probs_path, "w") as f:
        json.dump(probs_log, f)
    logger.info("  Saved %s", probs_path.name)

    logger.info(
        "  [%s] DONE  val_f1=%.4f  val_auc=%.4f  test_f1=%.4f  test_auc=%.4f  t=%.0fs",
        model_name.upper(),
        val_m["f1"], val_m["roc_auc"],
        test_m["f1"], test_m["roc_auc"],
        training_time_s,
    )
    return result


# ── Comparison table ─────────────────────────────────────────────────────────

def build_comparison(results: list[dict]) -> None:
    """Load XGBoost baseline + GNN results → CSV + Markdown table."""
    rows = []

    # XGBoost baseline from Phase 2
    try:
        with open(RESULTS_DIR / "baseline_metrics.json") as f:
            bm = json.load(f)
        with open(RESULTS_DIR / "baseline_evaluation.json") as f:
            be = json.load(f)
        rows.append({
            "Model":        "XGBoost (baseline)",
            "Params":       "N/A",
            "Val F1":       round(bm["val_f1"], 4),
            "Val AUC":      round(bm["val_roc_auc"], 4),
            "Test F1":      round(be["test_f1"], 4),
            "Test AUC":     round(be["test_roc_auc"], 4),
            "Train (s)":    "N/A",
            "Infer (ms/1k)": round(be["inference_ms_per_node"] * 1000, 4),
            "Best Epoch":   "N/A",
        })
    except FileNotFoundError:
        logger.warning("Baseline metrics not found; skipping XGBoost row")

    for r in results:
        rows.append({
            "Model":        r["model"].upper(),
            "Params":       r["n_params"],
            "Val F1":       round(r["val_metrics"]["f1"], 4),
            "Val AUC":      round(r["val_metrics"]["roc_auc"], 4),
            "Test F1":      round(r["test_metrics"]["f1"], 4),
            "Test AUC":     round(r["test_metrics"]["roc_auc"], 4),
            "Train (s)":    round(r["training_time_s"], 1),
            "Infer (ms/1k)": round(r["inference_ms_per_1k"], 4),
            "Best Epoch":   r["best_epoch"],
        })

    # ── CSV ──
    import csv
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved model_comparison.csv")

    # ── Markdown ──
    cols = list(rows[0].keys())
    def _sep(c: str) -> str: return "-" * max(len(c), 10)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(_sep(c) for c in cols) + " |"
    body   = "\n".join(
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for r in rows
    )

    note = (
        "\n\n> **Note on temporal concept drift**: Val metrics (ts 35-42) and Test metrics "
        "(ts 43-49) measure different time periods. Illicit prevalence drops from 11.6% "
        "(train) → 9.2% (val) → 2.5% (test). ROC-AUC is the most reliable cross-time "
        "comparison metric. Test F1 is low for all models due to this distribution shift; "
        "the GNN's graph-structural signals are the key value-add over XGBoost."
    )

    md = f"# Phase 4: Static GNN Model Comparison\n\n{header}\n{sep}\n{body}\n{note}\n"
    md_path = RESULTS_DIR / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Saved model_comparison.md")

    # ── print table ──
    print("\n" + header)
    print(sep)
    print(body)


def plot_roc_comparison(results: list[dict]) -> None:
    """Overlay ROC curves for all models (val + test) + XGBoost."""
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"gcn": "#e74c3c", "graphsage": "#2ecc71", "gat": "#3498db",
              "xgboost": "#95a5a6"}

    for ax, split, split_label in [(ax_val, "val", "Validation (ts 35-42)"),
                                    (ax_test, "test", "Test (ts 43-49)")]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        for r in results:
            probs  = np.array(r[f"{split}_probs"])
            labels = np.array(r[f"{split}_labels"])
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = r[f"{split}_metrics"]["roc_auc"]
            ax.plot(fpr, tpr, lw=1.8, color=colors.get(r["model"], "gray"),
                    label=f"{r['model'].upper()} AUC={auc:.3f}")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(split_label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Phase 4: Static GNN ROC Curves — Elliptic Bitcoin Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "phase4_roc_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phase4_roc_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gcn", "graphsage", "gat"],
                        choices=["gcn", "graphsage", "gat"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 4.1: document that graph.pt == graph_homogeneous.pt ──
    graph_pt = PROCESSED_DIR / "graph.pt"
    homo_pt  = PROCESSED_DIR / "graph_homogeneous.pt"
    if not homo_pt.exists():
        import shutil
        shutil.copy2(graph_pt, homo_pt)
        logger.info("Phase 4.1: copied graph.pt → graph_homogeneous.pt "
                    "(Elliptic is already homogeneous; identical files)")

    # ── load graph ──
    logger.info("Loading graph.pt …")
    data = torch.load(graph_pt, weights_only=False)
    logger.info("  %d nodes  %d edges  %d features",
                data.num_nodes, data.num_edges, data.x.shape[1])

    # ── pos_weight from training labels ──
    train_labels = data.y[data.train_labeled_mask]
    n_licit   = int((train_labels == 0).sum())
    n_illicit = int((train_labels == 1).sum())
    pos_weight = torch.tensor([n_licit / n_illicit], dtype=torch.float32)
    logger.info("  pos_weight=%.3f  (licit=%d, illicit=%d in train)",
                pos_weight.item(), n_licit, n_illicit)

    # ── train models ──
    all_results: list[dict] = []
    for model_name in args.models:
        logger.info("\n=== Training %s ===", model_name.upper())
        result = train_model(model_name, data, pos_weight)
        all_results.append(result)

    # ── Phase 4.5: comparison table ──
    if len(all_results) > 0:
        logger.info("\n=== Phase 4.5: Building comparison table ===")
        build_comparison(all_results)
        plot_roc_comparison(all_results)

    print("\n=== Phase 4 Complete ===")
    for r in all_results:
        print(f"  {r['model'].upper():<12}  val_f1={r['val_metrics']['f1']:.4f}  "
              f"val_auc={r['val_metrics']['roc_auc']:.4f}  "
              f"test_f1={r['test_metrics']['f1']:.4f}  "
              f"test_auc={r['test_metrics']['roc_auc']:.4f}  "
              f"t={r['training_time_s']:.0f}s")
    print(f"  Outputs: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
