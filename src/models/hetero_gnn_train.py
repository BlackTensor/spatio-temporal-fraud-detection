"""Phase 6: Heterogeneous GNN training and ablation.

Phases covered
--------------
6.1  HeteroSAGE  — static, two-direction SAGEConv (hetero analog of GraphSAGE)
     HGAT        — static, two-direction GATConv  (Phase 6.1)
6.2  HTGN        — temporal + heterogeneous (HeteroSAGE encoder + GRU)
6.3  Ablation    — full comparison table across Phases 2–6 + ROC curves

Heterogeneity construction
--------------------------
Elliptic has one node type (transactions).  Directed edges are split into two
typed relations:
  sends    : original edges A→B  (how B is funded)
  receives : reversed edges B→A  (where A sends money to)

Each relation uses independent convolution parameters so the model learns
asymmetric representations for upstream vs downstream fraud propagation.

Usage
-----
    python -m src.models.hetero_gnn_train                     # all three models
    python -m src.models.hetero_gnn_train --models hetero_sage hgat
    python -m src.models.hetero_gnn_train --models htgn
    python -m src.models.hetero_gnn_train --ablation-only     # just rebuild table
"""

from __future__ import annotations

import argparse
import copy
import csv
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

from src.models.hetero_gnn_models import (
    HeteroSAGE, HeteroGAT, HeteroTemporalGNN,
    STATIC_MODELS, TEMPORAL_MODELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hetero_gnn_train")

REPO_ROOT     = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR   = REPO_ROOT / "results"

SEED = 42

# Static HP (mirrors Phase 4)
HP_STATIC = {
    "hidden_channels": 64,
    "dropout":         0.3,
    "heads":           2,
    "lr":              0.01,
    "weight_decay":    5e-4,
    "max_epochs":      300,
    "patience":        30,
}

# Temporal HP (mirrors Phase 5)
HP_TEMPORAL = {
    "hidden_channels": 64,
    "dropout":         0.3,
    "lr":              0.005,
    "weight_decay":    5e-4,
    "max_epochs":      200,
    "patience":        25,
}


# ── Shared utilities ──────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(labels, probs)),
    }


def _save_result(result: dict, model_name: str) -> None:
    """Save model checkpoint, training JSON, and prob sidecar."""
    torch.save(result.pop("_state_dict"), RESULTS_DIR / f"{model_name}_model.pt")
    logger.info("  Saved %s_model.pt", model_name)

    probs_log = {k: result[k] for k in ("val_probs", "val_labels", "test_probs", "test_labels")}
    with open(RESULTS_DIR / f"{model_name}_probs.json", "w") as f:
        json.dump(probs_log, f)
    logger.info("  Saved %s_probs.json", model_name)

    training_log = {k: v for k, v in result.items()
                    if k not in ("val_probs", "val_labels", "test_probs", "test_labels")}
    with open(RESULTS_DIR / f"{model_name}_training.json", "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("  Saved %s_training.json", model_name)


# ── Phase 6.1: Static heterogeneous trainer ───────────────────────────────────

@torch.no_grad()
def _eval_static(
    model: torch.nn.Module,
    data,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    logits = model(data.x, data.edge_index).squeeze(-1)
    probs  = torch.sigmoid(logits[mask]).numpy()
    labels = data.y[mask].numpy()
    preds  = (probs >= threshold).astype(int)
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(labels, probs)),
    }, probs, labels


def train_static(model_name: str, data, pos_weight: torch.Tensor) -> dict:
    set_seed(SEED)
    HP = HP_STATIC
    in_ch = data.x.shape[1]

    if model_name == "hetero_sage":
        model: torch.nn.Module = HeteroSAGE(
            in_channels=in_ch,
            hidden_channels=HP["hidden_channels"],
            dropout=HP["dropout"],
        )
    else:  # hgat
        model = HeteroGAT(
            in_channels=in_ch,
            hidden_channels=HP["hidden_channels"],
            heads=HP["heads"],
            dropout=HP["dropout"],
        )

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

    t_start = time.perf_counter()

    for epoch in range(1, HP["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], data.y[train_mask].float(),
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        val_m, _, _ = _eval_static(model, data, val_mask)
        val_auc = val_m["roc_auc"]
        val_f1  = val_m["f1"]

        history.append({
            "epoch":   epoch,
            "loss":    round(float(loss.item()), 6),
            "val_f1":  round(val_f1, 6),
            "val_auc": round(val_auc, 6),
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(
                "  [%s] ep %3d  loss=%.4f  val_f1=%.4f  val_auc=%.4f  best=%d",
                model_name.upper(), epoch, loss.item(), val_f1, val_auc, best_epoch,
            )

        if no_improve >= HP["patience"]:
            logger.info("  [%s] early stop at ep %d (best=%d)",
                        model_name.upper(), epoch, best_epoch)
            break

    training_time_s = time.perf_counter() - t_start

    model.load_state_dict(best_state)

    _, val_probs, val_labels = _eval_static(model, data, val_mask)
    best_thresh, _ = find_best_threshold(val_probs, val_labels)

    val_m, val_probs, val_labels   = _eval_static(model, data, val_mask,  best_thresh)
    test_m, test_probs, test_labels = _eval_static(model, data, test_mask, best_thresh)

    val_report  = classification_report(
        val_labels,  (val_probs  >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )
    test_report = classification_report(
        test_labels, (test_probs >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )

    logger.info(
        "  [%s] DONE  val_f1=%.4f  val_auc=%.4f  test_f1=%.4f  test_auc=%.4f  t=%.0fs",
        model_name.upper(),
        val_m["f1"], val_m["roc_auc"], test_m["f1"], test_m["roc_auc"],
        training_time_s,
    )

    result = {
        "_state_dict":         best_state,
        "model":               model_name,
        "n_params":            n_params,
        "best_epoch":          best_epoch,
        "val_threshold":       round(best_thresh, 4),
        "training_time_s":     round(training_time_s, 2),
        "val_metrics":         {k: round(v, 6) for k, v in val_m.items()},
        "test_metrics":        {k: round(v, 6) for k, v in test_m.items()},
        "val_report":          val_report,
        "test_report":         test_report,
        "hyperparams":         HP,
        "history":             history,
        "val_probs":           val_probs.tolist(),
        "val_labels":          val_labels.tolist(),
        "test_probs":          test_probs.tolist(),
        "test_labels":         test_labels.tolist(),
    }
    _save_result(result, model_name)
    return result


# ── Phase 6.2: Temporal heterogeneous trainer ─────────────────────────────────

@torch.no_grad()
def _replay_temporal(
    model: HeteroTemporalGNN,
    snapshots: list,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    h = model.init_state(device)
    val_p, val_l, test_p, test_l = [], [], [], []

    for snap in snapshots:
        x          = snap.x.to(device)
        edge_index = snap.edge_index.to(device)
        logits, h  = model.forward_snapshot(x, edge_index, h)
        probs      = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        if snap.val_labeled_mask.any():
            val_p.append(probs[snap.val_labeled_mask.numpy()])
            val_l.append(snap.y[snap.val_labeled_mask].numpy())
        if snap.test_labeled_mask.any():
            test_p.append(probs[snap.test_labeled_mask.numpy()])
            test_l.append(snap.y[snap.test_labeled_mask].numpy())

    return (
        np.concatenate(val_p),  np.concatenate(val_l),
        np.concatenate(test_p) if test_p else np.array([]),
        np.concatenate(test_l) if test_l else np.array([]),
    )


def train_temporal(model_name: str, snapshots: list, pos_weight: torch.Tensor) -> dict:
    set_seed(SEED)
    HP     = HP_TEMPORAL
    device = torch.device("cpu")
    in_ch  = snapshots[0].x.shape[1]

    model = HeteroTemporalGNN(
        in_channels=in_ch,
        hidden_channels=HP["hidden_channels"],
        dropout=HP["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[%s] %d parameters", model_name.upper(), n_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )

    train_snaps = [s for s in snapshots if s.train_mask.any()]

    best_val_auc = -1.0
    best_epoch   = 0
    best_state   = None
    no_improve   = 0
    history: list[dict] = []

    t_start = time.perf_counter()

    for epoch in range(1, HP["max_epochs"] + 1):
        model.train()
        h          = model.init_state(device)
        epoch_loss = 0.0
        n_batches  = 0

        for snap in train_snaps:
            x          = snap.x.to(device)
            edge_index = snap.edge_index.to(device)
            logits, h_new = model.forward_snapshot(x, edge_index, h)

            mask = snap.train_labeled_mask
            if not mask.any():
                h = h_new.detach()
                continue

            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1)[mask],
                snap.y[mask].float().to(device),
                pos_weight=pos_weight.to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches  += 1
            h = h_new.detach()

        val_probs, val_labels, _, _ = _replay_temporal(model, snapshots, device)
        val_auc = float(roc_auc_score(val_labels, val_probs))
        val_f1  = float(f1_score(val_labels, (val_probs >= 0.5).astype(int), zero_division=0))
        avg_loss = epoch_loss / max(n_batches, 1)

        history.append({
            "epoch":   epoch,
            "loss":    round(avg_loss, 6),
            "val_f1":  round(val_f1, 6),
            "val_auc": round(val_auc, 6),
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "  [%s] ep %3d  loss=%.4f  val_f1=%.4f  val_auc=%.4f  best=%d",
                model_name.upper(), epoch, avg_loss, val_f1, val_auc, best_epoch,
            )

        if no_improve >= HP["patience"]:
            logger.info("  [%s] early stop at ep %d (best=%d)",
                        model_name.upper(), epoch, best_epoch)
            break

    training_time_s = time.perf_counter() - t_start

    model.load_state_dict(best_state)

    val_probs, val_labels, test_probs, test_labels = _replay_temporal(
        model, snapshots, device
    )
    best_thresh, _ = find_best_threshold(val_probs, val_labels)

    val_m  = compute_metrics(val_probs,  val_labels,  best_thresh)
    test_m = compute_metrics(test_probs, test_labels, best_thresh)

    val_report  = classification_report(
        val_labels,  (val_probs  >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )
    test_report = classification_report(
        test_labels, (test_probs >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )

    logger.info(
        "  [%s] DONE  val_f1=%.4f  val_auc=%.4f  test_f1=%.4f  test_auc=%.4f  t=%.0fs",
        model_name.upper(),
        val_m["f1"], val_m["roc_auc"], test_m["f1"], test_m["roc_auc"],
        training_time_s,
    )

    result = {
        "_state_dict":         best_state,
        "model":               model_name,
        "n_params":            n_params,
        "best_epoch":          best_epoch,
        "val_threshold":       round(best_thresh, 4),
        "training_time_s":     round(training_time_s, 2),
        "val_metrics":         {k: round(v, 6) for k, v in val_m.items()},
        "test_metrics":        {k: round(v, 6) for k, v in test_m.items()},
        "val_report":          val_report,
        "test_report":         test_report,
        "hyperparams":         HP,
        "history":             history,
        "val_probs":           val_probs.tolist(),
        "val_labels":          val_labels.tolist(),
        "test_probs":          test_probs.tolist(),
        "test_labels":         test_labels.tolist(),
    }
    _save_result(result, model_name)
    return result


# ── Phase 6.3: Heterogeneous ablation ────────────────────────────────────────

# All models in display order across Phases 2–6
_ABLATION_MODELS = [
    # (display_name, phase_label, training_json_or_None)
    ("XGBoost",            "Phase 2",  None),
    ("GCN",                "Phase 4",  "gcn_training.json"),
    ("GraphSAGE",          "Phase 4",  "graphsage_training.json"),
    ("GAT",                "Phase 4",  "gat_training.json"),
    ("SnapshotGNN",        "Phase 5",  "temporal_snapshot_gnn_training.json"),
    ("EvolveGCN",          "Phase 5",  "evolve_gcn_training.json"),
    ("HeteroSAGE",         "Phase 6",  "hetero_sage_training.json"),
    ("HeteroGAT (HGAT)",   "Phase 6",  "hgat_training.json"),
    ("HTGN",               "Phase 6",  "htgn_training.json"),
]

_COLORS = {
    "XGBoost":          "#95a5a6",
    "GCN":              "#e74c3c",
    "GraphSAGE":        "#2ecc71",
    "GAT":              "#9b59b6",
    "SnapshotGNN":      "#e67e22",
    "EvolveGCN":        "#3498db",
    "HeteroSAGE":       "#f39c12",
    "HeteroGAT (HGAT)": "#1abc9c",
    "HTGN":             "#e91e63",
}


def _load_ablation_rows() -> list[dict]:
    rows = []

    # XGBoost — different JSON layout
    xgb_m = RESULTS_DIR / "baseline_metrics.json"
    xgb_e = RESULTS_DIR / "baseline_evaluation.json"
    if xgb_m.exists() and xgb_e.exists():
        with open(xgb_m) as f:
            bm = json.load(f)
        with open(xgb_e) as f:
            be = json.load(f)
        rows.append({
            "name":       "XGBoost",
            "phase":      "Phase 2",
            "n_params":   "N/A",
            "val_f1":     round(bm["val_f1"], 4),
            "val_auc":    round(bm["val_roc_auc"], 4),
            "test_f1":    round(be["test_f1"], 4),
            "test_auc":   round(be["test_roc_auc"], 4),
            "train_s":    "N/A",
            "best_epoch": "N/A",
            "val_probs":  None,
            "val_labels": None,
            "test_probs":  None,
            "test_labels": None,
        })
    else:
        logger.warning("XGBoost results not found — skipping")

    # All GNN models
    for display_name, phase, fname in _ABLATION_MODELS[1:]:
        if fname is None:
            continue
        path = RESULTS_DIR / fname
        if not path.exists():
            logger.warning("  %s not found — skipping", fname)
            continue
        with open(path) as f:
            d = json.load(f)

        # Prob arrays: try sidecar if not inline
        val_probs = val_labels = test_probs = test_labels = None
        sidecar = RESULTS_DIR / fname.replace("_training.json", "_probs.json")
        if sidecar.exists():
            with open(sidecar) as f:
                pd = json.load(f)
            val_probs   = pd.get("val_probs")
            val_labels  = pd.get("val_labels")
            test_probs  = pd.get("test_probs")
            test_labels = pd.get("test_labels")

        rows.append({
            "name":       display_name,
            "phase":      phase,
            "n_params":   d.get("n_params", "N/A"),
            "val_f1":     round(d["val_metrics"]["f1"], 4),
            "val_auc":    round(d["val_metrics"]["roc_auc"], 4),
            "test_f1":    round(d["test_metrics"]["f1"], 4),
            "test_auc":   round(d["test_metrics"]["roc_auc"], 4),
            "train_s":    round(d.get("training_time_s", 0), 1),
            "best_epoch": d.get("best_epoch", "N/A"),
            "val_probs":  val_probs,
            "val_labels": val_labels,
            "test_probs":  test_probs,
            "test_labels": test_labels,
        })

    return rows


def build_ablation(rows: list[dict]) -> None:
    if not rows:
        logger.error("No model rows to build ablation from.")
        return

    cols = ["Model", "Phase", "Params", "Val F1", "Val AUC", "Test F1", "Test AUC",
            "Train (s)", "Best Ep"]

    table_rows = [{
        "Model":     r["name"],
        "Phase":     r["phase"],
        "Params":    r["n_params"],
        "Val F1":    r["val_f1"],
        "Val AUC":   r["val_auc"],
        "Test F1":   r["test_f1"],
        "Test AUC":  r["test_auc"],
        "Train (s)": r["train_s"],
        "Best Ep":   r["best_epoch"],
    } for r in rows]

    # ── CSV ──
    csv_path = RESULTS_DIR / "hetero_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table_rows)
    logger.info("Saved hetero_ablation.csv")

    # ── Markdown ──
    def _sep(c: str) -> str:
        return "-" * max(len(c), 8)

    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(_sep(c) for c in cols) + " |"
    body   = "\n".join(
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for r in table_rows
    )

    auc_by = {r["name"]: r["test_auc"] for r in rows if isinstance(r["test_auc"], float)}
    best_p4 = max(auc_by.get("GraphSAGE", 0), auc_by.get("GAT", 0), auc_by.get("GCN", 0))
    best_p5 = max(auc_by.get("SnapshotGNN", 0), auc_by.get("EvolveGCN", 0))
    best_p6 = max(
        auc_by.get("HeteroSAGE", 0),
        auc_by.get("HeteroGAT (HGAT)", 0),
        auc_by.get("HTGN", 0),
    )

    note = (
        "\n\n## Key Findings\n\n"
        "| Comparison | Test AUC delta |\n"
        "|------------|----------------|\n"
    )
    comparisons = [
        ("Best Phase 6 vs best Phase 4 (GraphSAGE)", best_p6 - best_p4),
        ("Best Phase 6 vs best Phase 5 (SnapshotGNN)", best_p6 - best_p5),
        ("Best Phase 6 vs XGBoost baseline", best_p6 - auc_by.get("XGBoost", 0)),
        ("HTGN vs GraphSAGE (temporal+hetero vs static-homo)", auc_by.get("HTGN", 0) - best_p4),
    ]
    for label, delta in comparisons:
        sign = "+" if delta >= 0 else ""
        note += f"| {label} | {sign}{delta:.4f} |\n"

    note += (
        "\n> **Heterogeneity construction**: Elliptic has one node type. "
        "Direction-typed edges (sends / receives) give each model asymmetric "
        "inductive bias for upstream vs downstream fraud propagation.\n"
        "> **Concept drift**: Illicit prevalence drops 11.6% (train) → 9.2% (val) → "
        "2.5% (test). ROC-AUC is the reliable cross-time metric; Test F1 is low "
        "for all models due to this distribution shift.\n"
    )

    md = (
        "# Phase 6: Heterogeneous GNN Ablation — Elliptic Bitcoin Dataset\n\n"
        + header + "\n" + sep + "\n" + body + "\n" + note
    )
    md_path = RESULTS_DIR / "hetero_ablation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Saved hetero_ablation.md")

    print("\n" + header)
    print(sep)
    print(body)


def plot_ablation_roc(rows: list[dict]) -> None:
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(16, 7))

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
            probs, labels = np.array(probs), np.array(labels)
            fpr, tpr, _ = roc_curve(labels, probs)
            auc   = r[f"{split}_auc"]
            phase = r["phase"]

            is_p6 = phase == "Phase 6"
            lw    = 2.4 if is_p6 else 1.2
            ls    = "-" if is_p6 else "--"
            marker = "★ " if is_p6 else "  "
            ax.plot(fpr, tpr, lw=lw, ls=ls, color=_COLORS.get(r["name"], "gray"),
                    label=f"{marker}{r['name']} AUC={auc:.3f}")

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Phase 6 Heterogeneous GNN Ablation — Elliptic Bitcoin Dataset\n"
        "(★ = Phase 6 heterogeneous models; -- = Phase 4/5 baselines)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = RESULTS_DIR / "phase6_roc_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phase6_roc_comparison.png")


# Patch rows with the split AUC from the metrics dict for ROC plot
def _enrich_rows_auc(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["val_auc"]  = r.get("val_auc",  0.0)
        r["test_auc"] = r.get("test_auc", 0.0)
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: Heterogeneous GNN training")
    parser.add_argument(
        "--models", nargs="+",
        default=["hetero_sage", "hgat", "htgn"],
        choices=["hetero_sage", "hgat", "htgn"],
        help="Which Phase 6 models to train",
    )
    parser.add_argument(
        "--ablation-only", action="store_true",
        help="Skip training; just rebuild the ablation table and ROC plot",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    static_to_train   = [m for m in args.models if m in STATIC_MODELS]
    temporal_to_train = [m for m in args.models if m in TEMPORAL_MODELS]

    phase6_results: list[dict] = []

    if not args.ablation_only:
        # ── Static models need graph.pt ──
        if static_to_train:
            logger.info("Loading graph.pt …")
            data = torch.load(PROCESSED_DIR / "graph.pt", weights_only=False)
            logger.info("  %d nodes  %d edges  %d features",
                        data.num_nodes, data.num_edges, data.x.shape[1])

            train_labels = data.y[data.train_labeled_mask]
            n_licit   = int((train_labels == 0).sum())
            n_illicit = int((train_labels == 1).sum())
            pos_weight = torch.tensor([n_licit / n_illicit], dtype=torch.float32)
            logger.info("  pos_weight=%.3f", pos_weight.item())

            for mn in static_to_train:
                logger.info("\n=== Training %s (Phase 6.1) ===", mn.upper())
                r = train_static(mn, data, pos_weight)
                phase6_results.append(r)

        # ── Temporal models need temporal_snapshots.pt ──
        if temporal_to_train:
            logger.info("Loading temporal_snapshots.pt …")
            snapshots: list = torch.load(
                PROCESSED_DIR / "temporal_snapshots.pt", weights_only=False
            )
            logger.info("  %d snapshots loaded", len(snapshots))

            all_train_y = torch.cat([
                s.y[s.train_labeled_mask]
                for s in snapshots if s.train_labeled_mask.any()
            ])
            n_licit   = int((all_train_y == 0).sum())
            n_illicit = int((all_train_y == 1).sum())
            pos_weight_t = torch.tensor([n_licit / n_illicit], dtype=torch.float32)
            logger.info("  pos_weight=%.3f", pos_weight_t.item())

            for mn in temporal_to_train:
                logger.info("\n=== Training %s (Phase 6.2) ===", mn.upper())
                r = train_temporal(mn, snapshots, pos_weight_t)
                phase6_results.append(r)

    # ── Phase 6.3: Ablation ──
    logger.info("\n=== Phase 6.3: Building ablation table ===")
    rows = _load_ablation_rows()
    rows = _enrich_rows_auc(rows)
    build_ablation(rows)
    plot_ablation_roc(rows)

    # ── Summary ──
    print("\n=== Phase 6 Complete ===")
    if phase6_results:
        for r in phase6_results:
            print(
                f"  {r['model']:<18}  "
                f"val_f1={r['val_metrics']['f1']:.4f}  "
                f"val_auc={r['val_metrics']['roc_auc']:.4f}  "
                f"test_f1={r['test_metrics']['f1']:.4f}  "
                f"test_auc={r['test_metrics']['roc_auc']:.4f}  "
                f"t={r['training_time_s']:.0f}s"
            )
    print(f"  Outputs: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
