"""Phase 5: Temporal GNN training — TemporalSnapshotGNN and EvolveGCN.

Training strategy
-----------------
Both models process snapshots sequentially (ordered by time step).

Per epoch:
  - Reset temporal state (GRU hidden / weight matrices)
  - Loop over train snapshots in order (ts 1-34)
  - For each snapshot: forward → compute loss on labeled nodes → backward → step
  - Temporal state is detached after each snapshot (TBPTT-1):
      this keeps memory linear in the number of nodes per snapshot while
      still letting the GRU/RNN parameters learn from per-snapshot gradients

Evaluation:
  - Replay full sequence (ts 1-49) with model.eval()
  - Collect predictions for val (ts 35-42) and test (ts 43-49) labeled nodes
  - Compute AUC, F1, precision, recall

Usage
-----
    python -m src.models.temporal_gnn_train
    python -m src.models.temporal_gnn_train --models temporal_snapshot_gnn
    python -m src.models.temporal_gnn_train --models evolve_gcn
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

from src.models.temporal_gnn_models import TemporalSnapshotGNN, EvolveGCN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("temporal_gnn_train")

REPO_ROOT     = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR   = REPO_ROOT / "results"

SEED = 42

HP = {
    "hidden_channels": 64,
    "dropout":         0.3,
    "lr":              0.005,
    "weight_decay":    5e-4,
    "max_epochs":      200,
    "patience":        25,
}


# ── Utilities ────────────────────────────────────────────────────────────────

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


# ── Replay (full sequence inference) ─────────────────────────────────────────

@torch.no_grad()
def replay_sequence(
    model: torch.nn.Module,
    snapshots: list,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run full snapshot sequence; collect val and test predictions.

    Returns (val_probs, val_labels, test_probs, test_labels).
    """
    model.eval()
    device = next(model.parameters()).device

    if model_name == "temporal_snapshot_gnn":
        state = model.init_state(device)
    else:
        state = model.init_state(device)

    val_probs_list:  list[np.ndarray] = []
    val_labels_list: list[np.ndarray] = []
    test_probs_list:  list[np.ndarray] = []
    test_labels_list: list[np.ndarray] = []

    for snap in snapshots:
        x          = snap.x.to(device)
        edge_index = snap.edge_index.to(device)

        if model_name == "temporal_snapshot_gnn":
            logits, state = model.forward_snapshot(x, edge_index, state)
        else:
            logits, state = model.forward_snapshot(x, edge_index, state)

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        if snap.val_labeled_mask.any():
            val_probs_list.append(probs[snap.val_labeled_mask.numpy()])
            val_labels_list.append(snap.y[snap.val_labeled_mask].numpy())

        if snap.test_labeled_mask.any():
            test_probs_list.append(probs[snap.test_labeled_mask.numpy()])
            test_labels_list.append(snap.y[snap.test_labeled_mask].numpy())

    val_probs  = np.concatenate(val_probs_list)
    val_labels = np.concatenate(val_labels_list)
    test_probs  = np.concatenate(test_probs_list)  if test_probs_list  else np.array([])
    test_labels = np.concatenate(test_labels_list) if test_labels_list else np.array([])

    return val_probs, val_labels, test_probs, test_labels


# ── Core trainer ─────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    snapshots:  list,
    pos_weight: torch.Tensor,
) -> dict:
    set_seed(SEED)

    in_channels = snapshots[0].x.shape[1]
    device      = torch.device("cpu")

    if model_name == "temporal_snapshot_gnn":
        model: torch.nn.Module = TemporalSnapshotGNN(
            in_channels=in_channels,
            hidden_channels=HP["hidden_channels"],
            dropout=HP["dropout"],
        )
    else:
        model = EvolveGCN(
            in_channels=in_channels,
            hidden_channels=HP["hidden_channels"],
            dropout=HP["dropout"],
        )

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[%s] %d parameters", model_name.upper(), n_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"],
    )

    # Split snapshots by time
    train_snaps = [s for s in snapshots if s.train_mask.any()]
    # val/test evaluated via replay_sequence on ALL snapshots

    best_val_auc  = -1.0
    best_epoch    = 0
    best_state    = None
    no_improve    = 0
    history: list[dict] = []

    t_start = time.perf_counter()

    for epoch in range(1, HP["max_epochs"] + 1):
        model.train()

        # Re-init temporal state at the start of each epoch
        state = model.init_state(device)

        epoch_loss = 0.0
        n_batches  = 0

        for snap in train_snaps:
            x          = snap.x.to(device)
            edge_index = snap.edge_index.to(device)

            if model_name == "temporal_snapshot_gnn":
                logits, state_new = model.forward_snapshot(x, edge_index, state)
            else:
                logits, state_new = model.forward_snapshot(x, edge_index, state)

            mask = snap.train_labeled_mask
            if not mask.any():
                state = _detach_state(state_new)
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

            # TBPTT-1: detach state so we don't backprop through it next step
            state = _detach_state(state_new)

        # ── validation ──
        val_probs, val_labels, _, _ = replay_sequence(model, snapshots, model_name)
        val_auc = float(roc_auc_score(val_labels, val_probs))
        val_f1  = float(f1_score(val_labels, (val_probs >= 0.5).astype(int), zero_division=0))
        avg_loss = epoch_loss / max(n_batches, 1)

        history.append({
            "epoch":    epoch,
            "loss":     round(avg_loss, 6),
            "val_f1":   round(val_f1, 6),
            "val_auc":  round(val_auc, 6),
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
            logger.info(
                "  [%s] early stop ep %d (best=%d)",
                model_name.upper(), epoch, best_epoch,
            )
            break

    training_time_s = time.perf_counter() - t_start

    # ── reload best checkpoint ──
    model.load_state_dict(best_state)

    # ── threshold from val ──
    val_probs, val_labels, test_probs, test_labels = replay_sequence(
        model, snapshots, model_name
    )
    best_thresh, _ = find_best_threshold(val_probs, val_labels)

    val_metrics  = compute_metrics(val_probs,  val_labels,  best_thresh)
    test_metrics = compute_metrics(test_probs, test_labels, best_thresh)

    val_report = classification_report(
        val_labels,  (val_probs  >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )
    test_report = classification_report(
        test_labels, (test_probs >= best_thresh).astype(int),
        target_names=["licit", "illicit"], output_dict=True, zero_division=0,
    )

    # ── inference latency ──
    infer_ms = _time_inference(model, snapshots, model_name)
    total_nodes = sum(s.num_nodes for s in snapshots)
    infer_ms_per_1k = infer_ms / (total_nodes / 1000)

    result = {
        "model":               model_name,
        "n_params":            n_params,
        "best_epoch":          best_epoch,
        "val_threshold":       round(best_thresh, 4),
        "training_time_s":     round(training_time_s, 2),
        "inference_ms_full":   round(infer_ms, 2),
        "inference_ms_per_1k": round(infer_ms_per_1k, 4),
        "val_metrics":         {k: round(v, 6) for k, v in val_metrics.items()},
        "test_metrics":        {k: round(v, 6) for k, v in test_metrics.items()},
        "val_report":          val_report,
        "test_report":         test_report,
        "hyperparams":         HP,
        "history":             history,
        "val_probs":           val_probs.tolist(),
        "val_labels":          val_labels.tolist(),
        "test_probs":          test_probs.tolist(),
        "test_labels":         test_labels.tolist(),
    }

    # ── save model ──
    model_path = RESULTS_DIR / f"{model_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("  Saved %s", model_path.name)

    training_log = {k: v for k, v in result.items()
                    if k not in ("val_probs", "val_labels", "test_probs", "test_labels")}
    log_path = RESULTS_DIR / f"{model_name}_training.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info("  Saved %s", log_path.name)

    # ── save prob arrays as sidecar (used by temporal_ablation.py for ROC plots) ──
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
        val_metrics["f1"], val_metrics["roc_auc"],
        test_metrics["f1"], test_metrics["roc_auc"],
        training_time_s,
    )
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detach_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach()
    return tuple(s.detach() for s in state)


@torch.no_grad()
def _time_inference(model, snapshots, model_name, n_runs: int = 3) -> float:
    model.eval()
    device = next(model.parameters()).device
    times = []
    for _ in range(n_runs):
        state = model.init_state(device)
        t0 = time.perf_counter()
        for snap in snapshots:
            _, state = model.forward_snapshot(
                snap.x.to(device), snap.edge_index.to(device), state
            )
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)


# ── ROC plot ─────────────────────────────────────────────────────────────────

def plot_roc(results: list[dict]) -> None:
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        "temporal_snapshot_gnn": "#e74c3c",
        "evolve_gcn":            "#3498db",
    }
    for ax, split, title in [
        (ax_val,  "val",  "Validation (ts 35-42)"),
        (ax_test, "test", "Test (ts 43-49)"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        for r in results:
            probs  = np.array(r[f"{split}_probs"])
            labels = np.array(r[f"{split}_labels"])
            fpr, tpr, _ = roc_curve(labels, probs)
            auc = r[f"{split}_metrics"]["roc_auc"]
            label = r["model"].replace("_", " ").upper()
            ax.plot(fpr, tpr, lw=1.8, color=colors.get(r["model"], "gray"),
                    label=f"{label} AUC={auc:.3f}")
        ax.set_xlabel("FPR", fontsize=11)
        ax.set_ylabel("TPR", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Phase 5: Temporal GNN ROC — Elliptic Bitcoin Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = RESULTS_DIR / "phase5_roc_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved phase5_roc_comparison.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["temporal_snapshot_gnn", "evolve_gcn"],
        choices=["temporal_snapshot_gnn", "evolve_gcn"],
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading temporal_snapshots.pt …")
    snapshots: list = torch.load(
        PROCESSED_DIR / "temporal_snapshots.pt", weights_only=False
    )
    logger.info("  %d snapshots loaded", len(snapshots))

    # pos_weight from all train-labeled nodes across snapshots
    all_train_y = torch.cat([
        s.y[s.train_labeled_mask] for s in snapshots if s.train_labeled_mask.any()
    ])
    n_licit   = int((all_train_y == 0).sum())
    n_illicit = int((all_train_y == 1).sum())
    pos_weight = torch.tensor([n_licit / n_illicit], dtype=torch.float32)
    logger.info("  pos_weight=%.3f  (licit=%d, illicit=%d in train)",
                pos_weight.item(), n_licit, n_illicit)

    all_results: list[dict] = []
    for model_name in args.models:
        logger.info("\n=== Training %s ===", model_name.upper())
        result = train_model(model_name, snapshots, pos_weight)
        all_results.append(result)

    if len(all_results) > 1:
        plot_roc(all_results)

    print("\n=== Phase 5 Results ===")
    for r in all_results:
        print(
            f"  {r['model']:<28}  "
            f"val_f1={r['val_metrics']['f1']:.4f}  "
            f"val_auc={r['val_metrics']['roc_auc']:.4f}  "
            f"test_f1={r['test_metrics']['f1']:.4f}  "
            f"test_auc={r['test_metrics']['roc_auc']:.4f}  "
            f"t={r['training_time_s']:.0f}s"
        )
    print(f"\n  Outputs: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
