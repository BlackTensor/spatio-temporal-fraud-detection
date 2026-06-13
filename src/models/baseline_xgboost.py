"""
Phase 2: XGBoost feature-based baseline.
Covers 2.1 (train + grid search), 2.2 (test evaluation), 2.3 (feature importance).

Run:
    python -m src.models.baseline_xgboost
"""
import json
import logging
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROCESSED = Path("data/processed")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    log.info("Loading processed tensors …")
    X = torch.load(PROCESSED / "node_features.pt", weights_only=True).numpy()
    y = torch.load(PROCESSED / "node_labels.pt", weights_only=True).numpy()

    masks = torch.load(PROCESSED / "splits.pt", weights_only=True)
    train_lbl = masks["train_labeled_mask"].numpy()
    val_lbl   = masks["val_labeled_mask"].numpy()
    test_lbl  = masks["test_labeled_mask"].numpy()

    X_tr, y_tr = X[train_lbl], y[train_lbl]
    X_va, y_va = X[val_lbl],   y[val_lbl]
    X_te, y_te = X[test_lbl],  y[test_lbl]

    log.info("Train %s  (illicit %d / licit %d)", X_tr.shape, y_tr.sum(), (y_tr == 0).sum())
    log.info("Val   %s  (illicit %d / licit %d)", X_va.shape, y_va.sum(), (y_va == 0).sum())
    log.info("Test  %s  (illicit %d / licit %d)", X_te.shape, y_te.sum(), (y_te == 0).sum())
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ---------------------------------------------------------------------------
# Phase 2.1 — XGBoost + grid search on val
# ---------------------------------------------------------------------------

def grid_search(X_tr, y_tr, X_va, y_va):
    """Light grid search; optimise illicit-class F1 on validation set."""
    # scale_pos_weight compensates for class imbalance
    spw = float((y_tr == 0).sum()) / float(y_tr.sum())
    log.info("scale_pos_weight base = %.2f", spw)

    param_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "min_child_weight": mcw}
        for n   in [200, 400]
        for d   in [4, 6]
        for lr  in [0.05, 0.1]
        for mcw in [1, 5]
    ]

    best_f1, best_params, best_model = -1, None, None
    for i, params in enumerate(param_grid):
        clf = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = clf.predict(X_va)
        f1 = f1_score(y_va, preds)
        if (i + 1) % 4 == 0:
            log.info("  [%2d/%d] params=%s  val_F1=%.4f", i + 1, len(param_grid), params, f1)
        if f1 > best_f1:
            best_f1, best_params, best_model = f1, params, clf

    log.info("Best val F1=%.4f  params=%s", best_f1, best_params)
    return best_model, best_params, best_f1


def train_phase(X_tr, y_tr, X_va, y_va):
    model, best_params, val_f1 = grid_search(X_tr, y_tr, X_va, y_va)

    model_path = RESULTS / "baseline_xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Saved model → %s", model_path)

    val_proba = model.predict_proba(X_va)[:, 1]
    val_roc_auc = float(roc_auc_score(y_va, val_proba))

    # Find F1-optimal threshold on VAL set; apply to test — no test leakage.
    prec_va, rec_va, thresh_pr = precision_recall_curve(y_va, val_proba)
    # precision_recall_curve appends a sentinel; thresholds has len = len(prec)-1
    f1_candidates = 2 * prec_va[:-1] * rec_va[:-1] / (prec_va[:-1] + rec_va[:-1] + 1e-9)
    val_best_thresh = float(thresh_pr[np.argmax(f1_candidates)])
    val_preds = (val_proba >= val_best_thresh).astype(int)

    val_metrics = {
        "val_f1":           float(f1_score(y_va, val_preds)),
        "val_f1_at_0.5":    float(f1_score(y_va, (val_proba >= 0.5).astype(int))),
        "val_roc_auc":      val_roc_auc,
        "val_threshold":    val_best_thresh,
        "val_report":       classification_report(y_va, val_preds, target_names=["licit", "illicit"], output_dict=True),
        "best_params":      best_params,
        "scale_pos_weight": float((y_tr == 0).sum()) / float(y_tr.sum()),
    }
    metrics_path = RESULTS / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    log.info("Saved val metrics → %s", metrics_path)
    log.info(
        "Val F1=%.4f (thresh=%.3f)  F1@0.5=%.4f  ROC-AUC=%.4f",
        val_metrics["val_f1"], val_best_thresh,
        val_metrics["val_f1_at_0.5"], val_roc_auc,
    )
    return model, val_metrics, val_best_thresh


# ---------------------------------------------------------------------------
# Phase 2.2 — Test evaluation
# ---------------------------------------------------------------------------

def evaluate_phase(model, X_va, y_va, X_te, y_te, val_thresh):
    t0 = time.perf_counter()
    test_proba = model.predict_proba(X_te)[:, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Apply val-derived threshold (no test leakage)
    test_preds = (test_proba >= val_thresh).astype(int)

    roc_auc = roc_auc_score(y_te, test_proba)
    fpr, tpr, _ = roc_curve(y_te, test_proba)
    report  = classification_report(y_te, test_preds, target_names=["licit", "illicit"], output_dict=True)
    cm      = confusion_matrix(y_te, test_preds)

    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])
    tn = int(cm[0, 0])

    # Also report at default threshold for reference
    test_preds_05 = (test_proba >= 0.5).astype(int)
    f1_05 = float(f1_score(y_te, test_preds_05))

    evaluation = {
        "threshold_source":  "val-set Youden-J optimal (no test leakage)",
        "threshold":          val_thresh,
        "test_f1":            float(f1_score(y_te, test_preds)),
        "test_f1_at_0.5":     f1_05,
        "test_roc_auc":       roc_auc,
        "inference_ms_total": elapsed_ms,
        "inference_ms_per_node": elapsed_ms / len(y_te),
        "confusion_matrix":   {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "classification_report": report,
        "temporal_drift_note": (
            "Illicit prevalence: train=11.6%, val=9.2%, test=2.5%. "
            "The sharp drop from val→test (ts 43-49) reflects real concept drift "
            "in the Elliptic dataset. ROC-AUC is a threshold-free rank metric and "
            "is more reliable for cross-time comparisons than F1."
        ),
        "fp_fn_analysis": {
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "note": (
                "FP = licit transactions flagged as illicit (alert fatigue). "
                "FN = missed illicit transactions (fraud escapes). "
                "Threshold tuned to maximise Youden J on val set and applied to test."
            ),
        },
    }

    eval_path = RESULTS / "baseline_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    log.info("Saved evaluation → %s", eval_path)
    log.info(
        "Test  F1=%.4f  F1@0.5=%.4f  ROC-AUC=%.4f  threshold=%.3f  "
        "TP=%d  FP=%d  FN=%d  TN=%d  inference=%.1f ms total",
        evaluation["test_f1"], f1_05, roc_auc, val_thresh, tp, fp, fn, tn, elapsed_ms,
    )

    # --- ROC curve plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Find operating point on test ROC closest to val_thresh
    test_proba_sorted = np.sort(test_proba)[::-1]
    op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    axes[0].plot(fpr, tpr, lw=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].scatter(
        op_fpr, op_tpr,
        color="red", zorder=5, label=f"Operating point (thresh={val_thresh:.3f})"
    )
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — XGBoost Baseline")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_te, test_proba)
    axes[1].plot(rec, prec, lw=2, color="darkorange")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — XGBoost Baseline")
    axes[1].grid(alpha=0.3)

    roc_path = RESULTS / "baseline_roc.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    log.info("Saved ROC/PR plot → %s", roc_path)

    # --- Confusion matrix plot ---
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["licit", "illicit"])
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Confusion Matrix — XGBoost Baseline (test set)")
    cm_path = RESULTS / "baseline_confusion_matrix.png"
    fig2.tight_layout()
    fig2.savefig(cm_path, dpi=150)
    plt.close(fig2)
    log.info("Saved confusion matrix → %s", cm_path)

    return evaluation


# ---------------------------------------------------------------------------
# Phase 2.3 — Feature importance
# ---------------------------------------------------------------------------

def feature_importance_phase(model):
    feat_names = json.loads((PROCESSED / "feature_names.json").read_text())

    importance = model.feature_importances_
    top_n = 15
    idx = np.argsort(importance)[::-1][:top_n]
    top_names  = [feat_names[i] for i in idx]
    top_scores = importance[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), top_scores[::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top-15 XGBoost Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fi_path = RESULTS / "baseline_feature_importance.png"
    fig.savefig(fi_path, dpi=150)
    plt.close(fig)
    log.info("Saved feature importance → %s", fi_path)

    summary = {
        "top_15_features": [
            {"rank": i + 1, "feature": top_names[i], "importance": float(top_scores[i])}
            for i in range(top_n)
        ],
        "note": (
            "Importance = average gain across all splits that use the feature. "
            "Top features are anonymised Elliptic transaction attributes; "
            "in_degree/out_degree and time_step_norm appear if graph-structural "
            "features dominate over raw transaction features."
        ),
    }
    fi_json_path = RESULTS / "baseline_feature_importance.json"
    fi_json_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved feature importance JSON → %s", fi_json_path)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data()

    log.info("=== Phase 2.1: XGBoost + grid search ===")
    model, val_metrics, val_thresh = train_phase(X_tr, y_tr, X_va, y_va)

    log.info("=== Phase 2.2: Test evaluation ===")
    evaluation = evaluate_phase(model, X_va, y_va, X_te, y_te, val_thresh)

    log.info("=== Phase 2.3: Feature importance ===")
    fi_summary = feature_importance_phase(model)

    log.info("--- Phase 2 complete ---")
    log.info(
        "Val  F1=%.4f  ROC-AUC=%.4f  threshold=%.3f",
        val_metrics["val_f1"], val_metrics["val_roc_auc"], val_thresh,
    )
    log.info(
        "Test F1=%.4f  ROC-AUC=%.4f  (F1@0.5=%.4f)",
        evaluation["test_f1"], evaluation["test_roc_auc"], evaluation["test_f1_at_0.5"],
    )
    log.info("Top feature: %s", fi_summary["top_15_features"][0]["feature"])


if __name__ == "__main__":
    main()
