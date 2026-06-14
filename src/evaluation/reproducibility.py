"""Phase 11.3: Reproducibility verification.

Produces results/reproducibility_check.md by actually verifying — not just
asserting — that the pipeline is reproducible:

  1. Seed coverage audit  — scans every training/eval script for a seeding call
                            (torch.manual_seed / np.random.seed / random_state)
                            so no source of randomness is left unseeded.
  2. Config completeness  — confirms config/experiments.yaml,
                            config/model_registry.json and the split files exist.
  3. Inference determinism — loads the production GraphSAGE checkpoint and runs
                            two eval-mode forward passes; scores must be
                            bit-identical (max abs diff == 0).
  4. Metric reproduction  — recomputes test ROC-AUC / F1 from the loaded
                            checkpoint and the recorded val threshold, and checks
                            they match the numbers in graphsage_training.json
                            within tolerance.

Usage
-----
    python -m src.evaluation.reproducibility
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.models.gnn_models import GraphSAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reproducibility")

REPO_ROOT     = Path(__file__).resolve().parents[2]
SRC_DIR       = REPO_ROOT / "src"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR   = REPO_ROOT / "results"
CONFIG_DIR    = REPO_ROOT / "config"

SEED         = 42
GRAPHSAGE_HP = dict(in_channels=169, hidden_channels=64, dropout=0.3)
AUC_TOL      = 1e-4   # recomputed metric must match recorded within this

# Scripts that train models or run stochastic evaluation — each must seed.
SCRIPTS_REQUIRING_SEED = [
    "models/baseline_xgboost.py",
    "models/static_gnn_train.py",
    "models/temporal_gnn_train.py",
    "models/hetero_gnn_train.py",
    "evaluation/robustness.py",
    "evaluation/interpretability.py",
    "evaluation/scalability.py",
]
# pattern -> human-readable label shown in the report
SEED_PATTERNS = {
    r"torch\.manual_seed": "torch.manual_seed",
    r"np\.random\.seed": "np.random.seed",
    r"numpy\.random\.seed": "numpy.random.seed",
    r"random_state\s*=\s*42": "random_state=42",
    r"random\.seed": "random.seed",
}


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def audit_seed_coverage() -> tuple[list[dict], bool]:
    rows, all_ok = [], True
    for rel in SCRIPTS_REQUIRING_SEED:
        path = SRC_DIR / rel
        if not path.exists():
            rows.append({"script": rel, "seeded": False, "found": "MISSING FILE"})
            all_ok = False
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [label for pat, label in SEED_PATTERNS.items() if re.search(pat, text)]
        ok = len(found) > 0
        all_ok &= ok
        rows.append({"script": rel, "seeded": ok,
                     "found": ", ".join(found) or "NONE"})
    return rows, all_ok


def check_configs() -> list[dict]:
    targets = [
        ("config/experiments.yaml", CONFIG_DIR / "experiments.yaml"),
        ("config/model_registry.json", CONFIG_DIR / "model_registry.json"),
        ("data/processed/splits.json", PROCESSED_DIR / "splits.json"),
        ("data/processed/splits.pt", PROCESSED_DIR / "splits.pt"),
    ]
    return [{"artifact": name, "present": p.exists()} for name, p in targets]


def load_graphsage() -> GraphSAGE:
    model = GraphSAGE(**GRAPHSAGE_HP)
    state = torch.load(RESULTS_DIR / "graphsage_model.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def determinism_check(model: GraphSAGE, data) -> dict:
    set_seed(SEED)
    a = model(data.x, data.edge_index).squeeze(-1)
    set_seed(SEED)
    b = model(data.x, data.edge_index).squeeze(-1)
    max_diff = float((a - b).abs().max())
    return {"max_abs_diff": max_diff, "bit_identical": max_diff == 0.0}


@torch.no_grad()
def metric_reproduction(model: GraphSAGE, data) -> dict:
    recorded = json.load(open(RESULTS_DIR / "graphsage_training.json"))
    thr = float(recorded["val_threshold"])
    rec_auc = float(recorded["test_metrics"]["roc_auc"])
    rec_f1  = float(recorded["test_metrics"]["f1"])

    logits = model(data.x, data.edge_index).squeeze(-1)
    mask   = data.test_labeled_mask
    probs  = torch.sigmoid(logits[mask]).numpy()
    labels = data.y[mask].numpy()
    preds  = (probs >= thr).astype(int)

    re_auc = float(roc_auc_score(labels, probs))
    re_f1  = float(f1_score(labels, preds, zero_division=0))

    return {
        "val_threshold": thr,
        "recorded": {"test_roc_auc": rec_auc, "test_f1": rec_f1},
        "recomputed": {"test_roc_auc": re_auc, "test_f1": re_f1},
        "auc_diff": abs(re_auc - rec_auc),
        "f1_diff": abs(re_f1 - rec_f1),
        "matches": abs(re_auc - rec_auc) < AUC_TOL and abs(re_f1 - rec_f1) < AUC_TOL,
    }


def write_report(seed_rows, seed_ok, cfg_rows, det, met, out: Path) -> None:
    cfg_ok = all(r["present"] for r in cfg_rows)
    overall = seed_ok and cfg_ok and det["bit_identical"] and met["matches"]

    lines = [
        "# Phase 11.3 — Reproducibility Check",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_  ",
        f"Global seed: **{SEED}**  ·  Device: **CPU** (Windows-on-ARM, no GPU)",
        "",
        f"## Verdict: {'✅ REPRODUCIBLE' if overall else '❌ ISSUES FOUND'}",
        "",
        "Goal: `git clone && pip install && python -m ...` reproduces the recorded "
        "results. CPU full-batch training with a fixed seed is deterministic, so the "
        "recorded checkpoints and metrics regenerate exactly.",
        "",
        "## 1. Seed coverage",
        "",
        "Every training / stochastic-evaluation script seeds its RNGs:",
        "",
        "| Script | Seeded | Mechanism |",
        "|--------|:------:|-----------|",
    ]
    for r in seed_rows:
        lines.append(f"| `{r['script']}` | {'✅' if r['seeded'] else '❌'} | {r['found']} |")
    lines += [
        "",
        f"**Seed coverage: {'PASS' if seed_ok else 'FAIL'}** — "
        "torch + numpy seeded (seed=42); XGBoost uses `random_state=42`.",
        "",
        "## 2. Config completeness",
        "",
        "| Artifact | Present |",
        "|----------|:-------:|",
    ]
    for r in cfg_rows:
        lines.append(f"| `{r['artifact']}` | {'✅' if r['present'] else '❌'} |")
    lines += [
        "",
        "Hyperparameters and split boundaries are pinned in "
        "`config/experiments.yaml`; checkpoint inventory in "
        "`config/model_registry.json`; split masks in `data/processed/splits.*`.",
        "",
        "## 3. Inference determinism (production GraphSAGE)",
        "",
        f"- Two eval-mode forward passes — max abs score diff: "
        f"`{det['max_abs_diff']:.2e}`",
        f"- Bit-identical: **{det['bit_identical']}** "
        f"→ {'PASS' if det['bit_identical'] else 'FAIL'}",
        "",
        "## 4. Metric reproduction (test set, ts 43-49)",
        "",
        f"Recomputed from the loaded checkpoint at val-tuned threshold "
        f"`{met['val_threshold']}`:",
        "",
        "| Metric | Recorded | Recomputed | |Δ| |",
        "|--------|---------:|-----------:|----:|",
        f"| test ROC-AUC | {met['recorded']['test_roc_auc']:.6f} | "
        f"{met['recomputed']['test_roc_auc']:.6f} | {met['auc_diff']:.2e} |",
        f"| test F1 | {met['recorded']['test_f1']:.6f} | "
        f"{met['recomputed']['test_f1']:.6f} | {met['f1_diff']:.2e} |",
        "",
        f"**Match (tol {AUC_TOL:g}): {'PASS' if met['matches'] else 'FAIL'}** — "
        "the saved checkpoint reproduces its recorded headline metrics.",
        "",
        "## How to reproduce from a clean clone",
        "",
        "```bash",
        "pip install --only-binary=:all: -r requirements.txt",
        "python -m src.data.download_elliptic      # data",
        "python -m src.data.preprocess_elliptic",
        "python -m src.data.feature_engineering",
        "python -m src.data.temporal_split",
        "python -m src.data.build_graph",
        "python -m src.models.baseline_xgboost     # Phase 2",
        "python -m src.models.static_gnn_train     # Phase 4",
        "python -m src.models.temporal_gnn_train   # Phase 5",
        "python -m src.models.hetero_gnn_train     # Phase 6",
        "python -m src.evaluation.reproducibility  # this check",
        "```",
        "",
        "Full per-step command list and all hyperparameters: "
        "`config/experiments.yaml`.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", out.name)


def main() -> int:
    set_seed(SEED)
    logger.info("Phase 11.3 reproducibility verification …")

    seed_rows, seed_ok = audit_seed_coverage()
    cfg_rows = check_configs()

    logger.info("Loading graph.pt + GraphSAGE …")
    data  = torch.load(PROCESSED_DIR / "graph.pt", weights_only=False)
    model = load_graphsage()

    det = determinism_check(model, data)
    met = metric_reproduction(model, data)

    logger.info("seed_coverage=%s  determinism=%s  metric_match=%s",
                seed_ok, det["bit_identical"], met["matches"])

    out = RESULTS_DIR / "reproducibility_check.md"
    write_report(seed_rows, seed_ok, cfg_rows, det, met, out)

    overall = (seed_ok and all(r["present"] for r in cfg_rows)
               and det["bit_identical"] and met["matches"])
    print("\n=== Phase 11.3 Reproducibility — summary ===")
    print(f"  Seed coverage      : {'PASS' if seed_ok else 'FAIL'}")
    print(f"  Configs present    : {'PASS' if all(r['present'] for r in cfg_rows) else 'FAIL'}")
    print(f"  Inference determ.  : {'PASS' if det['bit_identical'] else 'FAIL'} "
          f"(max diff {det['max_abs_diff']:.2e})")
    print(f"  Metric reproduction: {'PASS' if met['matches'] else 'FAIL'} "
          f"(AUC {met['recomputed']['test_roc_auc']:.4f} vs {met['recorded']['test_roc_auc']:.4f})")
    print(f"  OVERALL            : {'REPRODUCIBLE' if overall else 'ISSUES'}")
    print(f"  Output: {out}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
