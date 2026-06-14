"""Phase 11.2: Model checkpointing & registry.

Builds a single source-of-truth catalogue of every trained model in this repo:
file path, size, SHA-256 checksum, parameter count, hyperparameters, and the
headline val/test metrics — read straight from each model's *_training.json (or
the XGBoost baseline JSONs). The production scoring model is flagged.

Hosting decision (per the $0 / free-tooling constraint)
-------------------------------------------------------
Project plan (Phase 11.2): keep checkpoints small; push to HuggingFace Hub
(free) only if a file exceeds the ~100 MB git-comfortable limit. This script
checks every checkpoint's size and records `git_trackable` (size < 100 MB).
All Elliptic models are < 1 MB, so they live in git and no HF Hub upload is
needed — recorded here so the decision is auditable and re-checks itself if a
larger model is ever added.

Note: checkpoint files (*.pt / *.pkl) are gitignored as binaries; this registry
(plain JSON) is the tracked, reproducible record that points at them.

Usage
-----
    python -m src.models.model_registry
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model_registry")

REPO_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR  = REPO_ROOT / "config"

GIT_SIZE_LIMIT_MB = 100.0   # above this -> recommend HuggingFace Hub instead of git

# Each entry: model file + the JSON holding its metrics + descriptive metadata.
# `metrics_kind` tells us how to parse the metrics file.
CATALOG = [
    {"name": "xgboost", "phase": "2", "file": "baseline_xgboost_model.pkl",
     "metrics_file": ("baseline_metrics.json", "baseline_evaluation.json"),
     "metrics_kind": "xgboost",
     "architecture": "XGBoost (400 trees, depth 6) on 169 node features — no graph"},
    {"name": "gcn", "phase": "4.2", "file": "gcn_model.pt",
     "metrics_file": "gcn_training.json", "metrics_kind": "gnn",
     "architecture": "GCNConv(169->64) x2 + Linear; symmetric-normalised"},
    {"name": "graphsage", "phase": "4.3", "file": "graphsage_model.pt",
     "metrics_file": "graphsage_training.json", "metrics_kind": "gnn",
     "architecture": "SAGEConv(169->64, mean) x2 + Linear"},
    {"name": "gat", "phase": "4.4", "file": "gat_model.pt",
     "metrics_file": "gat_training.json", "metrics_kind": "gnn",
     "architecture": "GATConv(169->32, heads=2) x2 + Linear"},
    {"name": "temporal_snapshot_gnn", "phase": "5.1", "file": "temporal_snapshot_gnn_model.pt",
     "metrics_file": "temporal_snapshot_gnn_training.json", "metrics_kind": "gnn",
     "architecture": "GraphSAGE encoder + GRUCell temporal context over 49 snapshots"},
    {"name": "evolve_gcn", "phase": "5.2", "file": "evolve_gcn_model.pt",
     "metrics_file": "evolve_gcn_training.json", "metrics_kind": "gnn",
     "architecture": "EvolveGCN-O: GRU evolves GCN weight matrices per snapshot"},
    {"name": "hetero_sage", "phase": "6.1", "file": "hetero_sage_model.pt",
     "metrics_file": "hetero_sage_training.json", "metrics_kind": "gnn",
     "architecture": "2x SAGEConv per layer (sends + receives directions), summed"},
    {"name": "hgat", "phase": "6.1", "file": "hgat_model.pt",
     "metrics_file": "hgat_training.json", "metrics_kind": "gnn",
     "architecture": "2x GATConv per layer (sends + receives directions), summed"},
    {"name": "htgn", "phase": "6.2", "file": "htgn_model.pt",
     "metrics_file": "htgn_training.json", "metrics_kind": "gnn",
     "architecture": "HeteroSAGE encoder + GRU temporal context (MAIN hetero-temporal model)"},
]

PRODUCTION_MODEL = "graphsage"   # best test AUC=0.777; used by Phases 7-10


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def extract_metrics(entry: dict) -> dict:
    """Pull (val/test) F1, precision, recall, AUC + hparams from the metric file(s)."""
    if entry["metrics_kind"] == "xgboost":
        m_name, e_name = entry["metrics_file"]
        m, e = load_json(m_name), load_json(e_name)
        return {
            "n_params": None,
            "hyperparams": m.get("best_params", {}) | {"scale_pos_weight": m.get("scale_pos_weight")},
            "val_threshold": m.get("val_threshold"),
            "val":  {"f1": m.get("val_f1"), "roc_auc": m.get("val_roc_auc")},
            "test": {"f1": e.get("test_f1"), "roc_auc": e.get("test_roc_auc")},
        }
    # gnn
    d = load_json(entry["metrics_file"])
    return {
        "n_params": d.get("n_params"),
        "hyperparams": d.get("hyperparams", {}),
        "val_threshold": d.get("val_threshold"),
        "val":  d.get("val_metrics", {}),
        "test": d.get("test_metrics", {}),
    }


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    models = []
    missing = []
    for entry in CATALOG:
        path = RESULTS_DIR / entry["file"]
        rec = {
            "name": entry["name"],
            "phase": entry["phase"],
            "architecture": entry["architecture"],
            "checkpoint": f"results/{entry['file']}",
            "is_production": entry["name"] == PRODUCTION_MODEL,
        }
        if not path.exists():
            rec["status"] = "MISSING (re-run its training script to regenerate)"
            missing.append(entry["name"])
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            rec["status"] = "present"
            rec["size_mb"] = round(size_mb, 4)
            rec["sha256"] = sha256_of(path)
            rec["git_trackable"] = size_mb < GIT_SIZE_LIMIT_MB
            rec["hosting"] = ("git (size < 100 MB)" if size_mb < GIT_SIZE_LIMIT_MB
                              else "HuggingFace Hub (free) — exceeds git-comfortable size")
        rec.update(extract_metrics(entry))
        models.append(rec)
        logger.info("  %-22s %-8s %s",
                    entry["name"],
                    f"{rec.get('size_mb', '—')} MB" if "size_mb" in rec else "MISSING",
                    rec.get("hosting", ""))

    present = [m for m in models if m["status"] == "present"]
    max_mb = max((m["size_mb"] for m in present), default=0.0)

    registry = {
        "phase": "11.2 model checkpointing & registry",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_size_limit_mb": GIT_SIZE_LIMIT_MB,
        "production_model": PRODUCTION_MODEL,
        "production_rationale": "highest cross-time test ROC-AUC (0.777); used for scoring in Phases 7-10",
        "hosting_summary": {
            "largest_checkpoint_mb": round(max_mb, 4),
            "all_git_trackable": all(m.get("git_trackable", False) for m in present),
            "hf_hub_required": any(not m.get("git_trackable", True) for m in present),
            "note": ("All checkpoints are < 100 MB, so they stay in git. "
                     "Checkpoint binaries themselves are gitignored (*.pt/*.pkl); "
                     "this registry is the tracked record. If a future model exceeds "
                     "100 MB, upload it to the free HuggingFace Hub and set its "
                     "`checkpoint` to the hub URL."),
        },
        "models": models,
    }
    if missing:
        registry["missing_models"] = missing

    out = CONFIG_DIR / "model_registry.json"
    with open(out, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Saved %s (%d models, %d present)",
                out, len(models), len(present))

    print("\n=== Phase 11.2 Model Registry — summary ===")
    print(f"  {'model':<22}{'size':>9}  {'test AUC':>9}  hosting")
    for m in models:
        tauc = m.get("test", {}).get("roc_auc")
        tauc_s = f"{tauc:.3f}" if isinstance(tauc, (int, float)) else "  —  "
        size_s = f"{m['size_mb']:.3f}MB" if "size_mb" in m else "MISSING"
        flag = "  <- production" if m["is_production"] else ""
        print(f"  {m['name']:<22}{size_s:>9}  {tauc_s:>9}  {m.get('hosting','')}{flag}")
    print(f"  Largest checkpoint: {max_mb:.3f} MB  (git limit {GIT_SIZE_LIMIT_MB} MB)")
    print(f"  All git-trackable: {registry['hosting_summary']['all_git_trackable']}  "
          f"=> HF Hub required: {registry['hosting_summary']['hf_hub_required']}")
    print(f"  Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
