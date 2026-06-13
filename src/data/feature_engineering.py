"""Phase 1.4: Feature Engineering for the Elliptic Bitcoin dataset.

Adapts the generic Phase 1.4 plan to the Elliptic transaction graph.
The dataset has only one node type (Bitcoin transactions), not the
user/account/device/IP structure assumed in the plan. Node features:

  Original 165 features (feat_1 … feat_165)
  ├── feat_1        : time step (integer 1-49; redundant but kept for PyG compat)
  ├── feat_2-94     : local transaction-level info (amounts, fees, I/O counts, …)
  └── feat_95-165   : 1-hop aggregated neighbour features (mean/max of local feats)

  4 NEW engineered features (appended)
  ├── in_degree     : number of incoming directed edges
  ├── out_degree    : number of outgoing directed edges
  ├── total_degree  : in_degree + out_degree
  └── time_step_norm: time step normalised to [0, 1]  (= (t-1)/48)

Note on scaler scope: StandardScaler is fit on ALL nodes.  In the Elliptic
setting the full graph structure is observed during both training and inference
(transductive), so using global degree statistics for scaling is standard
practice and does not constitute label leakage.  If future phases move to an
inductive setting, the scaler should be refit on training nodes only.

Outputs
-------
data/processed/node_features.pt          float32 [N, 169]  (overwrites Phase 1.3 file)
data/processed/feature_scaler.pkl        sklearn StandardScaler for the 4 new features
data/processed/feature_names.json        list of 169 feature names
data/processed/feature_engineering_report.json

Usage
-----
    python -m src.data.feature_engineering
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_engineering")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_ORIG = 165


# ---------------------------------------------------------------------------
# Load processed tensors
# ---------------------------------------------------------------------------

def load_processed() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info("Loading processed tensors …")
    features = torch.load(PROCESSED_DIR / "node_features.pt", weights_only=True)
    times = torch.load(PROCESSED_DIR / "node_times.pt", weights_only=True)
    edge_index = torch.load(PROCESSED_DIR / "edge_index.pt", weights_only=True)
    logger.info("  node_features %s | node_times %s | edge_index %s",
                tuple(features.shape), tuple(times.shape), tuple(edge_index.shape))
    return features, times, edge_index


# ---------------------------------------------------------------------------
# Degree features
# ---------------------------------------------------------------------------

def compute_degree_features(edge_index: torch.Tensor, n_nodes: int) -> np.ndarray:
    """Returns float32 array [N, 3]: (in_degree, out_degree, total_degree)."""
    src, dst = edge_index[0], edge_index[1]

    out_deg = torch.zeros(n_nodes, dtype=torch.long)
    out_deg.scatter_add_(0, src, torch.ones_like(src))

    in_deg = torch.zeros(n_nodes, dtype=torch.long)
    in_deg.scatter_add_(0, dst, torch.ones_like(dst))

    total_deg = in_deg + out_deg
    return np.stack([
        in_deg.numpy().astype(np.float32),
        out_deg.numpy().astype(np.float32),
        total_deg.numpy().astype(np.float32),
    ], axis=1)  # [N, 3]


# ---------------------------------------------------------------------------
# Time-step normalisation
# ---------------------------------------------------------------------------

def compute_time_norm(times: torch.Tensor) -> np.ndarray:
    """Normalise time step to [0, 1]: (t - t_min) / (t_max - t_min)."""
    t = times.numpy().astype(np.float32)
    t_min, t_max = t.min(), t.max()
    norm = (t - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t)
    return norm.reshape(-1, 1)  # [N, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    features, times, edge_index = load_processed()
    n_nodes = features.shape[0]

    # ---- compute new features ----
    logger.info("Computing degree features …")
    deg_feats = compute_degree_features(edge_index, n_nodes)  # [N, 3]

    logger.info("Computing normalised time-step feature …")
    time_feats = compute_time_norm(times)  # [N, 1]

    new_feats = np.concatenate([deg_feats, time_feats], axis=1)  # [N, 4]

    # ---- standardise new features ----
    logger.info("Fitting StandardScaler on %d new features (all nodes) …", new_feats.shape[1])
    scaler = StandardScaler()
    new_feats_scaled = scaler.fit_transform(new_feats).astype(np.float32)

    logger.info("  new feature means (pre-scale): %s",
                [f"{v:.3f}" for v in scaler.mean_])
    logger.info("  new feature stds  (pre-scale): %s",
                [f"{v:.3f}" for v in np.sqrt(scaler.var_)])

    # ---- concatenate with original features ----
    new_feats_tensor = torch.from_numpy(new_feats_scaled)       # [N, 4]
    orig_np = features.numpy()                                   # [N, 165]
    combined = np.concatenate([orig_np, new_feats_scaled], axis=1)  # [N, 169]
    combined_tensor = torch.from_numpy(combined)

    logger.info("Final feature matrix: %s", tuple(combined_tensor.shape))

    # ---- feature name list ----
    feature_names = (
        [f"feat_{i}" for i in range(1, N_ORIG + 1)]
        + ["in_degree", "out_degree", "total_degree", "time_step_norm"]
    )
    assert len(feature_names) == combined_tensor.shape[1]

    # ---- save ----
    logger.info("Saving outputs …")

    torch.save(combined_tensor, PROCESSED_DIR / "node_features.pt")
    logger.info("  node_features.pt %s  dtype=%s", tuple(combined_tensor.shape), combined_tensor.dtype)

    scaler_path = PROCESSED_DIR / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("  feature_scaler.pkl")

    names_path = PROCESSED_DIR / "feature_names.json"
    with open(names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("  feature_names.json (%d names)", len(feature_names))

    # ---- report ----
    report = {
        "n_nodes": n_nodes,
        "n_features_original": N_ORIG,
        "n_features_engineered": 4,
        "n_features_total": combined_tensor.shape[1],
        "feature_groups": {
            "original_elliptic": {"indices": [0, N_ORIG - 1], "count": N_ORIG,
                                  "note": "pre-standardised by dataset publishers"},
            "in_degree":       {"index": N_ORIG,     "scaler": "StandardScaler"},
            "out_degree":      {"index": N_ORIG + 1, "scaler": "StandardScaler"},
            "total_degree":    {"index": N_ORIG + 2, "scaler": "StandardScaler"},
            "time_step_norm":  {"index": N_ORIG + 3, "scaler": "StandardScaler",
                                "note": "also min-max normalised before StandardScaler"},
        },
        "degree_stats_raw": {
            "in_degree":    {"mean": float(scaler.mean_[0]), "std": float(np.sqrt(scaler.var_[0]))},
            "out_degree":   {"mean": float(scaler.mean_[1]), "std": float(np.sqrt(scaler.var_[1]))},
            "total_degree": {"mean": float(scaler.mean_[2]), "std": float(np.sqrt(scaler.var_[2]))},
            "time_step_norm": {"mean": float(scaler.mean_[3]), "std": float(np.sqrt(scaler.var_[3]))},
        },
        "scaler_scope": "all nodes (transductive GNN setting — full graph observed at inference)",
        "files": {
            "node_features": "node_features.pt",
            "feature_scaler": "feature_scaler.pkl",
            "feature_names": "feature_names.json",
        },
    }
    report_path = PROCESSED_DIR / "feature_engineering_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("  feature_engineering_report.json")

    # ---- console summary ----
    print("\n=== Phase 1.4 Feature Engineering Summary ===")
    print(f"  Original features  : {N_ORIG}  (feat_1 … feat_165, pre-standardised)")
    print(f"  Engineered features: 4  (in_degree, out_degree, total_degree, time_step_norm)")
    print(f"  Final shape        : {tuple(combined_tensor.shape)}")
    print(f"  in_degree  — mean={scaler.mean_[0]:.3f}  std={np.sqrt(scaler.var_[0]):.3f}")
    print(f"  out_degree — mean={scaler.mean_[1]:.3f}  std={np.sqrt(scaler.var_[1]):.3f}")
    print(f"  total_deg  — mean={scaler.mean_[2]:.3f}  std={np.sqrt(scaler.var_[2]):.3f}")
    print(f"  time_norm  — mean={scaler.mean_[3]:.4f}  std={np.sqrt(scaler.var_[3]):.4f}")
    print(f"  Saved to           : {PROCESSED_DIR}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
