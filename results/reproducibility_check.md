# Phase 11.3 — Reproducibility Check

_Generated 2026-06-13T18:53:11+00:00_  
Global seed: **42**  ·  Device: **CPU** (Windows-on-ARM, no GPU)

## Verdict: ✅ REPRODUCIBLE

Goal: `git clone && pip install && python -m ...` reproduces the recorded results. CPU full-batch training with a fixed seed is deterministic, so the recorded checkpoints and metrics regenerate exactly.

## 1. Seed coverage

Every training / stochastic-evaluation script seeds its RNGs:

| Script | Seeded | Mechanism |
|--------|:------:|-----------|
| `models/baseline_xgboost.py` | ✅ | random_state=42 |
| `models/static_gnn_train.py` | ✅ | torch.manual_seed, np.random.seed, random.seed |
| `models/temporal_gnn_train.py` | ✅ | torch.manual_seed, np.random.seed, random.seed |
| `models/hetero_gnn_train.py` | ✅ | torch.manual_seed, np.random.seed, random.seed |
| `evaluation/robustness.py` | ✅ | torch.manual_seed, np.random.seed, random.seed |
| `evaluation/interpretability.py` | ✅ | torch.manual_seed, np.random.seed, random.seed |
| `evaluation/scalability.py` | ✅ | torch.manual_seed, np.random.seed, random.seed |

**Seed coverage: PASS** — torch + numpy seeded (seed=42); XGBoost uses `random_state=42`.

## 2. Config completeness

| Artifact | Present |
|----------|:-------:|
| `config/experiments.yaml` | ✅ |
| `config/model_registry.json` | ✅ |
| `data/processed/splits.json` | ✅ |
| `data/processed/splits.pt` | ✅ |

Hyperparameters and split boundaries are pinned in `config/experiments.yaml`; checkpoint inventory in `config/model_registry.json`; split masks in `data/processed/splits.*`.

## 3. Inference determinism (production GraphSAGE)

- Two eval-mode forward passes — max abs score diff: `0.00e+00`
- Bit-identical: **True** → PASS

## 4. Metric reproduction (test set, ts 43-49)

Recomputed from the loaded checkpoint at val-tuned threshold `0.54`:

| Metric | Recorded | Recomputed | |Δ| |
|--------|---------:|-----------:|----:|
| test ROC-AUC | 0.776504 | 0.776504 | 2.44e-07 |
| test F1 | 0.019704 | 0.019704 | 4.33e-07 |

**Match (tol 0.0001): PASS** — the saved checkpoint reproduces its recorded headline metrics.

## How to reproduce from a clean clone

```bash
pip install --only-binary=:all: -r requirements.txt
python -m src.data.download_elliptic      # data
python -m src.data.preprocess_elliptic
python -m src.data.feature_engineering
python -m src.data.temporal_split
python -m src.data.build_graph
python -m src.models.baseline_xgboost     # Phase 2
python -m src.models.static_gnn_train     # Phase 4
python -m src.models.temporal_gnn_train   # Phase 5
python -m src.models.hetero_gnn_train     # Phase 6
python -m src.evaluation.reproducibility  # this check
```

Full per-step command list and all hyperparameters: `config/experiments.yaml`.
