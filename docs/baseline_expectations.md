# Phase 0.2 — Baseline Expectations & Success Targets

These are the **a-priori targets** the project is benchmarked against. They set
the bar before any modeling, so each phase can be judged on whether it moved the
needle. Numbers are guidance from the GNN-fraud literature on comparable
heterogeneous/temporal transaction graphs (e.g. Elliptic, IEEE-CIS); the actual
dataset (Phase 1) may shift absolute values, but the **relative ordering** —
each added inductive bias (graph → temporal → heterogeneous) should improve
detection — is the real success criterion.

## Target detection metrics

Primary metric: **F1 on the minority (fraud/anomaly) class** — accuracy is
meaningless under heavy class imbalance. Reported alongside precision, recall,
and ROC-AUC / PR-AUC.

| Model | What it adds | Target F1 | Phase |
|-------|--------------|-----------|-------|
| XGBoost (features only) | tabular baseline, no graph | **~75–80%** | 2 |
| Static GNN (GCN / GraphSAGE / GAT) | neighborhood structure | **~83–85%** | 4 |
| Temporal GNN (snapshot / TGN) | + time dynamics | **~88–90%** | 5 |
| **Heterogeneous Temporal GNN (HTGN)** | + node/edge typing | **91%+** | 6 |

**Expected lift:** graph structure over tabular ≈ +5–8 F1; temporal over static
≈ +4–6 F1; heterogeneous over homogeneous ≈ +2–4 F1. Ablations in Phases 4–6
must isolate and report each increment.

## Latency target

- **Inference < 100 ms per node** at serving time.
- Benchmarked on **free CPU (local)** and **free Colab GPU** at batch sizes
  1 / 32 / 256 (Phase 8.3). The local dev box is CPU-only Windows-on-ARM, so the
  CPU number is the conservative floor; GPU numbers come from Colab.

## Repository / engineering goals

- **Clean README** — quick start, results table, demo link, model card.
- **Reproducible** — `git clone && pip install && python train.py` reproduces
  reported numbers; everything seeded; all hyperparameters in YAML configs.
- **Minimal, readable code** — small modules under `src/`, no dead code, typed
  where it helps, tested with pytest on small fixtures.
- **$0 cost** — free compute (local + Colab/Kaggle), free datasets, free hosting
  (HuggingFace Hub for large models, Streamlit Community Cloud / HF Spaces for
  the demo, GitHub Actions CI, GitHub Pages for docs). No paid services, ever.

## How these targets are used downstream

- Each model phase records metrics to `results/` and compares against the row
  above; a phase that fails to hit its target triggers error analysis
  (Phase 8.5) rather than silently moving on.
- The final `RESULTS.md` (Phase 14.4) reports achieved vs. target side by side.
