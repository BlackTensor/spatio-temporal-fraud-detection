# Spatio-Temporal Heterogeneous GNN Anomaly Detection

Production-grade Graph Neural Network system for detecting anomalies in dynamic,
heterogeneous graphs — fraud, account takeover, and bot networks. Built **end to
end on $0 of free tooling** (local CPU + free Colab/Kaggle GPU, free hosting).

> **Status:** Phase 0 — project setup. See [`the project roadmap`](the project roadmap) for the full
> 15-phase roadmap and the hard `$0`-budget constraint.

---

## Why this project

Real fraud lives in *relationships over time*: a device shared across many
accounts, an IP that lights up just before a burst of transfers. Tabular models
miss this structure. This repo builds up from an XGBoost baseline to a
**Heterogeneous Temporal GNN (HTGN)**, measuring the lift at every step.

| Model | Target F1 |
|-------|-----------|
| XGBoost (features only) | ~75–80% |
| Static GNN (GCN/SAGE/GAT) | ~83–85% |
| Temporal GNN (snapshot / TGN) | ~88–90% |
| **Heterogeneous Temporal GNN** | **91%+** |

Latency target: **< 100 ms / node** inference.

---

## Project structure

```
spatio-temporal-fraud-detection/
├── data/               # raw + processed graphs (gitignored; re-downloadable)
├── notebooks/          # EDA + Colab/Kaggle GPU runners
├── src/
│   ├── data/           # loading, preprocessing, graph construction
│   ├── models/         # GNN architectures
│   ├── evaluation/     # metrics, visualization
│   └── utils/          # helpers (seeding, config, logging)
├── config/             # YAML experiment configs
├── results/            # checkpoints, metrics, plots
├── api/                # FastAPI service (local / free HF Spaces)
├── dashboard/          # Streamlit UI (free Community Cloud)
├── deployment/         # Docker + free-deploy notes
├── docs/               # architecture diagrams
├── tests/              # pytest
├── the project roadmap           # phased build plan + $0 budget rules
└── requirements.txt    # pinned, free, open-source deps
```

---

## Quick start (local CPU)

```bash
git clone <repo-url>
cd spatio-temporal-fraud-detection

python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

Local CPU is enough for EDA, the XGBoost baseline, and the dashboard. **GNN
training uses a free GPU** — see below.

### Free GPU (no local NVIDIA card required)

This was developed on a **Windows-on-ARM (Snapdragon) laptop with no NVIDIA
GPU**, so all heavy training runs on free cloud GPU:

- **Google Colab** — free GPU, ~30 hrs/week. Open
  [`notebooks/colab_setup.ipynb`](notebooks/colab_setup.ipynb), which installs
  deps and clones the repo.
- **Kaggle Notebooks** — free GPU quota, same setup notebook works.

No paid cloud (AWS/GCP/Azure) is used anywhere in this project.

---

## Datasets (free only)

Candidates (Phase 1): **Elliptic Bitcoin** (real labeled transaction graph,
already graph-shaped) or **IEEE-CIS Fraud** (heterogeneous e-commerce).
Downloaded via the free Kaggle API into `data/raw/` (gitignored). License and
access steps documented in Phase 1.

---

## $0-budget guarantee

Compute (local + Colab/Kaggle), datasets (Kaggle/open repos), model hosting
(HuggingFace Hub), demo (Streamlit Community Cloud + HF Spaces), CI (GitHub
Actions free tier), docs (GitHub Pages) — **every dependency and service in this
project is free.**

---

## License

Code: MIT (to be added). Datasets retain their original licenses.
