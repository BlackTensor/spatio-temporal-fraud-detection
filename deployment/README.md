# Deployment Guide — Free Hosting Only ($0)

## Option 1: Streamlit Community Cloud (Recommended — Free)

1. Push this repo to GitHub (public repo is free)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo
4. Set the **Main file path** to `dashboard/app.py`
5. Click **Deploy!**

That's it — live URL within ~2 minutes, zero cost.

**Requirements**: `dashboard/requirements.txt` is auto-detected by Streamlit Cloud.

---

## Option 2: HuggingFace Spaces — Docker (Free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** SDK
3. Push a Space repo containing:
   - `Dockerfile` (see below)
   - All files from this repo
4. HF Spaces provides free CPU-tier hosting

```dockerfile
# Dockerfile for HuggingFace Spaces
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r dashboard/requirements.txt

EXPOSE 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

## Option 3: Local Docker (Free — runs on your machine)

```bash
# Build
docker build -t fraud-gnn-dashboard .

# Run
docker run -p 8501:8501 fraud-gnn-dashboard

# Open
open http://localhost:8501
```

---

## Local Development (no Docker)

```bash
# Install
pip install -r dashboard/requirements.txt

# Run from project root
streamlit run dashboard/app.py
```

App will open at **http://localhost:8501**

---

## Data Requirements

The dashboard reads pre-computed results from `results/` and `config/`.
No model weights are loaded at runtime — all inference was done offline.

Required files (already in the repo):
- `results/all_models_evaluation.json`
- `results/concept_drift_analysis.json`
- `results/top_100_anomalies.csv`
- `results/latency_benchmark.csv`
- `results/temporal_evolution.json`
- `results/shap_values.json`
- `results/attention_weights_summary.json`
- `results/scalability_analysis.json`
- `results/eda_stats.json`
- `config/model_registry.json`

---

## Cost: $0

| Service | Cost |
|---------|------|
| Streamlit Community Cloud | Free |
| HuggingFace Spaces (CPU) | Free |
| GitHub (public repo) | Free |
| Local Docker | Free |

No AWS, GCP, or Azure required.
