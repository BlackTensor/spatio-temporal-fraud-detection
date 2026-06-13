# Dataset — Elliptic Bitcoin (Phase 1.1)

## What it is

The **Elliptic Data Set** is a real, anonymized **Bitcoin transaction graph**
labelled for anti-money-laundering research. Each node is a transaction; each
directed edge is a payment flow from one transaction to another. A subset of
nodes is labelled **illicit** (scams, malware, ransomware, Ponzi schemes, etc.)
or **licit** (exchanges, miners, wallet providers); the rest are **unknown**.

| Property | Value |
|----------|-------|
| Nodes (transactions) | **203,769** |
| Edges (payment flows) | **234,355** |
| Features per node | **166** (1 time step + 165 numeric: 94 local tx + 72 aggregated) |
| Time steps | **49** (≈ 2 weeks apart, evenly spaced) |
| Labelled illicit (class 1) | 4,545 (~2%) |
| Labelled licit (class 2) | 42,019 (~21%) |
| Unknown (unlabelled) | 157,205 (~77%) |

This is a strongly **class-imbalanced**, **temporal** graph — exactly the
setting this project targets. It is *homogeneous* (a single node type), so the
heterogeneous modeling phases (3, 6) will derive typed structure / synthetic
heterogeneity or treat heterogeneity as an ablation; see those phases.

## Files (`data/raw/elliptic/`)

| File | Contents |
|------|----------|
| `elliptic_txs_features.csv` | `txId`, `time_step`, then 165 unnamed feature columns |
| `elliptic_txs_classes.csv` | `txId`, `class` (`1`=illicit, `2`=licit, `unknown`) |
| `elliptic_txs_edgelist.csv` | `txId1`, `txId2` directed edges |

> ⚠️ The features CSV has **no header row** for the 165 feature columns in the
> original release; columns are positional. Handled in Phase 1.3 preprocessing.

## Source & access

Primary (used by `src/data/download_elliptic.py`, **no credentials needed**):

```bash
python -m src.data.download_elliptic
```

Downloads the three CSVs (zipped) from the PyTorch Geometric mirror
`https://data.pyg.org/datasets/elliptic` into `data/raw/elliptic/`.

Fallback — Kaggle API (free account + `kaggle.json`):

```bash
kaggle datasets download -d ellipticco/elliptic-data-set \
    -p data/raw/elliptic --unzip
```

Raw data is **gitignored** (`data/raw/*`) and re-downloadable, so it is never
committed.

## License & terms

Released by **Elliptic** in collaboration with the **MIT-IBM Watson AI Lab** for
**research and educational use**. Publicly distributed on Kaggle
(`ellipticco/elliptic-data-set`). Use here is non-commercial academic research,
consistent with those terms. Verify the current Kaggle license field before any
redistribution; this repo redistributes **no data**, only a download script.

## Citation

> M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson,
> C. E. Leiserson. *Anti-Money Laundering in Bitcoin: Experimenting with Graph
> Convolutional Networks for Financial Forensics.* KDD '19 Workshop on Anomaly
> Detection in Finance, 2019. arXiv:1908.02591.

## Notes for later phases

- **Temporal split (Phase 1.5):** split by `time_step` (train = early, val =
  mid, test = late) — no leakage. The 49 steps make this natural.
- **Labels (Phase 7):** node classification on the labelled subset is the
  primary strategy; unknown nodes can be used for transductive message passing.
- **Imbalance (Phase 10.4):** ~2% illicit → weighted loss / focal loss; report
  minority-class recall, not accuracy.
