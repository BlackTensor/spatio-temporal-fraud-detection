# Phase 5: Temporal GNN Ablation — Elliptic Bitcoin Dataset

| Model | Phase | Params | Val F1 | Val AUC | Test F1 | Test AUC | Train (s) | Best Ep |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| XGBoost | Phase 2 | N/A | 0.9137 | 0.9716 | 0.0321 | 0.6833 | N/A | N/A |
| GCN | Phase 4 | 15105 | 0.6086 | 0.8874 | 0.0138 | 0.6212 | 243.4 | 144 |
| GraphSAGE | Phase 4 | 30017 | 0.764 | 0.9363 | 0.0197 | 0.7765 | 183.5 | 92 |
| GAT | Phase 4 | 15361 | 0.6529 | 0.9285 | 0.0127 | 0.6803 | 421.9 | 179 |
| SnapshotGNN (5.1) | Phase 5 | 55041 | 0.7537 | 0.9484 | 0.0108 | 0.7275 | 116.5 | 37 |
| EvolveGCN (5.2) | Phase 5 | 69121 | 0.4432 | 0.7947 | 0.0245 | 0.6016 | 128.3 | 47 |


## Key Findings

| Comparison | Test AUC delta |
|------------|----------------|
| SnapshotGNN vs GraphSAGE (best static) | -0.0490 |
| EvolveGCN vs GraphSAGE | -0.1749 |
| Best temporal vs XGBoost | +0.0442 |

> **Note on concept drift**: Illicit prevalence drops 11.6% (train) → 9.2% (val) → 2.5% (test). Test F1 is low for all models due to this distribution shift. ROC-AUC is the reliable cross-time metric.

