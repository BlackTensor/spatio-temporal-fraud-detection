# Phase 6: Heterogeneous GNN Ablation — Elliptic Bitcoin Dataset

| Model | Phase | Params | Val F1 | Val AUC | Test F1 | Test AUC | Train (s) | Best Ep |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| XGBoost | Phase 2 | N/A | 0.9137 | 0.9716 | 0.0321 | 0.6833 | N/A | N/A |
| GCN | Phase 4 | 15105 | 0.6086 | 0.8874 | 0.0138 | 0.6212 | 243.4 | 144 |
| GraphSAGE | Phase 4 | 30017 | 0.764 | 0.9363 | 0.0197 | 0.7765 | 183.5 | 92 |
| GAT | Phase 4 | 15361 | 0.6529 | 0.9285 | 0.0127 | 0.6803 | 421.9 | 179 |
| SnapshotGNN | Phase 5 | 55041 | 0.7537 | 0.9484 | 0.0108 | 0.7275 | 116.5 | 37 |
| EvolveGCN | Phase 5 | 69121 | 0.4432 | 0.7947 | 0.0245 | 0.6016 | 128.3 | 47 |
| HeteroSAGE | Phase 6 | 59969 | 0.7436 | 0.9498 | 0.019 | 0.7508 | 260.3 | 76 |
| HeteroGAT (HGAT) | Phase 6 | 30657 | 0.6689 | 0.9271 | 0.0124 | 0.7161 | 417.0 | 91 |
| HTGN | Phase 6 | 84993 | 0.801 | 0.9597 | 0.0302 | 0.7112 | 159.6 | 22 |


## Key Findings

| Comparison | Test AUC delta |
|------------|----------------|
| Best Phase 6 vs best Phase 4 (GraphSAGE) | -0.0257 |
| Best Phase 6 vs best Phase 5 (SnapshotGNN) | +0.0233 |
| Best Phase 6 vs XGBoost baseline | +0.0675 |
| HTGN vs GraphSAGE (temporal+hetero vs static-homo) | -0.0653 |

> **Heterogeneity construction**: Elliptic has one node type. Direction-typed edges (sends / receives) give each model asymmetric inductive bias for upstream vs downstream fraud propagation.
> **Concept drift**: Illicit prevalence drops 11.6% (train) → 9.2% (val) → 2.5% (test). ROC-AUC is the reliable cross-time metric; Test F1 is low for all models due to this distribution shift.
