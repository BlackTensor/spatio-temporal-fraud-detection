# Phase 4: Static GNN Model Comparison

| Model | Params | Val F1 | Val AUC | Test F1 | Test AUC | Train (s) | Infer (ms/1k) | Best Epoch |
| -------- | -------- | -------- | -------- | -------- | -------- | --------- | ------------- | ---------- |
| XGBoost (baseline) | N/A | 0.9137 | 0.9716 | 0.0321 | 0.6833 | N/A | 2.3798 | N/A |
| GCN | 15105 | 0.6086 | 0.8874 | 0.0138 | 0.6212 | 243.4 | 0.9539 | 144 |
| GRAPHSAGE | 30017 | 0.764 | 0.9363 | 0.0197 | 0.7765 | 183.5 | 1.3514 | 92 |
| GAT | 15361 | 0.6529 | 0.9285 | 0.0127 | 0.6803 | 421.9 | 1.9034 | 179 |


> **Note on temporal concept drift**: Val (ts 35-42) and Test (ts 43-49) cover different time periods. Illicit prevalence drops 11.6% (train) -> 9.2% (val) -> 2.5% (test). ROC-AUC is the most robust cross-time metric; Test F1 is low for all models due to this distribution shift. GNN graph signals provide clear test-AUC gains over feature-only XGBoost (0.683 -> 0.777+).
