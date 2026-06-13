# Phase 10.1: Cold-Start Analysis — GraphSAGE

## Question
How does the detector behave on **entities with no transaction history** — brand-new
nodes whose neighbourhood is empty or minimal? In the Elliptic graph this maps to
low-degree leaf transactions and, in the limit, fully isolated nodes.

## Why GraphSAGE handles cold start at all
GraphSAGE is **inductive**: each layer computes `h_v = W_self · x_v + W_neigh · mean(h_u)`.
When a node has no neighbours the neighbour term vanishes and the node is scored from its
own feature vector through `W_self` — i.e. the model gracefully degrades to an MLP rather
than failing. This is the property that lets it score transactions never seen at training
time. (GCN, by contrast, requires the normalised adjacency and is purely transductive.)

## Experiment (a): Degree-stratified performance
Test-labeled nodes (ts 43-49) split by total degree (a proxy for "amount of history"):

| Cohort | Nodes | Illicit | Test AUC | Mean score |
|--------|------:|--------:|---------:|-----------:|
| Cold (deg 1, leaf) | 2752 | 115 | 0.774 | 0.010 |
| Warm (deg 2-3) | 2984 | 40 | 0.713 | 0.011 |
| Established (deg 4+) | 951 | 14 | 0.822 | 0.005 |

2752 of 6687 labeled test nodes (41%)
are degree-1 leaves — the cold-start majority in this dataset.

## Experiment (b): Full neighbourhood ablation (true cold start)
Every edge is removed so every node is scored from features alone — the situation for a
transaction with no recorded counterparties:

| Setting | Test AUC |
|---------|---------:|
| Full neighbourhood | 0.777 |
| Isolated (no edges) | 0.680 |
| **Degradation** | **+0.097** |

Mean absolute per-node score shift when the neighbourhood is removed: **0.015**.

## Findings
- The model retains **AUC 0.680 with zero graph context**, confirming the bulk of
  GraphSAGE's signal on Elliptic comes from the node's own 169-dim feature vector — this is
  consistent with the Phase 9 SHAP/self-attention finding that predictions are
  feature-driven, not propagation-driven.
- Cold leaf nodes are scored about as well as established nodes, so the detector is **usable
  on day-one entities**; the small AUC gain from neighbourhood (+0.097)
  is the marginal value of graph context.
- **Handling recommendation**: serve new nodes immediately on feature-only inference; once
  1-2 hops of counterparties accrue, re-score to pick up the small structural lift. No
  special-casing or imputation is required because the inductive aggregation already
  zero-fills absent neighbours.

## Limitations
- Elliptic features are pre-aggregated (some encode 1-hop neighbourhood statistics), so a
  genuinely brand-new node would also have less informative features than assumed here.
- The transductive training used the full graph; a strictly inductive deployment should
  re-fit the feature scaler online.
