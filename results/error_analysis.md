# Phase 8.5: Error Analysis — GraphSAGE on Test Set (ts 43-49)
## Setup
- Model: GraphSAGE (test AUC=0.777, val threshold=0.54)
- Test labeled nodes: 6687 (169 illicit / 6518 licit)
- Illicit prevalence: 0.025 (concept drift from 0.116 in train)

## Threshold: F1-optimal (0.54)

| |  Pred Licit | Pred Illicit |
|---|---|---|
| **True Licit**   | 6486 TN | 32 FP |
| **True Illicit** | 167 FN | 2 TP |

Precision=0.059  Recall=0.012  F1=0.020

### Error rates by time step

| Time step | Illicit | TP | FN | FP | Recall | FPR |
|-----------|---------|----|----|----|---------|---------|
| ts 43 |  24 |   0 |  24 |   16 | 0.000 | 0.012 |
| ts 44 |  24 |   1 |  23 |   10 | 0.042 | 0.006 |
| ts 45 |   5 |   0 |   5 |    3 | 0.000 | 0.002 |
| ts 46 |   2 |   1 |   1 |    0 | 0.500 | 0.000 |
| ts 47 |  22 |   0 |  22 |    2 | 0.000 | 0.002 |
| ts 48 |  36 |   0 |  36 |    0 | 0.000 | 0.000 |
| ts 49 |  56 |   0 |  56 |    1 | 0.000 | 0.002 |

### Error rates by node degree quartile

| Degree quartile | Range | Illicit | Recall | FPR |
|-----------------|-------|---------|--------|-----|
| Q2               | 1-1 | 115 | 0.000 | 0.003 |
| Q3               | 2-2 |  30 | 0.000 | 0.007 |
| Q4 (high)        | 3-188 |  24 | 0.083 | 0.004 |
## Threshold: Youden-J (0.131)

| |  Pred Licit | Pred Illicit |
|---|---|---|
| **True Licit**   | 6399 TN | 119 FP |
| **True Illicit** | 166 FN | 3 TP |

Precision=0.025  Recall=0.018  F1=0.021

### Error rates by time step

| Time step | Illicit | TP | FN | FP | Recall | FPR |
|-----------|---------|----|----|----|---------|---------|
| ts 43 |  24 |   1 |  23 |   53 | 0.042 | 0.039 |
| ts 44 |  24 |   1 |  23 |   39 | 0.042 | 0.025 |
| ts 45 |   5 |   0 |   5 |   10 | 0.000 | 0.008 |
| ts 46 |   2 |   1 |   1 |    4 | 0.500 | 0.006 |
| ts 47 |  22 |   0 |  22 |    3 | 0.000 | 0.004 |
| ts 48 |  36 |   0 |  36 |    2 | 0.000 | 0.005 |
| ts 49 |  56 |   0 |  56 |    8 | 0.000 | 0.019 |

### Error rates by node degree quartile

| Degree quartile | Range | Illicit | Recall | FPR |
|-----------------|-------|---------|--------|-----|
| Q2               | 1-1 | 115 | 0.009 | 0.020 |
| Q3               | 2-2 |  30 | 0.000 | 0.024 |
| Q4 (high)        | 3-188 |  24 | 0.083 | 0.009 |

## Systematic vs Random Error

**Concept drift (systematic)**: The dominant error source is the illicit prevalence shift from 11.6% (train ts 1-34) to 2.5% (test ts 43-49).  The classifier was calibrated on a more balanced distribution; at test time, nearly all high-scoring nodes are licit, inflating FPR.

**Distribution shift at epoch boundary**: Time steps 43-49 represent the latest Bitcoin transactions in the dataset.  The Bitcoin ecosystem evolved significantly across the dataset's 49 time steps; illicit actors may have changed their behavioural patterns, reducing the model's ability to detect them.

**Unknown-label nodes**: 22,997 test-period nodes (77%) are unlabelled.  High-scoring unknown nodes are operationally relevant alerts and may include genuinely illicit transactions not captured in the labelling sample.

**Random component**: Degree analysis shows no strong correlation between node degree and misclassification rate — errors are not concentrated on hub or leaf nodes, suggesting the residual error is driven by distributional shift rather than a structural model limitation.
