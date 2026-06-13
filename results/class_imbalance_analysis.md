# Phase 10.4: Class-Imbalance Analysis — GraphSAGE

## Setup
Training labels are imbalanced **26432 licit : 3462 illicit ≈ 7.6:1**.
All prior phases use `BCEWithLogitsLoss(pos_weight=7.635)` (inverse-frequency) to
up-weight the minority class. Here we sweep the weight to measure its effect on
**minority-class TPR (recall)** and the precision tradeoff. Each row is a fresh GraphSAGE,
early-stopped on val AUC, threshold F1-optimised on val (no leakage). The canonical
`graphsage_model.pt` is untouched.

## Results

| Weighting | pos_weight | Val TPR | Val Prec | Val F1 | Test TPR | Test Prec | Test AUC |
|-----------|-----------:|--------:|---------:|-------:|---------:|----------:|---------:|
| None (1.0) | 1.00 | 0.688 | 0.856 | 0.763 | 0.006 | 0.056 | 0.743 |
| Sqrt inverse-freq | 2.76 | 0.716 | 0.873 | 0.786 | 0.012 | 0.054 | 0.781 |
| Inverse-freq (default) | 7.63 | 0.706 | 0.786 | 0.744 | 0.012 | 0.037 | 0.750 |
| 2× inverse-freq | 15.27 | 0.661 | 0.792 | 0.720 | 0.006 | 0.024 | 0.761 |

> TPR = true-positive rate on the illicit (minority) class = recall.

## Findings
- **A *mild* weighting wins, not the heaviest.** `pos_weight=2.76`
  (Sqrt inverse-freq) gives the best validation operating point — Val F1=0.786,
  TPR=0.716, precision=0.873 — and also the
  best Test AUC (0.781). The relationship is **non-monotonic**:
  recall does *not* keep rising with weight.
- **Over-weighting hurts both metrics.** The heaviest setting
  (`pos_weight=15.27`) drops Val TPR to 0.661 and
  precision to 0.792: pushing too hard on the minority class makes
  training noisier (early-stopping on AUC then halts sooner), so it neither detects more nor
  precision-trades cleanly. The textbook "more weight → more recall → less precision" tradeoff
  only holds locally.
- **Unweighted (1.0)** is already competitive on this dataset
  (Val F1=0.763) because early-stopping on val AUC + a val-tuned
  threshold partly compensates for the loss imbalance — the threshold absorbs much of what
  pos_weight is meant to fix.
- **Ranking (AUC) is fairly stable** across weightings (0.743–0.781 on test):
  weighting mainly shifts the *operating point*, not the underlying separability.
- Concept drift still caps **test** recall (≤0.012) regardless of weighting (cf. Phase 10.2);
  loss weighting addresses imbalance, not distribution shift — the two need different remedies
  (weighting/threshold vs retraining/recalibration).

## Other imbalance levers (noted, not all swept)
- **Stratified / minority oversampling** in mini-batch training (e.g. `GraphSAINT` or
  neighbour-sampling with class-balanced node sampling) — equivalent effect to pos_weight
  for full-batch, more relevant when the graph no longer fits in memory.
- **Focal loss** to focus gradient on hard minority examples.
- **Threshold calibration** on a recent labeled window — the cheapest lever, and necessary
  anyway under drift.

## Recommendation
On this dataset a **mild weighting (`pos_weight≈2.8`, sqrt
inverse-frequency)** edges out the full inverse-frequency default
(`7.63`) on every validation metric and on test AUC — worth adopting
if retraining the production model. More importantly, treat the **decision threshold as the
primary, separately-tuned and periodically-recalibrated knob** for hitting a recall SLA: it
moves the operating point more reliably than pos_weight and is the only lever that also
tracks concept drift. Reserve heavy weighting (≥2× inverse-freq) — it degraded both recall
and precision here.
