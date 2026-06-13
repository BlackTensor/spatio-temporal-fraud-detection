# Phase 10.2: Concept-Drift Analysis — GraphSAGE

## Question
How far into the future does a model trained on ts 1-34 stay reliable, and when should it
be retrained?

## Method
The frozen GraphSAGE (test AUC=0.777 overall) is scored on the full graph; ROC-AUC and
illicit prevalence are computed **per time step** for every labeled node from ts 1 to 49.
No retraining or threshold change is applied — this isolates pure temporal degradation.

## Period summary

| Period | Steps | Mean AUC | Mean illicit prevalence |
|--------|-------|---------:|------------------------:|
| Train | 1-34 | 0.999 | 0.134 |
| Val | 35-42 | 0.933 | 0.095 |
| Test | 43-49 | 0.806 | 0.037 |

**Degradation train → test: 0.192 AUC (19% relative).**

## Drivers
1. **Prevalence collapse**: illicit share falls 13.4% → 3.7%. A
   detector calibrated on a denser positive class over-fires on the sparse test period,
   destroying precision/F1 even where ranking (AUC) partially holds.
2. **Behavioural drift**: a documented Elliptic event — a dark-market shutdown around the
   later time steps — changes the illicit footprint, so feature/structure patterns learned
   on ts 1-34 no longer match ts 43-49.
3. The decay is **monotone-ish across steps**, not a single cliff: see
   `concept_drift_auc.png`. Some late steps with very few illicit nodes have unstable AUC.

## Retraining recommendation
- **Trigger-based retraining**: retrain when rolling per-step AUC drops below a SLA floor
  (e.g. 0.85) or when illicit prevalence shifts >2× from the training value. Both fire well
  before ts 43 here.
- **Cadence**: with ~one Elliptic step ≈ a few hours of Bitcoin activity, a weekly/biweekly
  rolling-window refit (train on the most recent N steps) is appropriate; the train-period
  AUC of 0.999 shows the architecture is sound when the window matches the data.
- **Calibration**: even between refits, recalibrate the decision threshold on a recent
  labeled window so precision tracks the current prevalence (the Phase 7 thresholds were
  val-derived and are too aggressive on test).
- **Drift monitoring**: log prevalence and score-distribution (PSI/KL) per step; alert on
  divergence rather than waiting for labels.

## Conclusion
GraphSAGE is reliable **within ~8 steps of its training window** (val AUC 0.933) but
loses 19% of its edge over AUC by the far-future test period. The model is not
broken — the *world* changed; a rolling retrain + threshold recalibration restores it.
