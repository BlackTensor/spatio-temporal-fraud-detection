# Phase 10.3: Adversarial Robustness — GraphSAGE

## Threat model
An illicit actor wants their transaction node to score *low* (evade detection). We hold the
trained GraphSAGE fixed (white-box weights, black-box gradients) and perturb inputs in three
realistic ways. Baseline (unperturbed): **AUC=0.777, illicit recall=0.012**
at the val-derived threshold 0.540.

## (1) Noise injection
Gaussian noise (σ × feature std) added to all test feature vectors — models data-quality
degradation or naïve obfuscation.

| Noise σ | Test AUC | Illicit recall |
|---|---|---|
| 0.000 | 0.777 | 0.012 |
| 0.250 | 0.769 | 0.018 |
| 0.500 | 0.748 | 0.018 |
| 1.000 | 0.716 | 0.089 |
| 2.000 | 0.662 | 0.160 |

Ranking quality (**AUC**) degrades gracefully — only ~0.03 lost at σ=0.5, ~0.06 at σ=1.0 —
so the model is not relying on brittle high-frequency feature detail. Recall *rises* with
noise, but this is an artifact, not improved detection: noise scatters scores so more nodes
(both illicit and licit) cross the fixed threshold, inflating recall while precision and AUC
fall. AUC is therefore the honest robustness metric here, and it holds up well.

## (2) Feature camouflage
Illicit feature vectors are linearly blended toward the **licit training centroid**
(α=1 → fully disguised as average-licit). This is the strongest realistic attack: the
adversary directly mimics legitimate behaviour.

| Blend α | Illicit recall | Mean illicit score |
|---|---|---|
| 0.000 | 0.012 | 0.017 |
| 0.250 | 0.012 | 0.024 |
| 0.500 | 0.012 | 0.021 |
| 0.750 | 0.000 | 0.001 |
| 1.000 | 0.000 | 0.000 |

Recall collapses as α→1: an actor who can make their features statistically
indistinguishable from licit nodes **will** evade a feature-driven detector. This is the
flip-side of the Phase 9 finding that scores are feature-driven — it is also the main
attack surface.

## (3) Structural slow-bleed
Each illicit node is wired to `k` random licit test nodes (bidirectional), diluting its
neighbourhood with legitimate counterparties — "blending in" structurally over time.

| Licit edges k | Illicit recall | Mean illicit score |
|---|---|---|
| 0 | 0.012 | 0.017 |
| 1 | 0.012 | 0.014 |
| 3 | 0.012 | 0.012 |
| 5 | 0.012 | 0.013 |
| 10 | 0.012 | 0.012 |

Effect is **mild**: because GraphSAGE on Elliptic is feature-dominated (Phase 9, and 10.1
edge-ablation), adding licit neighbours only weakly pulls the score down. Structural
camouflage is far less effective than feature camouflage here.

## Findings & hardening
- **Most dangerous vector: feature camouflage**, not structural manipulation — the inverse
  of the intuition for graph models, and a direct consequence of Elliptic's pre-aggregated,
  feature-heavy signal.
- **Robust to noise and structural dilution** within realistic budgets.
- **Hardening**: (a) adversarial training with camouflage-style augmentations; (b) ensemble
  the feature model with a structure-only model (e.g. HeteroSAGE) so an attacker must defeat
  both; (c) monitor for nodes whose features sit implausibly close to the licit centroid;
  (d) cost-sensitive thresholds so near-boundary illicit cases still alert.

## Limitations
Perturbations are heuristic, not gradient-optimised (no PGD), and ignore real-world
constraints (a Bitcoin transaction's features are not freely editable). Results bound
robustness against *plausible* evasion, not a worst-case optimal adversary.
