# SafeShift Evaluation Results

- Dataset: 8000 synthetic interfaces (independent non-linear ground truth); test set 2400; positive rate 0.43.

## Held-out performance (30% test set)

| Model | ROC-AUC | 95% CI (bootstrap) | PR-AUC | Brier | F1 |
|-------|--------:|:------------------:|-------:|------:|---:|
| Random | 0.515 | [0.491, 0.539] | 0.437 | 0.323 | 0.47 |
| Safety-only (1 rule) | 0.697 | [0.679, 0.715] | 0.571 | 0.307 | 0.67 |
| Safety x immaturity (2 rule) | 0.715 | [0.695, 0.734] | 0.604 | 0.268 | 0.49 |
| Heuristic (linear) | 0.806 | [0.789, 0.823] | 0.756 | 0.198 | 0.70 |
| Logistic Regression | 0.812 | [0.794, 0.828] | 0.759 | 0.174 | 0.70 |
| Random Forest | 0.803 | [0.785, 0.820] | 0.749 | 0.178 | 0.68 |

5-fold CV (Random Forest) ROC-AUC: **0.810 ± 0.006** (folds: [0.8179, 0.8081, 0.8048, 0.8039, 0.8163]).

**Circularity diagnostic.** Heuristic ROC-AUC 0.806; with all feature signs flipped it inverts to 0.194, and the best single feature (safety) already reaches 0.697. This synthetic study measures recovery of a target built from the same features and signs as the scorer — construct recovery, not predictive validity on real outcomes.

## Ablation — ROC-AUC drop when a feature group is removed (Random Forest)

| Feature group removed | ROC-AUC without | AUC drop |
|-----------------------|----------------:|---------:|
| safety/ASIL | 0.645 | 0.158 |
| structural | 0.758 | 0.045 |
| integration | 0.784 | 0.019 |
| maturity | 0.798 | 0.005 |

## Robustness — RF ROC-AUC vs label-noise temperature

| Noise (1.0=default, higher=noisier) | RF ROC-AUC |
|---:|---:|
| 0.5 | 0.932 |
| 1.0 | 0.819 |
| 1.5 | 0.733 |
| 2.0 | 0.673 |
| 3.0 | 0.642 |

## Random-forest feature importance (descending)

- safety_related: 0.251
- min_maturity: 0.161
- signals: 0.125
- tgt_in_cycle: 0.098
- tgt_fan_in: 0.083
- src_fan_out: 0.082
- max_asil_rank: 0.070
- protocol_mismatch_risk: 0.061
- timing_critical: 0.036
- supplier_boundary: 0.033

Figures: `figures/roc.png`, `figures/feature_importance.png`.