# SafeShift Evaluation Results

- Dataset: 8000 synthetic interfaces (independent non-linear ground truth); test set 2400; positive rate 0.43.

## Held-out performance (30% test set)

| Model | ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 |
|-------|--------:|-------:|------:|----------:|-------:|---:|
| Random | 0.515 | 0.437 | 0.323 | 0.44 | 0.50 | 0.47 |
| Heuristic (linear) | 0.806 | 0.756 | 0.198 | 0.60 | 0.84 | 0.70 |
| Logistic Regression | 0.812 | 0.759 | 0.174 | 0.73 | 0.68 | 0.70 |
| Random Forest | 0.803 | 0.749 | 0.178 | 0.70 | 0.66 | 0.68 |

5-fold CV (Random Forest) ROC-AUC: **0.810 ± 0.006** (folds: [0.8179, 0.8081, 0.8048, 0.8039, 0.8163]).

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