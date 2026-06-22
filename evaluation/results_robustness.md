# SafeShift — Robustness across alternative ground truths

The first four rows are *aligned* latent risk functions: each is a different functional form (linear, interaction, threshold, structure-emphasis) over the SAME engineering features with the SAME signs as SafeShift's scorer. They test robustness to functional form and coefficients -- not to the choice of feature directions -- so informed models are expected to do well, and these rows do NOT establish independence from the scorer's assumptions. The final row, 'off-feature latent driver', is the genuinely harder test: about half the systematic risk comes from a hidden factor the models never observe, so performance drops toward chance, bounding what the synthetic study can claim.

| Ground truth | pos. rate | Random | Heuristic | LogReg | RandomForest |
|--------------|----------:|-------:|----------:|-------:|-------------:|
| interaction (reference) | 0.433 | 0.510 | 0.818 | 0.824 | 0.812 |
| linear-additive | 0.537 | 0.507 | 0.766 | 0.774 | 0.760 |
| threshold / rule-like | 0.365 | 0.524 | 0.747 | 0.772 | 0.835 |
| structure-emphasis | 0.596 | 0.511 | 0.758 | 0.800 | 0.778 |
| off-feature latent driver | 0.262 | 0.506 | 0.631 | 0.667 | 0.656 |