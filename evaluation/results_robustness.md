# SafeShift — Robustness across independent, aligned ground truths

Each column is a *different* latent risk function (independent of SafeShift's scorer). All are plausible, aligned risk models. The point: informed models beat the random baseline across every generator, so the headline result is not an artifact of one chosen synthetic target.

| Ground truth | pos. rate | Random | Heuristic | LogReg | RandomForest |
|--------------|----------:|-------:|----------:|-------:|-------------:|
| interaction (reference) | 0.433 | 0.510 | 0.818 | 0.824 | 0.812 |
| linear-additive | 0.537 | 0.507 | 0.766 | 0.774 | 0.760 |
| threshold / rule-like | 0.365 | 0.524 | 0.747 | 0.772 | 0.835 |
| structure-emphasis | 0.596 | 0.511 | 0.758 | 0.800 | 0.778 |