# SafeShift — Graph-relational model vs per-interface models (α dose-response)

Mean ± std over 5 seeds; held-out test of 20 graphs per seed; GNN epoch chosen by a separate validation split (never the test set).

**Modeling assumption (a hypothesis, not established fact):** integration risk *cascades* multi-hop along dependencies, with strength α. At **α=0** propagation is off and the label is a pure function of the standard per-interface features — a built-in **negative control** where a topology-aware model should have no advantage.

## ROC-AUC by propagation strength α

| α | Bayes (ceiling) | Heuristic | RandomForest | RiskGNN | RiskGNN − RF |
|---|----------------:|----------:|-------------:|--------:|-------------:|
| 0.0 (control) | 0.756 | 0.748 ± 0.018 | 0.725 ± 0.017 | 0.734 ± 0.027 | +0.009 |
| 0.3 | 0.763 | 0.750 ± 0.015 | 0.724 ± 0.015 | 0.740 ± 0.018 | +0.016 |
| 0.6 | 0.775 | 0.748 ± 0.021 | 0.713 ± 0.024 | 0.751 ± 0.019 | +0.037 |
| 0.9 | 0.793 | 0.731 ± 0.018 | 0.697 ± 0.023 | 0.767 ± 0.013 | +0.069 |

## PR-AUC by propagation strength α

| α | Heuristic | RandomForest | RiskGNN |
|---|----------:|-------------:|--------:|
| 0.0 (control) | 0.717 ± 0.016 | 0.683 ± 0.019 | 0.697 ± 0.028 |
| 0.3 | 0.777 ± 0.012 | 0.749 ± 0.015 | 0.766 ± 0.017 |
| 0.6 | 0.835 ± 0.022 | 0.807 ± 0.019 | 0.840 ± 0.024 |
| 0.9 | 0.874 ± 0.016 | 0.856 ± 0.007 | 0.894 ± 0.017 |

## Reading the dose-response (reported as-is)

- At the **α=0 control**, RiskGNN − RandomForest = **+0.009** (expected ≈ 0).
- At the strongest propagation (α=0.9), the gap is **+0.069**.

The graph model's advantage over the per-interface RandomForest is ~0 in the control and **grows with propagation strength** — the signature of a model recovering multi-hop neighbour information that per-interface features cannot carry, rather than being a generically stronger learner. The heuristic stays flat across α (it is, by construction, blind to propagation).

## Honest scope
These are **synthetic, construct-level** results: the propagation structure is a modeling assumption baked into the generator, and the Bayes-optimal ceiling bounds what any model can reach. They demonstrate *when* a topology-aware model is warranted (when risk genuinely propagates) — NOT that the GNN is superior on real integration outcomes. Whether real integration risk propagates this way, and how strongly, must be settled by calibration on an organisation's own historical integration defects (see the README roadmap). Reproduce: `python evaluation/graph_eval.py`.

Figure: `figures/graph_eval.png`.