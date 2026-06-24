# SafeShift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20780068.svg)](https://doi.org/10.5281/zenodo.20780068)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

**Open, vendor-neutral shift-left integration-risk prediction for automotive software architectures.**

Modern vehicles are software-defined: dozens to hundreds of electronic control units and software
modules, developed in separate streams and integrated late — often only when physical prototypes
exist. Integration and architectural defects found at that stage are expensive and a recurring
cause of recalls. **SafeShift** analyzes a description of a vehicle's software/E-E architecture and
predicts *where integration trouble is most likely*, so teams can review and fix it during virtual
design instead of after build — a practical "shift-left" of validation effort.

SafeShift is released openly so that the broader automotive base — including suppliers who cannot
fund bespoke tooling — can adopt a common, auditable approach to early validation.

## What it does
- Reads an open **JSON/YAML schema** describing components (supplier, ASIL, maturity) and
  interfaces (protocol, signals, safety/timing relevance).
- Builds a **dependency graph** and computes structural risk metrics (fan-in/out and
  cycle membership via strongly-connected components; betweenness is computed for the
  dependency-map visualisation but is not a scoring feature).
- Scores each interface for **integration-defect likelihood** using a transparent, explainable
  **heuristic**, an optional **machine-learning** model (RandomForest), or — experimentally — a
  **graph-relational model (`RiskGNN`)** that captures risk *propagating* across the architecture.
- Emits a ranked **risk report** with plain-language reasons to focus early review.

## Install
```bash
pip install -e ".[ml]"     # ml extra enables the learned model; core works without it
```

## Quickstart
```bash
# Heuristic (no training needed)
python -m safeshift analyze examples/example_adas_architecture.yaml

# Train the optional ML model on synthetic data, write a report
python -m safeshift analyze examples/example_adas_architecture.yaml --train --out report.md

# A larger, connected-vehicle / software-defined-vehicle example
python -m safeshift analyze examples/example_connected_vehicle_architecture.yaml --train --out cv_report.md

# Experimental graph-relational model (RiskGNN, synthetic-trained — research/demo, not calibrated)
python -m safeshift analyze examples/example_adas_architecture.yaml --gnn --out gnn_report.md
```

## Worked examples
Two illustrative, non-proprietary architectures ship in `examples/`:
- **`example_adas_architecture.yaml`** — a camera+radar ADAS domain (11 components / 12 interfaces).
- **`example_connected_vehicle_architecture.yaml`** — a software-defined / connected-vehicle
  architecture (14 components / 19 interfaces) spanning telematics, V2X, OTA update, and
  infotainment, designed to exercise the externally-reachable attack surface governed by
  UNECE R155/R156 and ISO/SAE 21434.

## Example output (abridged)
```
# SafeShift Risk Report — Reference ADAS / E-E Architecture (illustrative)
- Components: 11  |  Interfaces: 12
- Model mode: learned
- Interfaces flagged HIGH risk: 7

## Ranked integration-risk hotspots
| Rank | Interface | From → To | Protocol | Risk | Band |
| 1 | if_vpm_fusion | vpm → fusion | Ethernet | 0.92 | HIGH |
| 2 | if_rf_fusion | radar_front → fusion | CAN-FD | 0.90 | HIGH |
| 3 | if_rc_fusion | radar_corner → fusion | CAN-FD | 0.84 | HIGH |
...
```

## How it decides
Interfaces are flagged using factors well established in integration practice: safety/timing
criticality, crossing a supplier boundary, higher-complexity protocols, high signal counts,
participation in a dependency cycle, high ASIL on an immature component, and structural centrality.
See [`docs/methodology.md`](docs/methodology.md) and [`docs/schema.md`](docs/schema.md).

## Evaluation

**Core (held-out synthetic benchmark).** Labels come from a *latent, non-linear risk function
distinct from SafeShift's own heuristic*. Informed models recover the target at ~0.81 ROC-AUC —
above a random baseline (0.52) **and** an informed single-rule baseline (0.70) — with 5-fold
cross-validation stable at 0.810 ± 0.006. The transparent heuristic (0.806, 95% CI [0.79, 0.82]) is
statistically indistinguishable from logistic regression (0.812) and a random forest (0.803),
supporting the explainable model in safety contexts. An ablation shows the safety/ASIL group
contributes most (ROC-AUC drops 0.158 when removed). **This measures construct recovery, not
predictive validity:** the benchmark shares the scorer's features and signs, so flipping the
heuristic's signs inverts its AUC to 0.19, and a latent driver *outside* the feature set lowers
informed performance to ~0.63–0.67.

**Extended (v0.3.0).**
- **Robustness** — the ranking holds across *four alternative functional forms* of the risk function
  (all sharing the scorer's features and signs); an *off-feature latent-driver* stress test drops
  informed models to ~0.63–0.67, honestly bounding what the synthetic study establishes.
- **Scalability** — a 500-component / 956-interface architecture is analyzed in ~36 ms; with
  SCC-based cycle detection, a densely cyclic 500-component graph (128 nodes on directed cycles) is
  analyzed in ~0.1 s.
- **Standards overlap** — on the connected-vehicle example a single design-time pass *enumerates* the
  externally-reachable attack surface (UNECE R155/R156, ISO/SAE 21434) alongside integration risk.
  Most interfaces (15/19) are reachable, so high-risk interfaces fall on that surface at the base
  rate (10/12; hypergeometric P≈0.5), **not** above it — a workflow convenience (one pass, two work
  products), not a correlation.

**Graph-relational extension (v0.4.0).** The heuristic and RandomForest score each interface from
its *own* feature vector, so they are structurally blind to integration risk that **propagates**
through the architecture — an immature, defect-prone subsystem raising the risk of interfaces several
hops away. `RiskGNN` (a small, pure-NumPy directed graph neural network, `src/safeshift/gnn.py`,
hand-derived gradients verified by a finite-difference check) adds message passing over the
dependency graph. A 5-seed **dose-response** study (`evaluation/graph_eval.py`) sweeps a propagation
strength α, with **α=0 as a built-in negative control** (no propagation; the label is a pure function
of the standard per-interface features, so a topology-aware model can have no advantage):

| α | Heuristic | RandomForest | RiskGNN | RiskGNN − RF |
|---|----------:|-------------:|--------:|-------------:|
| 0 (control) | 0.748 | 0.725 | 0.734 | +0.009 |
| 0.6 | 0.748 | 0.713 | 0.751 | +0.037 |
| 0.9 | 0.731 | 0.697 | 0.767 | +0.069 |

(held-out ROC-AUC, mean over 5 seeds.) The graph model's edge over the *learned* per-interface model
(RandomForest) is ≈0 in the control and **grows monotonically with propagation strength** — the
signature of recovering multi-hop information no per-interface feature vector contains, not of being
a generically stronger learner. As α rises the per-interface RandomForest and heuristic actually
*decline* (the propagation signal is invisible noise to them) while only `RiskGNN` tracks the rising
Bayes-optimal ceiling. In fairness, the **transparent heuristic is a strong baseline**: RiskGNN
overtakes it only once propagation is at least moderate (α ≥ 0.6); at weak propagation the heuristic
remains competitive. **This is synthetic, construct-level evidence:** it shows *when* a topology-aware
model is warranted (when integration
risk genuinely propagates), not that the GNN is superior on real outcomes — which, as for every
result here, requires calibration on real labeled integration defects. See
`evaluation/results_graph.md`.

Reproduce:
```bash
pip install -e ".[dev]"
python evaluation/run_eval.py        # core: writes evaluation/results.md and figures/
python evaluation/extended.py all    # extended: robustness, scalability, dependency maps, overlap
python evaluation/graph_eval.py      # graph-relational dose-response: writes results_graph.md
```

See `evaluation/results.md` for full tables (held-out metrics, ablation, noise robustness, feature
importance). Results are on synthetic data; external validity requires calibration on real labeled
integration outcomes.

An **expert-validation harness** (`evaluation/expert_study.py`) is also included: it compares
SafeShift's interface rankings against blinded domain-expert rankings (Spearman, Kendall's W,
Fleiss' κ, top-k overlap, HIGH-band F1) once expert ratings are collected — a face-validity check
that is independent of the synthetic evaluation. (The harness ships ready to run; collecting the
expert ratings is future work.)

## Tests
```bash
pip install -e ".[dev]"
pytest        # unit tests for schema, graph, features, model, and report
```

## Scope & honesty
SafeShift is a **reference implementation and method**, not a certified tool. Risk scores are
**decision-support indicators** derived from the architecture description and, in learned mode, a
**synthetic** training set that encodes the same risk relationships. The repository ships with
**illustrative, non-proprietary** example data. Real-world use should calibrate the model against
an organization's own historical integration outcomes (see the evaluation roadmap in the paper).

## Roadmap
- ARXML / AUTOSAR import; richer timing and resource models.
- Calibration against labeled historical outcomes.
- CI integration and architecture-diff risk deltas.
- Alignment hooks for ASPICE work products and ISO 26262 / ISO 21434 evidence.

## Citation
If you use SafeShift, please cite the archived software (the concept DOI resolves to the latest
version):

> Abdelmaksoud, O. *SafeShift: Open Shift-Left Integration-Risk Prediction for Automotive Software
> Architectures.* Zenodo. https://doi.org/10.5281/zenodo.20780068

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License
Apache-2.0. See [`LICENSE`](LICENSE).

## Author
Created and maintained by Omar Abdelmaksoud.
