# SafeShift

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
- Builds a **dependency graph** and computes structural risk metrics (fan-in/out, betweenness,
  dependency cycles).
- Scores each interface for **integration-defect likelihood** using either a transparent,
  explainable **heuristic** or an optional **machine-learning** model (RandomForest).
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
```

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

## Scope & honesty
SafeShift is a **reference implementation and method**, not a certified tool. Risk scores are
**decision-support indicators** derived from the architecture description and, in learned mode, a
**synthetic** training set that encodes the same risk relationships. The repository ships with
**illustrative, non-proprietary** example data. Real-world use should calibrate the model against
an organization's own historical integration outcomes (see the evaluation roadmap in the paper).

## Evaluation

An evaluation harness is included (`evaluation/`). Against an *independent*, non-linear synthetic
ground truth (distinct from SafeShift's own heuristic, to avoid circular evaluation), informed
models reach ~0.81 ROC-AUC versus ~0.52 for a random baseline; 5-fold cross-validation is stable
at 0.810 +/- 0.006. The transparent heuristic (0.806) is competitive with logistic regression
(0.812) and a random forest (0.803), supporting the use of the explainable model in safety
contexts. An ablation shows the safety/ASIL feature group contributes most (ROC-AUC drops 0.158
when removed). Reproduce with:

```bash
pip install -e ".[dev]"
python evaluation/run_eval.py    # writes evaluation/results.md and figures/
```

See `evaluation/results.md` for full tables (held-out metrics, ablation, noise robustness,
feature importance). Results are on synthetic data; external validity requires calibration on
real labeled integration outcomes.

## Roadmap
- ARXML / AUTOSAR import; richer timing and resource models.
- Calibration against labeled historical outcomes.
- CI integration and architecture-diff risk deltas.
- Alignment hooks for ASPICE work products and ISO 26262 / ISO 21434 evidence.

## License
Apache-2.0. See [`LICENSE`](LICENSE). Cite via [`CITATION.cff`](CITATION.cff).

## Author
Created and maintained by Omar Abdelmaksoud.
