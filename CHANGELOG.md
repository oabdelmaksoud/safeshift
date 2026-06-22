# Changelog

All notable changes to SafeShift are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-06-22 — evaluation hardening

This release responds to an adversarial methodological self-review of the evaluation (§3, §5, §6
of the technical paper). No headline number from v0.2.0 changed; the revisions correct how results
are *framed* and add rigor so the synthetic study is not over-read. Intellectual honesty was the
governing constraint: every change either reframes a claim accurately or adds an additive,
honesty-preserving experiment. None manufacture real-world validation.

### Changed (framing/accuracy)
- **Evaluation framed as construct recovery, not predictive validity.** The synthetic benchmark
  shares the scorer's features and their directional signs, so ~0.81 ROC-AUC measures recoverability
  of a known target, not accuracy on real integration outcomes. Abstract, §3.4, §6, §9, and §13
  now state this explicitly.
- **"Independent ground truths" → "alternative functional forms."** The four §6.4 generators are
  *aligned* with the scorer (same features/signs, different form); the paper no longer calls them
  "independent."
- **Cybersecurity attack-surface overlap (§5.2) corrected.** In the connected-vehicle example 79%
  of interfaces are externally reachable, so the high-risk/attack-surface overlap (10/12; top-10 8/10)
  is **at the base rate** (hypergeometric P ≈ 0.5; no rank association). Reframed from "risk
  concentrates on the attack surface" to "one design-time pass *enumerates* both surfaces" — a
  workflow benefit, not a correlation.
- **Betweenness centrality** is documented as computed for the dependency-map visualisation only; it
  is not a scoring feature (it never was).

### Added (rigor)
- **Informed baselines** in the held-out comparison (Table 6): a single-feature (safety-only, 0.70)
  and a two-rule (safety × immaturity, 0.72) baseline, so the full model's marginal value (~0.09–0.11
  ROC-AUC over a trivial rule, non-overlapping CIs) is visible rather than measured only against a
  random (0.52) floor.
- **Bootstrap 95% confidence intervals** on every held-out ROC-AUC. The three informed models'
  intervals overlap, making "the explainable heuristic is competitive" a statistically supported claim.
- **Circularity diagnostics** (`results.json`): a sign-flip test (heuristic AUC inverts to 0.194) and
  a single-feature reference, making the construct-recovery dependence explicit.
- **Off-feature latent-driver stress test** (Table 8, final row): when ~half the systematic risk comes
  from a factor outside the feature set, informed models drop to ~0.63–0.67 — bounding what the
  synthetic study can claim.
- **Cyclic scalability stress test** (Table 9): a densely cyclic 500-component graph (128 nodes on
  directed cycles) analyzed in ~0.1 s.

### Performance / correctness
- **Cycle membership now uses strongly-connected components (Tarjan, O(V+E))** instead of
  `networkx.simple_cycles` (exponential worst case). Membership is identical on all tested graphs;
  this removes the cycle-enumeration scalability caveat noted in earlier versions (§3.5, §6.5, §12).

### Notes
- `evaluation/expert_study.py` is labeled a forward-looking template: no expert data has been
  collected and no paper claim depends on it.
- All results remain on synthetic data; external validity still requires calibration on real labeled
  integration outcomes (§8).

## [0.2.0] — 2026-06-20 — extended evaluation
- Robustness study across alternative ground-truth forms; scalability benchmark; connected-vehicle
  worked example with cyber attack-surface analysis; risk-coloured dependency-map figures.
- Archived with a citable Zenodo DOI.

## [0.1.0] — 2026-06 — initial public release
- Open schema, dependency-graph features, transparent heuristic + optional learned model, CLI,
  examples, tests, and the core synthetic evaluation. Apache-2.0.
