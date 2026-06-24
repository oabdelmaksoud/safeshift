# SafeShift Methodology

## Problem
In a distributed automotive E/E architecture, software components are developed in separate
streams and integrated late, often only when physical prototypes exist. Integration and
architectural defects discovered at that stage are expensive and a recurring source of recalls.
SafeShift aims to surface likely integration trouble spots during *virtual* design — a
"shift-left" of validation effort.

## Approach
1. **Describe** the architecture in an open schema: components (with supplier, ASIL, maturity)
   and interfaces (protocol, signal count, safety/timing relevance).
2. **Build** a directed dependency graph and compute structural metrics: fan-in/out and cycle
   membership (via strongly-connected components, O(V+E) — no cycle enumeration). Betweenness
   centrality is also computed for the dependency-map visualisation, but it is not used as a
   scoring feature.
3. **Extract features** per interface combining structural position with engineering attributes
   known to elevate integration risk: supplier-boundary crossings, protocol complexity, high
   signal counts, safety/timing criticality, high ASIL on immature components.
4. **Score** each interface with either:
   - a **transparent heuristic** (explainable weighted sum), or
   - a **learned RandomForest** trained on a synthetic generator encoding the same relationships
     plus noise (optional; falls back to the heuristic).
5. **Report** ranked hotspots with plain-language reasons, to focus early review.

## Graph-relational model (v0.4.0)
The heuristic and RandomForest score each interface from its own feature vector. They therefore
cannot represent integration risk that **propagates** through the architecture — e.g. a defect-prone,
immature upstream subsystem elevating the risk of interfaces several hops downstream. `RiskGNN`
(`src/safeshift/gnn.py`) is a compact, pure-NumPy directed graph neural network: two message-passing
layers with separate mean aggregation over in- and out-neighbours, then an edge-readout MLP over
`[h_source, h_target, edge_features]`. Its gradients are hand-derived and verified against finite
differences; training uses full-batch AdamW with validation-based early stopping.

**The advantage is conditional, and we test it honestly.** `evaluation/graph_synth.py` generates
synthetic architectures whose risk cascades multi-hop with an adjustable strength α (a *modeling
assumption*, not an established fact: that integration defects propagate along dependencies). The
per-interface label is `LOCAL(standard features) + c · (propagated neighbour trouble)`; the second
term is identically zero when α=0, so α=0 is a **negative control** in which the label is a pure
function of the standard features and a graph model can have no advantage. A 5-seed dose-response
(`evaluation/graph_eval.py`) shows the RiskGNN−RandomForest ROC-AUC gap is ≈0 at α=0 and grows
monotonically with α (to ≈+0.07 at α=0.9), while per-interface models *decline* as risk propagates.
(We disclose the most favourable assumption: the generator's propagation depth is set equal to the
GNN's two message-passing layers, so the model class can in principle represent the generative
process. This is a deliberate, stated choice — the point is to demonstrate the conditional necessity
of topology awareness, not to claim a depth the model could not capture.)
This demonstrates **when** a topology-aware model is warranted — when integration risk genuinely
propagates — not that the GNN is superior on real integration outcomes. Establishing whether, and
how strongly, real risk propagates requires calibration on real labeled defects (see the roadmap).

## Honesty and scope
SafeShift is a reference implementation and method. Scores are decision-support indicators
derived from the architecture description and (in learned mode) synthetic training data. They
are not guarantees, and the tool ships with illustrative, non-proprietary example data. Real
deployment would calibrate the model against an organization's own historical integration
outcomes — an explicit item on the evaluation roadmap.

## Why open
Large OEMs can fund bespoke tooling; much of the supplier base cannot. Releasing the method and
a working implementation openly lets the broader U.S. automotive base adopt a common, auditable
approach to early validation, rather than each firm rebuilding it in isolation.
