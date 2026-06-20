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
2. **Build** a directed dependency graph and compute structural metrics (fan-in/out,
   betweenness, cycle membership).
3. **Extract features** per interface combining structural position with engineering attributes
   known to elevate integration risk: supplier-boundary crossings, protocol complexity, high
   signal counts, safety/timing criticality, high ASIL on immature components.
4. **Score** each interface with either:
   - a **transparent heuristic** (explainable weighted sum), or
   - a **learned RandomForest** trained on a synthetic generator encoding the same relationships
     plus noise (optional; falls back to the heuristic).
5. **Report** ranked hotspots with plain-language reasons, to focus early review.

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
