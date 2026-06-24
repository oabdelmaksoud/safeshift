"""Synthetic architectures with PROPAGATING integration risk — for the graph-relational study and
for training the optional RiskGNN. (Lives in the package, like model.generate_synthetic, so the CLI
and the evaluation harness share one canonical generator.)

MODELING ASSUMPTION (a hypothesis, NOT an established empirical fact): integration defects
*cascade* along dependencies. A defect-prone upstream component (immature, high-ASIL) raises the
integration risk not only of its immediate consumers but, attenuated, of components several hops
downstream. We encode this as a short multi-hop diffusion of a latent per-component "trouble" along
edge direction (source -> target = data/dependency flow), then make each interface's defect
probability depend on local engineering factors PLUS the *propagated* trouble at its endpoints.

Why this enables an HONEST comparison (and a clean negative control):

  label_logit(e=s->t) = LOCAL(standard per-interface features) + c_trouble * (excess_s + excess_t)

  excess_v = T_v - t0_v   where   T_v = t0_v + alpha * mean_{u: u->v} T_u   (iterated `hops` times)

* `excess_v` is *strictly* the neighbour-derived contribution (it is identically 0 when alpha=0).
* The LOCAL term is a function of exactly the features `extract_interface_features` already exposes,
  so a per-interface model (RandomForest) is NOT crippled — it keeps its full legitimate signal.
* At alpha=0 the label is a pure function of those standard features => a per-interface model and a
  graph model see the SAME recoverable signal (the NEGATIVE CONTROL: they should tie).
* At alpha>0 the label depends on multi-hop neighbour attributes that NO per-interface feature
  vector contains. Only a model that aggregates over the graph (a GNN) can recover them.

A component's OWN hidden node noise enters `t0` but cancels out of `excess` (excess removes t0), so
it only influences *downstream* interfaces — i.e. it is an unobserved upstream driver, exactly the
kind of latent factor real integration risk has. This bounds achievable accuracy for every model.

Whether REAL integration risk propagates this way is an empirical question to be settled by
calibration on real labeled outcomes — not by this generator. Parameters below were chosen once on
engineering grounds; they are NOT tuned to force any particular model-comparison outcome.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from .schema import Architecture, Component, Interface, ASIL_LEVELS
from .graph import build_dependency_graph
from .features import extract_interface_features

_PROTOS = ["CAN", "CAN-FD", "Ethernet", "FlexRay", "LIN"]
_ASILS = ["QM", "A", "B", "C", "D"]


def make_architecture(n_comp: int = 24, seed: int = 0, back_edge_p: float = 0.08) -> Architecture:
    """A layered, sparse, mostly feed-forward automotive-like architecture with a few back-edges
    (so genuine directed cycles exist). Topology + node attributes only — no labels."""
    rng = np.random.default_rng(seed)
    n_layers = max(4, int(round(np.sqrt(max(n_comp, 1)))))
    layer = {i: int(rng.integers(0, n_layers)) for i in range(n_comp)}
    comps = [Component(id=f"c{i}", name=f"c{i}", kind="ecu",
                       supplier=f"S{int(rng.integers(0, 8))}",
                       asil=_ASILS[int(rng.integers(0, 5))],
                       maturity=float(round(rng.random(), 2))) for i in range(n_comp)]
    itfs: list[Interface] = []
    eid = 0
    for i in range(n_comp):
        nxt = [t for t in range(n_comp) if layer[t] > layer[i]]
        if not nxt:
            continue
        for _ in range(int(rng.integers(1, 4))):  # ~1-3 forward edges -> sparse
            t = int(rng.choice(nxt))
            itfs.append(Interface(id=f"e{eid}", source=f"c{i}", target=f"c{t}",
                                  protocol=_PROTOS[int(rng.integers(0, len(_PROTOS)))],
                                  signals=int(rng.integers(1, 40)),
                                  safety_related=bool(rng.integers(0, 2)),
                                  timing_critical=bool(rng.integers(0, 2))))
            eid += 1
    for i in range(n_comp):  # back-edges -> directed cycles (feedback paths)
        if rng.random() < back_edge_p:
            prev = [t for t in range(n_comp) if layer[t] < layer[i]]
            if prev:
                t = int(rng.choice(prev))
                itfs.append(Interface(id=f"e{eid}", source=f"c{i}", target=f"c{t}",
                                      protocol="CAN", signals=int(rng.integers(1, 20)),
                                      safety_related=bool(rng.integers(0, 2))))
                eid += 1
    return Architecture(name=f"synthgraph-{n_comp}-{seed}", components=comps, interfaces=itfs)


def _node_arrays(arch: Architecture):
    ids = [c.id for c in arch.components]
    idx = {cid: i for i, cid in enumerate(ids)}
    mat = np.array([c.maturity for c in arch.components], dtype=float)
    asil = np.array([ASIL_LEVELS.get(str(c.asil).upper().replace("ASIL", "").strip(), 0)
                     for c in arch.components], dtype=float)
    return ids, idx, mat, asil


def propagated_trouble(arch: Architecture, seed: int = 0, alpha: float = 0.6, hops: int = 2,
                       node_noise: float = 0.3, a_mat: float = 1.0, b_asil: float = 1.0):
    """Return (t0, T, excess) per node. `excess = T - t0` is the strictly neighbour-derived
    (multi-hop) contribution; it is identically zero when alpha == 0."""
    rng = np.random.default_rng(2_000_000 + seed)
    ids, idx, mat, asil = _node_arrays(arch)
    n = len(ids)
    t0 = a_mat * (1.0 - mat) + b_asil * (asil / 4.0) + node_noise * rng.standard_normal(n)
    g = nx.DiGraph(build_dependency_graph(arch))  # collapse parallel edges for propagation
    preds = {cid: list(g.predecessors(cid)) for cid in ids}
    T = t0.copy()
    for _ in range(hops):
        newT = t0.copy()
        for cid in ids:
            ps = preds[cid]
            if ps:
                newT[idx[cid]] += alpha * float(np.mean([T[idx[u]] for u in ps]))
        T = newT
    return t0, T, (T - t0)


def label_architecture(arch: Architecture, seed: int = 0, alpha: float = 0.6, hops: int = 2,
                       c_trouble: float = 0.8, node_noise: float = 0.3,
                       a_mat: float = 1.0, b_asil: float = 1.0):
    """Return (labels, probs): per-interface Bernoulli labels and their probabilities.

    label_logit = LOCAL(standard features) + c_trouble * (excess_source + excess_target).
    With alpha=0, excess == 0 and the label is a pure function of the standard per-interface features.
    """
    _, idx, _, _ = _node_arrays(arch)
    _, _, excess = propagated_trouble(arch, seed=seed, alpha=alpha, hops=hops,
                                      node_noise=node_noise, a_mat=a_mat, b_asil=b_asil)
    feats = extract_interface_features(arch)
    rng = np.random.default_rng(3_000_000 + seed)  # independent label-draw stream
    probs: dict[str, float] = {}
    labels: dict[str, int] = {}
    for itf in arch.interfaces:
        f = feats[itf.id]
        local = (-3.0
                 + 1.4 * f["safety_related"]
                 + 0.9 * f["timing_critical"]
                 + 1.4 * f["supplier_boundary"] * f["protocol_mismatch_risk"]
                 + 0.22 * np.sqrt(f["signals"])
                 + 0.15 * f["max_asil_rank"]
                 - 1.0 * f["min_maturity"]
                 + 0.9 * f["tgt_in_cycle"])
        p_e = float(excess[idx[itf.source]] + excess[idx[itf.target]])
        z = local + c_trouble * p_e
        prob = 1.0 / (1.0 + np.exp(-z))
        probs[itf.id] = float(prob)
        labels[itf.id] = int(rng.random() < prob)
    return labels, probs


def make_graph_dataset(n_graphs: int, seed: int = 0, alpha: float = 0.6,
                       n_comp_range=(18, 30), **kw):
    """A list of {'arch', 'labels', 'probs'} samples. alpha controls propagation strength;
    alpha=0 is the local-only negative control."""
    rng = np.random.default_rng(seed)
    data = []
    for k in range(n_graphs):
        nc = int(rng.integers(n_comp_range[0], n_comp_range[1]))
        arch = make_architecture(n_comp=nc, seed=seed * 10_000 + k)
        labels, probs = label_architecture(arch, seed=seed * 10_000 + k, alpha=alpha, **kw)
        data.append({"arch": arch, "labels": labels, "probs": probs})
    return data
