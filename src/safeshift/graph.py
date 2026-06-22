"""Build a directed dependency graph from an architecture and compute structural metrics.

Structural position in the integration graph is a strong, well-established predictor of where
integration problems concentrate: highly connected hubs, components inside dependency cycles,
and interfaces that bridge otherwise separate clusters are classic trouble spots.
"""
from __future__ import annotations
import networkx as nx
from .schema import Architecture


def build_dependency_graph(arch: Architecture) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph(name=arch.name)
    for c in arch.components:
        g.add_node(c.id, **{"name": c.name, "kind": c.kind, "supplier": c.supplier,
                            "asil": c.asil, "maturity": c.maturity})
    for itf in arch.interfaces:
        g.add_edge(itf.source, itf.target, key=itf.id, protocol=itf.protocol,
                signals=itf.signals, safety_related=itf.safety_related,
                timing_critical=itf.timing_critical)
    return g


def structural_metrics(g: nx.MultiDiGraph) -> dict[str, dict[str, float]]:
    """Per-component structural metrics used as risk features."""
    simple = nx.DiGraph(g)  # collapse parallel edges for centrality measures
    metrics: dict[str, dict[str, float]] = {}
    try:
        betw = nx.betweenness_centrality(simple) if simple.number_of_nodes() > 2 else {}
    except Exception:
        betw = {}
    # Components that participate in any directed cycle. We need cycle *membership* (a boolean),
    # not an enumeration of cycles, so we use strongly-connected components (Tarjan, O(V+E))
    # rather than nx.simple_cycles, which enumerates every simple cycle and is exponential in
    # the worst case. A node lies on a directed cycle iff its SCC has more than one node, or it
    # carries a self-loop. This yields membership identical to a full cycle enumeration while
    # remaining linear even on dense or deeply cyclic graphs.
    in_cycle: set[str] = set()
    try:
        for scc in nx.strongly_connected_components(simple):
            if len(scc) > 1:
                in_cycle.update(scc)
        in_cycle.update(n for n in simple.nodes() if simple.has_edge(n, n))
    except Exception:
        pass
    for n in g.nodes():
        metrics[n] = {
            "fan_in": float(g.in_degree(n)),
            "fan_out": float(g.out_degree(n)),
            "degree": float(g.degree(n)),
            "betweenness": float(betw.get(n, 0.0)),
            "in_cycle": 1.0 if n in in_cycle else 0.0,
        }
    return metrics
