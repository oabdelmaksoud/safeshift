"""Extract numeric feature vectors for interfaces and components.

Features combine (a) structural graph position and (b) engineering attributes known to raise
integration risk: crossing a supplier boundary, mismatched protocols, high signal counts,
safety/timing criticality, low component maturity, and high ASIL on an immature component.
"""
from __future__ import annotations
from .schema import Architecture, Interface
from .graph import build_dependency_graph, structural_metrics

INTERFACE_FEATURE_NAMES = [
    "signals", "safety_related", "timing_critical", "supplier_boundary",
    "protocol_mismatch_risk", "src_fan_out", "tgt_fan_in", "tgt_in_cycle",
    "max_asil_rank", "min_maturity",
]

# coarse, transparent protocol-risk weighting (higher = more integration-prone)
_PROTOCOL_RISK = {"internal": 0.0, "LIN": 0.3, "SPI": 0.3, "CAN": 0.5,
                  "CAN-FD": 0.5, "FlexRay": 0.7, "Ethernet": 0.8}


def extract_interface_features(arch: Architecture) -> dict[str, dict[str, float]]:
    g = build_dependency_graph(arch)
    sm = structural_metrics(g)
    idx = arch.component_index()
    out: dict[str, dict[str, float]] = {}
    for itf in arch.interfaces:
        src, tgt = idx[itf.source], idx[itf.target]
        out[itf.id] = {
            "signals": float(itf.signals),
            "safety_related": 1.0 if itf.safety_related else 0.0,
            "timing_critical": 1.0 if itf.timing_critical else 0.0,
            "supplier_boundary": 1.0 if src.supplier != tgt.supplier else 0.0,
            "protocol_mismatch_risk": _PROTOCOL_RISK.get(itf.protocol, 0.5),
            "src_fan_out": sm[itf.source]["fan_out"],
            "tgt_fan_in": sm[itf.target]["fan_in"],
            "tgt_in_cycle": sm[itf.target]["in_cycle"],
            "max_asil_rank": float(max(src.asil_rank(), tgt.asil_rank())),
            "min_maturity": float(min(src.maturity, tgt.maturity)),
        }
    return out


def extract_component_features(arch: Architecture) -> dict[str, dict[str, float]]:
    g = build_dependency_graph(arch)
    return structural_metrics(g)
