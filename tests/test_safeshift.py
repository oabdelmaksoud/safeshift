import os
import pytest
from safeshift import (Architecture, Component, Interface, load_architecture,
                    build_dependency_graph, extract_interface_features,
                    RiskModel, generate_report)

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "example_adas_architecture.yaml")


def _toy() -> Architecture:
    comps = [Component(id="a", supplier="X", asil="D", maturity=0.2),
            Component(id="b", supplier="Y", asil="B", maturity=0.9)]
    itfs = [Interface(id="e1", source="a", target="b", protocol="Ethernet",
                    signals=20, safety_related=True, timing_critical=True)]
    return Architecture(name="toy", components=comps, interfaces=itfs)


def test_load_example():
    arch = load_architecture(EX)
    assert len(arch.components) == 11
    assert len(arch.interfaces) == 12
    assert arch.validate() == []


def test_validation_catches_bad_edge():
    arch = Architecture(components=[Component(id="a")],
                        interfaces=[Interface(id="e", source="a", target="ghost")])
    problems = arch.validate()
    assert any("ghost" in p for p in problems)


def test_graph_build():
    g = build_dependency_graph(_toy())
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1


def test_features_have_all_keys():
    feats = extract_interface_features(_toy())
    assert "e1" in feats
    assert feats["e1"]["supplier_boundary"] == 1.0  # a and b differ in supplier
    assert feats["e1"]["safety_related"] == 1.0


def test_heuristic_scores_in_range():
    model = RiskModel()  # heuristic
    feats = extract_interface_features(load_architecture(EX))
    for _, s in model.rank_interfaces(feats):
        assert 0.0 <= s <= 1.0


def test_learned_model_trains_and_ranks():
    model = RiskModel().train(n=500, seed=1)
    assert model.mode in ("learned", "heuristic")
    feats = extract_interface_features(load_architecture(EX))
    ranked = model.rank_interfaces(feats)
    assert len(ranked) == 12
    assert ranked == sorted(ranked, key=lambda t: t[1], reverse=True)


def test_high_risk_safety_interface_outranks_benign():
    """A safety+timing-critical Ethernet link across suppliers should outrank a benign HMI link."""
    arch = load_architecture(EX)
    model = RiskModel()
    feats = extract_interface_features(arch)
    scores = dict(model.rank_interfaces(feats))
    assert scores["if_fusion_adc"] > scores["if_gw_hmi"]


def test_report_generation():
    arch = load_architecture(EX)
    report = generate_report(arch, RiskModel())
    assert "SafeShift Risk Report" in report
    assert "Ranked integration-risk hotspots" in report
