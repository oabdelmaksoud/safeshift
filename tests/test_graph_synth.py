"""US-001 — propagating synthetic generator: validity, determinism, and the crucial
feature/label SEPARATION (the multi-hop signal must be absent from per-interface features)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from safeshift import graph_synth as G  # noqa: E402
from safeshift.features import extract_interface_features  # noqa: E402


def _local_logit_from_features(f) -> float:
    """Exact copy of the LOCAL term in graph_synth.label_architecture (features-only)."""
    return (-3.0
            + 1.4 * f["safety_related"]
            + 0.9 * f["timing_critical"]
            + 1.4 * f["supplier_boundary"] * f["protocol_mismatch_risk"]
            + 0.22 * np.sqrt(f["signals"])
            + 0.15 * f["max_asil_rank"]
            - 1.0 * f["min_maturity"]
            + 0.9 * f["tgt_in_cycle"])


def test_generator_is_valid_and_deterministic():
    a1 = G.make_architecture(n_comp=24, seed=7)
    a2 = G.make_architecture(n_comp=24, seed=7)
    assert a1.validate() == []
    assert [c.id for c in a1.components] == [c.id for c in a2.components]
    assert [(i.id, i.source, i.target) for i in a1.interfaces] == \
           [(i.id, i.source, i.target) for i in a2.interfaces]


def test_alpha_zero_gives_exactly_zero_excess__alpha_positive_propagates():
    arch = G.make_architecture(n_comp=26, seed=3)
    _, _, exc0 = G.propagated_trouble(arch, seed=3, alpha=0.0)
    _, _, exc6 = G.propagated_trouble(arch, seed=3, alpha=0.6)
    assert np.allclose(exc0, 0.0), "alpha=0 must disable propagation exactly (negative control)"
    assert np.max(np.abs(exc6)) > 0.1, "alpha>0 must inject a real multi-hop signal"


def test_label_is_pure_feature_function_at_alpha0_but_not_at_alpha_positive():
    """At alpha=0 the label probability is reconstructible from the standard per-interface
    features alone (a per-interface model can recover it). At alpha>0 it is NOT — the difference
    is multi-hop neighbour information that no per-interface feature vector contains."""
    arch = G.make_architecture(n_comp=28, seed=5)
    feats = extract_interface_features(arch)  # NOTE: features never see `alpha`

    _, probs0 = G.label_architecture(arch, seed=5, alpha=0.0)
    _, probs6 = G.label_architecture(arch, seed=5, alpha=0.6)

    # (a) alpha=0: probability == sigmoid(local(features)) for EVERY interface => feature-recoverable
    for iid, f in feats.items():
        expected = 1.0 / (1.0 + np.exp(-_local_logit_from_features(f)))
        assert probs0[iid] == pytest.approx(expected, abs=1e-9)

    # (b) alpha>0: at least some interfaces shift, driven by propagation absent from features
    shifts = [abs(probs6[iid] - probs0[iid]) for iid in feats]
    assert max(shifts) > 0.05, "propagation must change labels beyond what features encode"


def test_dataset_shapes_and_control():
    prop = G.make_graph_dataset(4, seed=2, alpha=0.6)
    ctrl = G.make_graph_dataset(4, seed=2, alpha=0.0)
    assert len(prop) == len(ctrl) == 4
    for s in prop:
        assert s["arch"].validate() == []
        assert set(s["labels"]) == {i.id for i in s["arch"].interfaces}
        assert all(v in (0, 1) for v in s["labels"].values())
