"""US-002 / US-004 — RiskGNN correctness: finite-difference gradient check (proves backprop),
loss decrease, prediction range, and a learning-sanity check on strongly-propagating data."""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from safeshift import load_architecture, RiskGNN, graph_synth as G  # noqa: E402
from safeshift.gnn import (architecture_to_tensors, init_params,  # noqa: E402
                           loss_and_grads, loss_only)

ADAS = os.path.join(HERE, "..", "examples", "example_adas_architecture.yaml")


def _tiny_batch():
    graphs = []
    for k in range(2):
        arch = G.make_architecture(n_comp=12, seed=100 + k)
        labels, _ = G.label_architecture(arch, seed=100 + k, alpha=0.6)
        graphs.append(architecture_to_tensors(arch, labels))
    return graphs


def test_gradient_check_matches_finite_differences():
    """Analytic gradients must match central finite differences for every parameter block."""
    graphs = _tiny_batch()
    p = init_params(hidden=5, readout_hidden=4, seed=1)
    _, grads = loss_and_grads(p, graphs)

    eps = 1e-5
    worst = 0.0
    for key, val in p.items():
        flat = val.reshape(-1)
        gflat = grads[key].reshape(-1)
        # check up to 12 entries per block (deterministic stride) to keep the test fast
        n = flat.size
        idxs = range(0, n, max(1, n // 12))
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss_only(p, graphs)
            flat[i] = orig - eps
            lm = loss_only(p, graphs)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
            worst = max(worst, rel)
    assert worst < 1e-4, f"gradient check failed: worst relative error {worst:.2e}"


def test_training_decreases_loss():
    data = G.make_graph_dataset(12, seed=4, alpha=0.6)
    gnn = RiskGNN(hidden=12, seed=0).train(data, epochs=120, lr=0.02)
    assert gnn.history[0] > gnn.history[-1], "BCE loss must decrease over training"
    assert gnn.history[-1] < gnn.history[0] * 0.9, "loss should drop meaningfully"


def test_predict_returns_one_probability_per_interface():
    arch = load_architecture(ADAS)
    gnn = RiskGNN(hidden=8, seed=0)
    preds = gnn.predict(arch)
    assert set(preds) == {i.id for i in arch.interfaces}
    assert all(0.0 <= s <= 1.0 for s in preds.values())


def test_learns_above_chance_on_propagating_data():
    """Learning-sanity (NOT a comparison vs RF): on strongly-propagating data the GNN should
    reach held-out ROC-AUC clearly above chance."""
    train = G.make_graph_dataset(40, seed=10, alpha=0.7)
    val = G.make_graph_dataset(12, seed=500, alpha=0.7)
    test = G.make_graph_dataset(20, seed=999, alpha=0.7)
    gnn = RiskGNN(hidden=16, seed=0).train(train, epochs=600, lr=0.02,
                                           weight_decay=3e-3, val_graphs=val, patience=60)
    ys, ps = [], []
    for s in test:
        pred = gnn.predict(s["arch"])
        for iid, lab in s["labels"].items():
            ys.append(lab)
            ps.append(pred[iid])
    auc = roc_auc_score(ys, ps)
    assert auc >= 0.65, f"GNN failed to learn above chance on propagating data (AUC={auc:.3f})"
