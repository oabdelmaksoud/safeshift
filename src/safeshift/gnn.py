"""RiskGNN — a compact graph-relational risk model for SafeShift (pure NumPy, fully auditable).

Per-interface models (the heuristic, RandomForest) score each interface from its own local feature
vector. They cannot represent integration risk that *propagates* through the architecture — an
immature, defect-prone subsystem raising the risk of interfaces several hops away. A graph neural
network can: it passes messages along the dependency graph so a node's representation absorbs its
multi-hop neighbourhood, learned end-to-end.

Design (deliberately small and dependency-free — numpy only, no torch/sklearn):
  * Node features  : [maturity, ASIL/4, log1p(fan_in), log1p(fan_out), in_cycle]   (raw, local)
  * Edge features  : [safety_related, timing_critical, supplier_boundary,
                      protocol_mismatch_risk, sqrt(signals)/6]
  * 2 message-passing layers, DIRECTED with separate mean aggregation over in- and out-neighbours:
        H' = tanh( H Ws + (Pin H) Wi + (Pout H) Wo + b ),
    where Pin = row-normalised A^T and Pout = row-normalised A are fixed per graph.
  * Edge readout MLP over [h_src, h_tgt, edge_features] -> sigmoid risk in [0, 1].
  * Trained with full-batch Adam on binary cross-entropy over interfaces, pooled across graphs.

Crucially, the GNN's node inputs are RAW local attributes (NOT any precomputed propagated value);
it must reconstruct propagation itself via message passing. Gradients are hand-derived and verified
against finite differences (see tests/test_gnn.py)."""
from __future__ import annotations

import numpy as np

from .schema import Architecture
from .graph import build_dependency_graph, structural_metrics
from .features import extract_interface_features

NODE_FEATURE_NAMES = ["maturity", "asil_over_4", "log1p_fan_in", "log1p_fan_out", "in_cycle"]
EDGE_FEATURE_NAMES = ["safety_related", "timing_critical", "supplier_boundary",
                      "protocol_mismatch_risk", "sqrt_signals_over_6"]
FN_NODE = len(NODE_FEATURE_NAMES)
FE_EDGE = len(EDGE_FEATURE_NAMES)


# --------------------------------------------------------------------------------------------------
# Graph -> tensors
# --------------------------------------------------------------------------------------------------
def _rownorm(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s


def architecture_to_tensors(arch: Architecture, labels: dict | None = None) -> dict:
    """Build the per-graph tensors the GNN consumes. `labels` maps interface_id -> 0/1 (optional)."""
    ids = [c.id for c in arch.components]
    idx = {cid: i for i, cid in enumerate(ids)}
    n = len(ids)
    sm = structural_metrics(build_dependency_graph(arch))
    comp = {c.id: c for c in arch.components}

    Xnode = np.zeros((n, FN_NODE))
    for cid in ids:
        c = comp[cid]
        Xnode[idx[cid]] = [c.maturity, c.asil_rank() / 4.0,
                           np.log1p(sm[cid]["fan_in"]), np.log1p(sm[cid]["fan_out"]),
                           sm[cid]["in_cycle"]]

    # collapsed directed adjacency for message passing (mean over unique neighbours)
    A = np.zeros((n, n))
    for itf in arch.interfaces:
        A[idx[itf.source], idx[itf.target]] = 1.0
    Pin = _rownorm(A.T)    # Pin @ H = mean over in-neighbours (predecessors)
    Pout = _rownorm(A)     # Pout @ H = mean over out-neighbours (successors)

    feats = extract_interface_features(arch)
    eids = [itf.id for itf in arch.interfaces]
    E = len(eids)
    eidx = np.zeros((E, 2), dtype=int)
    Efeat = np.zeros((E, FE_EDGE))
    y = np.zeros(E)
    for r, itf in enumerate(arch.interfaces):
        f = feats[itf.id]
        eidx[r] = (idx[itf.source], idx[itf.target])
        Efeat[r] = [f["safety_related"], f["timing_critical"], f["supplier_boundary"],
                    f["protocol_mismatch_risk"], min(np.sqrt(f["signals"]) / 6.0, 1.5)]
        if labels is not None:
            y[r] = float(labels[itf.id])
    return {"Xnode": Xnode, "Pin": Pin, "Pout": Pout, "eidx": eidx,
            "Efeat": Efeat, "y": y, "eids": eids}


# --------------------------------------------------------------------------------------------------
# Parameters / forward / backward
# --------------------------------------------------------------------------------------------------
def init_params(hidden: int = 16, readout_hidden: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def w(a, b):
        return rng.standard_normal((a, b)) * np.sqrt(2.0 / (a + b))

    H, Hr = hidden, readout_hidden
    return {
        "Ws1": w(FN_NODE, H), "Wi1": w(FN_NODE, H), "Wo1": w(FN_NODE, H), "b1": np.zeros(H),
        "Ws2": w(H, H), "Wi2": w(H, H), "Wo2": w(H, H), "b2": np.zeros(H),
        "Wr1": w(2 * H + FE_EDGE, Hr), "br1": np.zeros(Hr),
        "Wr2": w(Hr, 1)[:, 0], "br2": np.zeros(()),
    }


def _forward(p: dict, gt: dict):
    H0 = gt["Xnode"]
    Pin, Pout = gt["Pin"], gt["Pout"]

    Ain1, Aout1 = Pin @ H0, Pout @ H0
    Z1 = H0 @ p["Ws1"] + Ain1 @ p["Wi1"] + Aout1 @ p["Wo1"] + p["b1"]
    H1 = np.tanh(Z1)

    Ain2, Aout2 = Pin @ H1, Pout @ H1
    Z2 = H1 @ p["Ws2"] + Ain2 @ p["Wi2"] + Aout2 @ p["Wo2"] + p["b2"]
    H2 = np.tanh(Z2)

    src, tgt = gt["eidx"][:, 0], gt["eidx"][:, 1]
    C = np.concatenate([H2[src], H2[tgt], gt["Efeat"]], axis=1)
    R = np.tanh(C @ p["Wr1"] + p["br1"])
    logit = R @ p["Wr2"] + p["br2"]
    cache = dict(H0=H0, Ain1=Ain1, Aout1=Aout1, H1=H1, Ain2=Ain2, Aout2=Aout2,
                 H2=H2, C=C, R=R, src=src, tgt=tgt, Pin=Pin, Pout=Pout)
    return logit, cache


def _backward(p: dict, cache: dict, dlogit: np.ndarray) -> dict:
    """dlogit: dL/dlogit per edge (already scaled by 1/total_edges)."""
    H = p["b1"].shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}

    # readout
    g["Wr2"] += cache["R"].T @ dlogit
    g["br2"] += dlogit.sum()
    dR = np.outer(dlogit, p["Wr2"])
    dpreR = dR * (1.0 - cache["R"] ** 2)
    g["Wr1"] += cache["C"].T @ dpreR
    g["br1"] += dpreR.sum(axis=0)
    dC = dpreR @ p["Wr1"].T
    dH2 = np.zeros_like(cache["H2"])
    np.add.at(dH2, cache["src"], dC[:, :H])
    np.add.at(dH2, cache["tgt"], dC[:, H:2 * H])

    # layer 2
    dZ2 = dH2 * (1.0 - cache["H2"] ** 2)
    g["Ws2"] += cache["H1"].T @ dZ2
    g["Wi2"] += cache["Ain2"].T @ dZ2
    g["Wo2"] += cache["Aout2"].T @ dZ2
    g["b2"] += dZ2.sum(axis=0)
    dH1 = dZ2 @ p["Ws2"].T \
        + cache["Pin"].T @ (dZ2 @ p["Wi2"].T) \
        + cache["Pout"].T @ (dZ2 @ p["Wo2"].T)

    # layer 1
    dZ1 = dH1 * (1.0 - cache["H1"] ** 2)
    g["Ws1"] += cache["H0"].T @ dZ1
    g["Wi1"] += cache["Ain1"].T @ dZ1
    g["Wo1"] += cache["Aout1"].T @ dZ1
    g["b1"] += dZ1.sum(axis=0)
    return g


def loss_and_grads(p: dict, graphs: list[dict]):
    """Pooled BCE and gradients over a list of tensor-graphs (each must carry labels in 'y')."""
    total = max(sum(len(gt["y"]) for gt in graphs), 1)
    loss = 0.0
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    for gt in graphs:
        logit, cache = _forward(p, gt)
        loss += (np.maximum(logit, 0) - logit * gt["y"] + np.log1p(np.exp(-np.abs(logit)))).sum()
        dlogit = (1.0 / (1.0 + np.exp(-logit)) - gt["y"]) / total
        gi = _backward(p, cache, dlogit)
        for k in grads:
            grads[k] += gi[k]
    return loss / total, grads


def loss_only(p: dict, graphs: list[dict]) -> float:
    total = max(sum(len(gt["y"]) for gt in graphs), 1)
    s = 0.0
    for gt in graphs:
        logit, _ = _forward(p, gt)
        s += (np.maximum(logit, 0) - logit * gt["y"] + np.log1p(np.exp(-np.abs(logit)))).sum()
    return s / total


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
class RiskGNN:
    """A small directed graph neural network predicting per-interface integration-risk probability."""

    def __init__(self, hidden: int = 16, readout_hidden: int = 16, seed: int = 0):
        self.hidden = hidden
        self.readout_hidden = readout_hidden
        self.seed = seed
        self.params = init_params(hidden, readout_hidden, seed)
        self.history: list[float] = []

    @staticmethod
    def _to_tensors(graphs_with_labels) -> list[dict]:
        out = []
        for item in graphs_with_labels:
            if isinstance(item, dict) and "arch" in item:
                out.append(architecture_to_tensors(item["arch"], item["labels"]))
            else:
                arch, labels = item
                out.append(architecture_to_tensors(arch, labels))
        return out

    def train(self, graphs_with_labels, epochs: int = 400, lr: float = 0.02,
              weight_decay: float = 1e-3, val_graphs=None, patience: int = 50,
              beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, verbose: bool = False):
        """Full-batch AdamW (decoupled weight decay on weight matrices, not biases). If
        `val_graphs` is given, keep the parameters at the lowest validation loss (early stopping),
        which controls the over-fitting seen with an unregularised model. Epoch selection uses ONLY
        the validation split — never any test set."""
        graphs = self._to_tensors(graphs_with_labels)
        val = self._to_tensors(val_graphs) if val_graphs else None
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.history = []
        best = None
        best_val = np.inf
        wait = 0
        for t in range(1, epochs + 1):
            loss, grads = loss_and_grads(self.params, graphs)
            self.history.append(float(loss))
            for k in self.params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                mhat = m[k] / (1 - beta1 ** t)
                vhat = v[k] / (1 - beta2 ** t)
                upd = mhat / (np.sqrt(vhat) + eps)
                if k.startswith("W"):           # decoupled L2 on weights only
                    upd = upd + weight_decay * self.params[k]
                self.params[k] -= lr * upd
            if val is not None:
                vloss = loss_only(self.params, val)
                if vloss < best_val - 1e-6:
                    best_val, best, wait = vloss, {k: x.copy() for k, x in self.params.items()}, 0
                else:
                    wait += 1
                    if wait >= patience:
                        break
            if verbose and (t % 50 == 0 or t == 1):
                print(f"epoch {t:4d}  loss {loss:.4f}")
        if best is not None:
            self.params = best
        return self

    def predict(self, arch: Architecture) -> dict:
        gt = architecture_to_tensors(arch)
        logit, _ = _forward(self.params, gt)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return {eid: float(prob[r]) for r, eid in enumerate(gt["eids"])}

    def rank_interfaces(self, arch: Architecture):
        scored = list(self.predict(arch).items())
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
