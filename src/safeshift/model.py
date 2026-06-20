"""Risk model for predicting integration-defect likelihood per interface.

Two modes, by design:

* Heuristic (always available): a transparent weighted score over the engineering features.
  It needs no training data and is fully explainable -- important for safety contexts where
  black-box models are viewed with caution.
* Learned (optional, scikit-learn): a RandomForest trained on a synthetic generator that encodes
  the same risk relationships plus noise. This demonstrates the ML pathway while staying honest:
  it is trained on representative synthetic data, not proprietary program data.

The learned model falls back to the heuristic if scikit-learn is unavailable.
"""
from __future__ import annotations
import numpy as np
from .features import INTERFACE_FEATURE_NAMES

# transparent weights for the heuristic (sum used then squashed to 0..1)
_W = {
    "signals": 0.010, "safety_related": 1.6, "timing_critical": 1.0,
    "supplier_boundary": 0.7, "protocol_mismatch_risk": 1.2, "src_fan_out": 0.05,
    "tgt_fan_in": 0.05, "tgt_in_cycle": 1.0, "max_asil_rank": 0.20, "min_maturity": -1.0,
}
_BIAS = -3.0


def _vec(feat: dict[str, float]) -> np.ndarray:
    return np.array([feat[k] for k in INTERFACE_FEATURE_NAMES], dtype=float)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def heuristic_score(feat: dict[str, float]) -> float:
    s = _BIAS + sum(_W[k] * feat[k] for k in INTERFACE_FEATURE_NAMES)
    return float(_sigmoid(s))


def generate_synthetic(n: int = 4000, seed: int = 7):
    """Generate a synthetic training set whose labels follow the same risk logic plus noise."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        feat = {
            "signals": float(rng.integers(1, 40)),
            "safety_related": float(rng.integers(0, 2)),
            "timing_critical": float(rng.integers(0, 2)),
            "supplier_boundary": float(rng.integers(0, 2)),
            "protocol_mismatch_risk": float(rng.choice([0.0, 0.3, 0.5, 0.7, 0.8])),
            "src_fan_out": float(rng.integers(0, 12)),
            "tgt_fan_in": float(rng.integers(0, 12)),
            "tgt_in_cycle": float(rng.integers(0, 2)),
            "max_asil_rank": float(rng.integers(0, 5)),
            "min_maturity": float(rng.random()),
        }
        p = heuristic_score(feat)
        label = 1 if rng.random() < p else 0  # noisy label from the risk probability
        X.append(_vec(feat)); y.append(label)
    return np.array(X), np.array(y)


class RiskModel:
    def __init__(self) -> None:
        self._clf = None
        self.mode = "heuristic"

    def train(self, n: int = 4000, seed: int = 7) -> "RiskModel":
        try:
            from sklearn.ensemble import RandomForestClassifier
        except Exception:
            self.mode = "heuristic"
            return self
        X, y = generate_synthetic(n, seed)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
        clf.fit(X, y)
        self._clf = clf
        self.mode = "learned"
        return self

    def predict(self, feat: dict[str, float]) -> float:
        if self._clf is not None:
            proba = self._clf.predict_proba(_vec(feat).reshape(1, -1))[0]
            classes = list(self._clf.classes_)
            return float(proba[classes.index(1)]) if 1 in classes else 0.0
        return heuristic_score(feat)

    def rank_interfaces(self, interface_features: dict[str, dict[str, float]]):
        scored = [(iid, self.predict(f)) for iid, f in interface_features.items()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
