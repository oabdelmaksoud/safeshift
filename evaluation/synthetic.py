"""Independent ground-truth generator for evaluating SafeShift.

To avoid circular evaluation, the *labels* here are NOT produced by SafeShift's own linear
heuristic. They come from a separate latent risk function with non-linear interactions
(e.g., safety x immaturity, supplier-boundary x protocol complexity, cycle x ASIL). Models are
then judged on how well they recover this independent target. This lets us fairly compare the
transparent linear heuristic, logistic regression, and a random forest.
"""
from __future__ import annotations
import numpy as np

# feature order must match safeshift.features.INTERFACE_FEATURE_NAMES
FEATURES = ["signals", "safety_related", "timing_critical", "supplier_boundary",
            "protocol_mismatch_risk", "src_fan_out", "tgt_fan_in", "tgt_in_cycle",
            "max_asil_rank", "min_maturity"]

FEATURE_GROUPS = {
    "safety/ASIL": ["safety_related", "timing_critical", "max_asil_rank"],
    "integration": ["supplier_boundary", "protocol_mismatch_risk", "signals"],
    "structural": ["src_fan_out", "tgt_fan_in", "tgt_in_cycle"],
    "maturity": ["min_maturity"],
}


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def sample_features(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((n, len(FEATURES)))
    X[:, 0] = rng.integers(1, 40, n)                       # signals
    X[:, 1] = rng.integers(0, 2, n)                        # safety_related
    X[:, 2] = rng.integers(0, 2, n)                        # timing_critical
    X[:, 3] = rng.integers(0, 2, n)                        # supplier_boundary
    X[:, 4] = rng.choice([0.0, 0.3, 0.5, 0.7, 0.8], n)     # protocol_mismatch_risk
    X[:, 5] = rng.integers(0, 12, n)                       # src_fan_out
    X[:, 6] = rng.integers(0, 12, n)                       # tgt_fan_in
    X[:, 7] = rng.integers(0, 2, n)                        # tgt_in_cycle
    X[:, 8] = rng.integers(0, 5, n)                        # max_asil_rank
    X[:, 9] = rng.random(n)                                # min_maturity
    return X


def true_logit(X: np.ndarray) -> np.ndarray:
    f = {name: X[:, i] for i, name in enumerate(FEATURES)}
    z = (-4.7
         + 1.4 * f["safety_related"]
         + 0.9 * f["timing_critical"]
         + 1.6 * f["safety_related"] * (1.0 - f["min_maturity"])          # immature safety (interaction)
         + 1.6 * f["supplier_boundary"] * f["protocol_mismatch_risk"]     # cross-supplier complex protocol
         + 0.9 * f["tgt_in_cycle"] * (1.0 + 0.25 * f["max_asil_rank"])    # cycle amplified by ASIL
         + 0.22 * np.sqrt(f["signals"])                                   # non-linear signal load
         + 0.15 * f["max_asil_rank"]
         + 0.04 * f["src_fan_out"] + 0.05 * f["tgt_fan_in"])
    return z


def make_dataset(n: int = 8000, seed: int = 11, noise: float = 1.0):
    """Return (X, y). `noise` scales label stochasticity (1.0 = default Bernoulli draw)."""
    rng = np.random.default_rng(seed)
    X = sample_features(n, rng)
    p = _sigmoid(true_logit(X))
    # temperature-style noise: higher noise -> labels closer to coin flips
    p_noisy = _sigmoid(true_logit(X) / max(noise, 1e-6))
    y = (rng.random(n) < p_noisy).astype(int)
    return X, y
