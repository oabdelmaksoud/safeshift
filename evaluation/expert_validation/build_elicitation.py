"""Prepare the independent expert-validation study so it is one step from execution.

This builds everything needed to run the blinded expert study that `evaluation/expert_study.py`
analyses — the face-validity check the synthetic evaluation cannot provide:

  1. A BLINDED elicitation sheet per worked example: the interface list an expert ranks, with
     SafeShift's own scores WITHHELD (so the expert is not anchored).               -> blinded
  2. A separate SafeShift score KEY per example (kept by the study owner, never sent).-> key
  3. A clearly-labeled SYNTHETIC demo ratings file in the exact long format
     `expert_study.py` consumes, so the whole pipeline can be exercised end-to-end
     before any real expert is involved.                                            -> demo (FAKE)
  4. `merge_returned()` — turn a returned, filled blinded sheet (one expert) + the
     key into rows of the long-format CSV `expert_study.py` expects.

IMPORTANT: the demo ratings are RANDOM, SIMULATED, and exist ONLY to prove the analysis pipeline
runs. Expert ids are prefixed `DEMO-` and the file name carries `SYNTHETIC`. They are NOT real
expert judgements and MUST NOT be cited as validation. Real validation requires real ratings from
the vetted independent experts (see PROTOCOL.md).
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from safeshift import load_architecture, RiskModel  # noqa: E402
from safeshift.features import extract_interface_features  # noqa: E402

EX_DIR = os.path.join(HERE, "..", "..", "examples")
EXAMPLES = {
    "adas": "example_adas_architecture.yaml",
    "connected_vehicle": "example_connected_vehicle_architecture.yaml",
}
BANDS = ("HIGH", "MEDIUM", "LOW")


def _band(score: float) -> str:
    return "HIGH" if score >= 0.66 else ("MEDIUM" if score >= 0.33 else "LOW")


def _scored(arch):
    """Return {interface_id: (safeshift_score, safeshift_band)} using the transparent heuristic."""
    feats = extract_interface_features(arch)
    model = RiskModel()  # heuristic — deterministic, no training
    return {iid: (float(s), _band(float(s))) for iid, s in model.rank_interfaces(feats)}


def build_blinded_and_key(tag: str, yaml_name: str):
    arch = load_architecture(os.path.join(EX_DIR, yaml_name))
    scored = _scored(arch)

    blinded = os.path.join(HERE, f"elicitation_{tag}_BLINDED.csv")
    with open(blinded, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["architecture", "interface", "from", "to", "protocol", "signals",
                    "safety_related", "timing_critical", "your_band(HIGH/MEDIUM/LOW)",
                    "your_score_0_10(optional)"])
        for itf in arch.interfaces:   # NOTE: deliberately NOT sorted by SafeShift score (no anchor)
            w.writerow([arch.name, itf.id, itf.source, itf.target, itf.protocol, itf.signals,
                        str(itf.safety_related).lower(), str(itf.timing_critical).lower(), "", ""])

    key = os.path.join(HERE, f"safeshift_key_{tag}.csv")
    with open(key, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["architecture", "interface", "safeshift_score", "safeshift_band"])
        for iid, (sc, bd) in scored.items():
            w.writerow([arch.name, iid, f"{sc:.4f}", bd])
    return arch, scored, blinded, key


def _long_rows_from_ratings(arch, scored, expert_id, bands_by_iid, scores_by_iid=None):
    """Build long-format rows for one expert: one row per interface."""
    rows = []
    for itf in arch.interfaces:
        sc, bd = scored[itf.id]
        rows.append({
            "architecture": arch.name, "interface": itf.id, "expert_id": expert_id,
            "expert_band": bands_by_iid[itf.id],
            "expert_score": "" if scores_by_iid is None else scores_by_iid[itf.id],
            "safeshift_score": f"{sc:.4f}", "safeshift_band": bd,
        })
    return rows


LONG_COLUMNS = ["architecture", "interface", "expert_id", "expert_band", "expert_score",
                "safeshift_score", "safeshift_band"]


def build_demo(arches, seed: int = 0):
    """Write a SYNTHETIC long-format ratings file (3 fake experts) to exercise expert_study.py.
    These are NOT real judgements — random ratings loosely correlated with SafeShift, for a pipeline
    smoke test only."""
    rng = np.random.default_rng(seed)
    out = os.path.join(HERE, "demo_ratings_SYNTHETIC.csv")
    rows = []
    for arch, scored in arches:
        for e in range(3):
            bands = {}
            for itf in arch.interfaces:
                sc, _ = scored[itf.id]
                noisy = np.clip(sc + rng.normal(0, 0.18), 0, 1)  # fake expert ~ SafeShift + noise
                bands[itf.id] = _band(float(noisy))
            rows += _long_rows_from_ratings(arch, scored, f"DEMO-E{e + 1}", bands)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LONG_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out


def merge_returned(filled_blinded_csv: str, key_csv: str, expert_id: str, out_csv: str):
    """REAL-USE helper: merge one expert's returned (filled) blinded sheet with the SafeShift key
    into long-format rows ready for expert_study.py. Append-friendly across experts."""
    key = {}
    with open(key_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key[(r["architecture"], r["interface"])] = (r["safeshift_score"], r["safeshift_band"])
    new_rows = []
    with open(filled_blinded_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            arch_name, iid = r["architecture"], r["interface"]
            band = (r.get("your_band(HIGH/MEDIUM/LOW)") or "").strip().upper()
            score = (r.get("your_score_0_10(optional)") or "").strip()
            if band not in BANDS:
                raise ValueError(f"{filled_blinded_csv}: interface {iid!r} has invalid band {band!r}")
            ss_score, ss_band = key[(arch_name, iid)]
            new_rows.append({"architecture": arch_name, "interface": iid, "expert_id": expert_id,
                             "expert_band": band, "expert_score": score,
                             "safeshift_score": ss_score, "safeshift_band": ss_band})
    exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LONG_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerows(new_rows)
    return out_csv


if __name__ == "__main__":
    arches = []
    for tag, yaml_name in EXAMPLES.items():
        arch, scored, blinded, key = build_blinded_and_key(tag, yaml_name)
        arches.append((arch, scored))
        print(f"[{tag}] wrote {os.path.basename(blinded)} (blinded) and {os.path.basename(key)} (key)")
    demo = build_demo(arches)
    print(f"wrote {os.path.basename(demo)} (SYNTHETIC demo ratings — NOT real)")
