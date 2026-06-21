#!/usr/bin/env python3
"""Analyse the SafeShift expert validation study.

Reads a long-format CSV (one row per interface x expert) and computes, per
architecture AND pooled across architectures:

  * Inter-expert concordance: Kendall's W + pairwise Spearman.
  * SafeShift vs expert CONSENSUS (per-interface median rating): Spearman rho
    and Kendall tau against the SafeShift score.
  * Band agreement: treating expert-majority-HIGH as the positive class and
    safeshift_band==HIGH as the prediction -> precision / recall / F1, plus
    Fleiss' kappa across experts on bands when >= 3 experts are present.
  * Top-5 overlap: Jaccard (and raw intersection count) of the SafeShift top-5
    interfaces vs the consensus top-5, per architecture.
  * A GO / PARTIAL / NO-GO verdict.

Statistical methods use only numpy, scipy.stats, and the standard library.
Small-N situations (too few raters/interfaces, constant inputs) are detected and
skipped with a clear message rather than crashing.

Usage:
    python3 expert_study.py <results.csv>

Input CSV columns (header required):
    architecture,interface,expert_id,expert_band,expert_score,
    safeshift_score,safeshift_band

    expert_band   in {HIGH, MEDIUM, LOW}
    expert_score  numeric 0-10 (may be blank -> falls back to band mapping)
    safeshift_score in [0, 1]
    safeshift_band  in {HIGH, MEDIUM, LOW}
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau, rankdata

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BAND_TO_SCORE = {"HIGH": 2.0, "MEDIUM": 1.0, "LOW": 0.0}
TOP_K = 5
MIN_RATERS_FOR_W = 2          # Kendall's W needs >= 2 raters
MIN_RATERS_FOR_FLEISS = 3     # task spec: Fleiss only when >= 3 experts
MIN_N_FOR_CORR = 3            # correlations are meaningless below this

# Verdict thresholds
GO_SPEARMAN = 0.6
PARTIAL_SPEARMAN_LO = 0.4
PARTIAL_SPEARMAN_HI = 0.6
GO_TOP5_OVERLAP = 3           # raw intersection count out of 5
GO_F1 = 0.6


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

class Row:
    """A single (interface, expert) observation."""

    __slots__ = (
        "architecture",
        "interface",
        "expert_id",
        "expert_band",
        "expert_score",
        "safeshift_score",
        "safeshift_band",
        "resolved",
    )

    def __init__(self, architecture, interface, expert_id, expert_band,
                 expert_score, safeshift_score, safeshift_band, resolved):
        self.architecture = architecture
        self.interface = interface
        self.expert_id = expert_id
        self.expert_band = expert_band
        self.expert_score = expert_score
        self.safeshift_score = safeshift_score
        self.safeshift_band = safeshift_band
        self.resolved = resolved


def _parse_float(value, label, line_no):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"line {line_no}: cannot parse {label} value {value!r} as a number"
        ) from exc


def _norm_band(value, line_no, column):
    value = (value or "").strip().upper()
    if value not in BAND_TO_SCORE:
        raise ValueError(
            f"line {line_no}: {column} value {value!r} is not one of "
            f"{sorted(BAND_TO_SCORE)}"
        )
    return value


def load_rows(path):
    """Load and validate the CSV into a list of Row objects."""
    required = {
        "architecture", "interface", "expert_id", "expert_band",
        "expert_score", "safeshift_score", "safeshift_band",
    }
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty (no header row found)")
        header = {h.strip() for h in reader.fieldnames}
        missing = required - header
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {sorted(missing)}"
            )
        for i, raw in enumerate(reader, start=2):  # line 1 is the header
            architecture = (raw.get("architecture") or "").strip()
            interface = (raw.get("interface") or "").strip()
            expert_id = (raw.get("expert_id") or "").strip()
            if not architecture or not interface or not expert_id:
                raise ValueError(
                    f"line {i}: architecture, interface and expert_id "
                    "must all be non-empty"
                )
            expert_band = _norm_band(raw.get("expert_band"), i, "expert_band")
            safeshift_band = _norm_band(
                raw.get("safeshift_band"), i, "safeshift_band")
            expert_score = _parse_float(
                raw.get("expert_score"), "expert_score", i)
            safeshift_score = _parse_float(
                raw.get("safeshift_score"), "safeshift_score", i)
            if safeshift_score is None:
                raise ValueError(f"line {i}: safeshift_score must not be blank")

            # Resolved expert rating: prefer the numeric score, else map band.
            resolved = (
                expert_score if expert_score is not None
                else BAND_TO_SCORE[expert_band]
            )
            rows.append(Row(
                architecture, interface, expert_id, expert_band,
                expert_score, safeshift_score, safeshift_band, resolved,
            ))
    if not rows:
        raise ValueError("CSV contains a header but no data rows")
    return rows


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #

class ArchData:
    """Pivoted view of one architecture's data.

    interfaces : list[str]                 ordered interface names
    experts    : list[str]                 ordered expert ids
    ratings    : ndarray (n_interfaces, n_experts)   resolved ratings, NaN gaps
    bands      : ndarray (n_interfaces, n_experts) object   expert band labels
    ss_score   : ndarray (n_interfaces,)   SafeShift score per interface
    ss_band    : ndarray (n_interfaces,) object   SafeShift band per interface
    """

    def __init__(self, name, interfaces, experts, ratings, bands,
                 ss_score, ss_band):
        self.name = name
        self.interfaces = interfaces
        self.experts = experts
        self.ratings = ratings
        self.bands = bands
        self.ss_score = ss_score
        self.ss_band = ss_band

    @property
    def n_interfaces(self):
        return len(self.interfaces)

    @property
    def n_experts(self):
        return len(self.experts)


def build_arch(name, rows):
    """Pivot the rows for a single architecture into an ArchData."""
    interfaces = sorted({r.interface for r in rows})
    experts = sorted({r.expert_id for r in rows})
    iidx = {name_: k for k, name_ in enumerate(interfaces)}
    eidx = {name_: k for k, name_ in enumerate(experts)}

    ratings = np.full((len(interfaces), len(experts)), np.nan)
    bands = np.full((len(interfaces), len(experts)), None, dtype=object)

    # SafeShift values are per-interface; verify they are consistent across the
    # repeated rows for the same interface.
    ss_score_map = {}
    ss_band_map = {}
    for r in rows:
        ii, ee = iidx[r.interface], eidx[r.expert_id]
        if not np.isnan(ratings[ii, ee]):
            raise ValueError(
                f"duplicate row for interface {r.interface!r} / expert "
                f"{r.expert_id!r} in architecture {name!r}"
            )
        ratings[ii, ee] = r.resolved
        bands[ii, ee] = r.expert_band

        if r.interface in ss_score_map:
            if abs(ss_score_map[r.interface] - r.safeshift_score) > 1e-9:
                raise ValueError(
                    f"inconsistent safeshift_score for interface "
                    f"{r.interface!r} in architecture {name!r}"
                )
            if ss_band_map[r.interface] != r.safeshift_band:
                raise ValueError(
                    f"inconsistent safeshift_band for interface "
                    f"{r.interface!r} in architecture {name!r}"
                )
        else:
            ss_score_map[r.interface] = r.safeshift_score
            ss_band_map[r.interface] = r.safeshift_band

    ss_score = np.array([ss_score_map[i] for i in interfaces], dtype=float)
    ss_band = np.array([ss_band_map[i] for i in interfaces], dtype=object)
    return ArchData(name, interfaces, experts, ratings, bands,
                    ss_score, ss_band)


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def _complete_rating_matrix(ratings):
    """Return the subset of interfaces with NO missing expert ratings.

    Kendall's W and pairwise rankings require a complete matrix; ragged rows
    are dropped (and the caller is told how many).
    """
    mask = ~np.isnan(ratings).any(axis=1)
    return ratings[mask], int(mask.sum()), int((~mask).sum())


def kendalls_w(ratings):
    """Kendall's coefficient of concordance W with tie correction.

    ratings : ndarray (n_items, m_raters), complete (no NaN).

    Returns (W, message). W is None when it cannot be computed.
    """
    n_items, m_raters = ratings.shape
    if m_raters < MIN_RATERS_FOR_W:
        return None, f"need >= {MIN_RATERS_FOR_W} raters (have {m_raters})"
    if n_items < 2:
        return None, f"need >= 2 items (have {n_items})"

    # Rank each rater's column (average ranks for ties).
    ranks = np.empty_like(ratings, dtype=float)
    tie_correction = 0.0
    for j in range(m_raters):
        col = ratings[:, j]
        ranks[:, j] = rankdata(col, method="average")
        # Tie correction term T = sum(t^3 - t) over tied groups for this rater.
        _, counts = np.unique(col, return_counts=True)
        tie_correction += np.sum(counts**3 - counts)

    rank_sums = ranks.sum(axis=1)
    mean_rank_sum = m_raters * (n_items + 1) / 2.0
    s = np.sum((rank_sums - mean_rank_sum) ** 2)

    denom = (m_raters**2 * (n_items**3 - n_items)) - (m_raters * tie_correction)
    if denom <= 0:
        return None, "denominator is zero (all raters fully tied)"
    w = 12.0 * s / denom
    return float(w), None


def pairwise_spearman(ratings, experts):
    """Pairwise Spearman rho across rater columns.

    ratings : ndarray (n_items, m_raters), complete (no NaN).
    experts : list[str] rater ids, aligned to the columns of ratings.

    Returns (mean_rho, pairs, message) where pairs is a list of
    (expert_a, expert_b, rho) tuples. mean_rho is None when not computable.
    """
    n_items, m_raters = ratings.shape
    if m_raters < 2:
        return None, [], f"need >= 2 raters (have {m_raters})"
    if n_items < MIN_N_FOR_CORR:
        return None, [], f"need >= {MIN_N_FOR_CORR} items (have {n_items})"

    pairs = []
    rhos = []
    skipped_constant = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in range(m_raters):
            for b in range(a + 1, m_raters):
                col_a, col_b = ratings[:, a], ratings[:, b]
                if np.ptp(col_a) == 0 or np.ptp(col_b) == 0:
                    skipped_constant += 1
                    pairs.append((experts[a], experts[b], None))
                    continue
                rho, _ = spearmanr(col_a, col_b)
                rho = None if np.isnan(rho) else float(rho)
                pairs.append((experts[a], experts[b], rho))
                if rho is not None:
                    rhos.append(rho)
    if not rhos:
        return None, pairs, "all rater pairs constant or undefined"
    msg = None
    if skipped_constant:
        msg = f"{skipped_constant} pair(s) skipped (constant rater)"
    return float(np.mean(rhos)), pairs, msg


def consensus_ratings(ratings):
    """Per-interface median of the resolved expert ratings (NaN-aware)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(ratings, axis=1)


def correlation_vs_consensus(consensus, ss_score):
    """Spearman rho and Kendall tau of SafeShift score vs consensus.

    Returns dict with 'spearman', 'kendall', and 'message'. Values are None
    when not computable.
    """
    out = {"spearman": None, "kendall": None, "message": None}
    valid = ~(np.isnan(consensus) | np.isnan(ss_score))
    c, s = consensus[valid], ss_score[valid]
    n = len(c)
    if n < MIN_N_FOR_CORR:
        out["message"] = f"need >= {MIN_N_FOR_CORR} interfaces (have {n})"
        return out
    if np.ptp(c) == 0 or np.ptp(s) == 0:
        out["message"] = "consensus or SafeShift score is constant"
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(s, c)
        tau, _ = kendalltau(s, c)
    out["spearman"] = None if np.isnan(rho) else float(rho)
    out["kendall"] = None if np.isnan(tau) else float(tau)
    return out


def majority_high(bands_row):
    """True when a strict majority (> 50%) of present experts rated HIGH.

    With 3 experts this means >= 2 HIGH votes. Blank cells (None) are ignored
    in both numerator and denominator.
    """
    present = [b for b in bands_row if b is not None]
    if not present:
        return False
    n_high = sum(1 for b in present if b == "HIGH")
    return n_high > (len(present) / 2.0)


def band_agreement(arch):
    """Precision / recall / F1 with expert-majority-HIGH as positive class and
    SafeShift band == HIGH as the prediction."""
    y_true = np.array(
        [majority_high(arch.bands[i]) for i in range(arch.n_interfaces)])
    y_pred = np.array([b == "HIGH" for b in arch.ss_band])

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_positive": int(np.sum(y_true)),
    }


def fleiss_kappa(arch):
    """Fleiss' kappa across experts on the 3 band categories.

    Requires >= MIN_RATERS_FOR_FLEISS experts. Returns (kappa, message);
    kappa is None when it cannot be computed.

    Fleiss assumes an equal number of raters per item. Interfaces that do not
    have ratings from every expert are dropped (with a note).
    """
    if arch.n_experts < MIN_RATERS_FOR_FLEISS:
        return None, (f"need >= {MIN_RATERS_FOR_FLEISS} experts "
                      f"(have {arch.n_experts})")

    categories = ["HIGH", "MEDIUM", "LOW"]
    cat_idx = {c: k for k, c in enumerate(categories)}

    rows_counts = []
    dropped = 0
    for i in range(arch.n_interfaces):
        present = [b for b in arch.bands[i] if b is not None]
        if len(present) != arch.n_experts:
            dropped += 1
            continue
        counts = np.zeros(len(categories))
        for b in present:
            counts[cat_idx[b]] += 1
        rows_counts.append(counts)

    if len(rows_counts) < 2:
        return None, "too few fully-rated interfaces for Fleiss' kappa"

    table = np.array(rows_counts)              # (n_items, n_categories)
    n_items, _ = table.shape
    n_raters = arch.n_experts

    # P_i: agreement per item.
    p_i = (np.sum(table**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))
    # P_e: expected agreement from category marginals.
    p_j = np.sum(table, axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j**2))

    msg = None
    if dropped:
        msg = f"{dropped} interface(s) dropped (not rated by all experts)"
    if abs(1.0 - p_e) < 1e-12:
        # No expected disagreement -> kappa undefined; report perfect/observed.
        note = "P_e == 1 (no category variance); kappa undefined"
        return None, note if msg is None else f"{msg}; {note}"
    kappa = (p_bar - p_e) / (1.0 - p_e)
    return float(kappa), msg


def top_k_overlap(arch, k=TOP_K):
    """Jaccard and raw intersection count of SafeShift top-k vs consensus top-k.

    Returns dict with 'jaccard', 'intersection', 'k_effective', 'message'.
    """
    consensus = consensus_ratings(arch.ratings)
    valid = ~np.isnan(consensus)
    n_valid = int(valid.sum())
    k_eff = min(k, n_valid, arch.n_interfaces)
    if k_eff < 1:
        return {"jaccard": None, "intersection": None,
                "k_effective": 0, "message": "no rankable interfaces"}

    # Rank descending; ties broken deterministically by interface order.
    ss_order = np.argsort(-arch.ss_score, kind="stable")[:k_eff]

    cons = consensus.copy()
    cons[~valid] = -np.inf          # push missing consensus to the bottom
    cons_order = np.argsort(-cons, kind="stable")[:k_eff]

    ss_set = {arch.interfaces[i] for i in ss_order}
    cons_set = {arch.interfaces[i] for i in cons_order}
    inter = ss_set & cons_set
    union = ss_set | cons_set
    jaccard = len(inter) / len(union) if union else None
    msg = None
    if k_eff < k:
        msg = f"only {k_eff} interface(s) rankable (k reduced from {k})"
    return {
        "jaccard": jaccard,
        "intersection": len(inter),
        "k_effective": k_eff,
        "message": msg,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(value, fmt="{:.4f}"):
    return "n/a" if value is None else fmt.format(value)


def analyse_block(arch, header):
    """Run and print every metric for one ArchData block. Returns a results
    dict used by the verdict."""
    print()
    print("=" * 70)
    print(header)
    print("=" * 70)
    print(f"  interfaces: {arch.n_interfaces}    experts: {arch.n_experts}")

    n_missing = int(np.isnan(arch.ratings).sum())
    if n_missing:
        print(f"  note: {n_missing} blank expert rating(s) present "
              "(band fallback / NaN-aware handling applied)")

    # --- Inter-expert concordance -------------------------------------- #
    print("\n  [Inter-expert concordance]")
    complete, n_complete, n_dropped = _complete_rating_matrix(arch.ratings)
    if n_dropped:
        print(f"    {n_dropped} interface(s) excluded from W / pairwise "
              "(incomplete expert coverage)")
    if n_complete >= 2 and arch.n_experts >= MIN_RATERS_FOR_W:
        w, w_msg = kendalls_w(complete)
        if w is None:
            print(f"    Kendall's W: skipped ({w_msg})")
        else:
            print(f"    Kendall's W: {w:.4f}  (over {n_complete} interfaces, "
                  f"{arch.n_experts} experts)")
    else:
        print(f"    Kendall's W: skipped (need >= {MIN_RATERS_FOR_W} experts "
              f"and >= 2 complete interfaces; have {arch.n_experts} experts, "
              f"{n_complete} complete)")

    # Experts aligned to the complete-matrix columns (column order unchanged).
    if n_complete >= MIN_N_FOR_CORR:
        mps, pairs, mps_msg = pairwise_spearman(complete, arch.experts)
    else:
        mps, pairs, mps_msg = (None, [],
                               f"need >= {MIN_N_FOR_CORR} complete interfaces "
                               f"(have {n_complete})")
    if mps is None and not pairs:
        print(f"    Pairwise Spearman: skipped ({mps_msg})")
    else:
        for ea, eb, rho in pairs:
            print(f"    Pairwise Spearman {ea} vs {eb}: {_fmt(rho)}")
        n_valid_pairs = sum(1 for _, _, r in pairs if r is not None)
        if mps is None:
            print(f"    Mean pairwise Spearman: n/a ({mps_msg})")
        else:
            extra = f"; {mps_msg}" if mps_msg else ""
            print(f"    Mean pairwise Spearman: {mps:.4f}  "
                  f"({n_valid_pairs} pair(s){extra})")

    # --- SafeShift vs consensus ---------------------------------------- #
    print("\n  [SafeShift vs expert consensus (per-interface median)]")
    consensus = consensus_ratings(arch.ratings)
    corr = correlation_vs_consensus(consensus, arch.ss_score)
    if corr["spearman"] is None and corr["kendall"] is None:
        print(f"    skipped ({corr['message']})")
    else:
        print(f"    Spearman rho: {_fmt(corr['spearman'])}")
        print(f"    Kendall tau : {_fmt(corr['kendall'])}")

    # --- Band agreement ------------------------------------------------ #
    print("\n  [Band agreement: expert-majority-HIGH vs SafeShift HIGH]")
    ba = band_agreement(arch)
    print(f"    positives (expert-majority-HIGH): {ba['n_positive']}"
          f"   TP={ba['tp']} FP={ba['fp']} FN={ba['fn']} TN={ba['tn']}")
    print(f"    precision: {ba['precision']:.4f}   "
          f"recall: {ba['recall']:.4f}   F1: {ba['f1']:.4f}")

    fk, fk_msg = fleiss_kappa(arch)
    if fk is None:
        print(f"    Fleiss' kappa: skipped ({fk_msg})")
    else:
        extra = f"; {fk_msg}" if fk_msg else ""
        print(f"    Fleiss' kappa: {fk:.4f}  ({arch.n_experts} experts{extra})")

    # --- Top-5 overlap ------------------------------------------------- #
    print("\n  [Top-5 overlap: SafeShift vs consensus]")
    overlap = top_k_overlap(arch)
    if overlap["intersection"] is None:
        print(f"    skipped ({overlap['message']})")
    else:
        k_eff = overlap["k_effective"]
        extra = f"  ({overlap['message']})" if overlap["message"] else ""
        print(f"    intersection: {overlap['intersection']}/{k_eff}   "
              f"Jaccard: {_fmt(overlap['jaccard'])}{extra}")

    return {
        "spearman": corr["spearman"],
        "f1": ba["f1"],
        "top5_intersection": overlap["intersection"],
        "top5_k": overlap["k_effective"],
    }


def verdict(per_arch_results, pooled_results):
    """Compute and print the GO / PARTIAL / NO-GO verdict.

    Gates (per task spec):
      GO      : pooled consensus Spearman >= 0.6
                AND top-5 intersection >= 3/5 in BOTH architectures
                AND band F1 (pooled) >= 0.6
      PARTIAL : pooled consensus Spearman in [0.4, 0.6)
      NO-GO   : otherwise
    """
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    pooled_spearman = pooled_results.get("spearman")
    pooled_f1 = pooled_results.get("f1")

    # Top-5 in BOTH architectures.
    arch_names = list(per_arch_results.keys())
    top5_ok = len(arch_names) >= 1
    top5_detail = []
    for name in arch_names:
        inter = per_arch_results[name]["top5_intersection"]
        ok = inter is not None and inter >= GO_TOP5_OVERLAP
        top5_ok = top5_ok and ok
        shown = "n/a" if inter is None else f"{inter}/5"
        top5_detail.append(f"{name}={shown}{' OK' if ok else ' FAIL'}")
    # Require at least two architectures for the "BOTH" condition to hold.
    if len(arch_names) < 2:
        top5_ok = False

    sp_str = _fmt(pooled_spearman)
    f1_str = _fmt(pooled_f1)
    print(f"  pooled consensus Spearman : {sp_str}  "
          f"(GO needs >= {GO_SPEARMAN})")
    print(f"  pooled band F1            : {f1_str}  (GO needs >= {GO_F1})")
    print(f"  top-5 overlap per arch    : {', '.join(top5_detail)}  "
          f"(GO needs >= {GO_TOP5_OVERLAP}/5 in BOTH)")

    sp = pooled_spearman
    f1 = pooled_f1

    go = (
        sp is not None and sp >= GO_SPEARMAN
        and top5_ok
        and f1 is not None and f1 >= GO_F1
    )
    partial = (
        sp is not None
        and PARTIAL_SPEARMAN_LO <= sp < PARTIAL_SPEARMAN_HI
    )

    if go:
        result = "GO"
    elif partial:
        result = "PARTIAL"
    else:
        result = "NO-GO"

    print(f"\n  >>> {result} <<<")
    if result == "NO-GO" and sp is not None and sp >= GO_SPEARMAN:
        print("  (Spearman clears the GO bar but another GO gate failed and "
              "Spearman is not in the PARTIAL band -> NO-GO by literal rule.)")
    return result


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(path):
    rows = load_rows(path)

    by_arch = defaultdict(list)
    for r in rows:
        by_arch[r.architecture].append(r)

    arch_names = sorted(by_arch)
    print(f"Loaded {len(rows)} observation(s) across "
          f"{len(arch_names)} architecture(s): {', '.join(arch_names)}")

    per_arch_results = {}
    for name in arch_names:
        arch = build_arch(name, by_arch[name])
        per_arch_results[name] = analyse_block(
            arch, f"ARCHITECTURE: {name}")

    # Pooled block: all rows together under a single synthetic architecture.
    # Interface names are namespaced by architecture so they don't collide.
    pooled_rows = []
    for r in rows:
        pooled_rows.append(Row(
            "POOLED", f"{r.architecture}::{r.interface}", r.expert_id,
            r.expert_band, r.expert_score, r.safeshift_score,
            r.safeshift_band, r.resolved,
        ))
    pooled_arch = build_arch("POOLED", pooled_rows)
    pooled_results = analyse_block(pooled_arch, "POOLED (all architectures)")

    verdict(per_arch_results, pooled_results)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse the SafeShift expert validation study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input CSV columns:\n"
            "  architecture, interface, expert_id, expert_band, expert_score,\n"
            "  safeshift_score, safeshift_band\n\n"
            "expert_band/safeshift_band in {HIGH, MEDIUM, LOW};\n"
            "expert_score 0-10 (blank -> band fallback HIGH=2/MEDIUM=1/LOW=0);\n"
            "safeshift_score in [0, 1]."
        ),
    )
    parser.add_argument("csv_path", help="path to the results CSV file")
    args = parser.parse_args(argv)

    try:
        run(args.csv_path)
    except FileNotFoundError:
        print(f"error: file not found: {args.csv_path}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())