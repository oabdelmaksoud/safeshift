"""Extended, additive evaluation for SafeShift.

This module ADDS experiments without touching the frozen canonical artifacts produced by
run_eval.py (results.json / results.md / figures/roc.png / figures/feature_importance.png).
Everything here writes to new, separately-named files.

Subcommands:
  security     external attack-surface (UNECE R155/R156, ISO-SAE 21434) overlap on the
               connected-vehicle worked example: do SafeShift's integration-risk hotspots
               coincide with the externally-reachable interfaces?
  maps         risk-coloured dependency-map figures for both worked examples
  robustness   discrimination across several *independent, aligned* ground truths, to show
               the headline result is not an artifact of one synthetic risk function
  scalability  end-to-end analysis runtime vs architecture size (closes the paper's §3.5 gap)
  all          run all of the above

Usage:  python evaluation/extended.py all
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import synthetic as S
from safeshift.schema import load_architecture, Architecture, Component, Interface
from safeshift.features import extract_interface_features, INTERFACE_FEATURE_NAMES as FN
from safeshift.model import RiskModel, heuristic_score
from safeshift.graph import build_dependency_graph

EX_DIR = os.path.join(HERE, "..", "examples")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 11


def _band(s: float) -> str:
    return "HIGH" if s >= 0.66 else ("MEDIUM" if s >= 0.33 else "LOW")


# --------------------------------------------------------------------------------------
# 1) Security / external attack-surface overlap (the cyber tie-in)
# --------------------------------------------------------------------------------------
# Off-board connectivity entry points in the connected-vehicle example. These are the
# components that expose the vehicle to the outside world (cellular, V2X, OTA back end,
# consumer apps) -- the surface UNECE R155/R156 and ISO-SAE 21434 are written around.
EXTERNAL_ENTRY = {"tcu", "v2x", "ota", "ivi"}


def security() -> dict:
    arch = load_architecture(os.path.join(EX_DIR, "example_connected_vehicle_architecture.yaml"))
    g = nx.DiGraph(build_dependency_graph(arch))  # collapse multi-edges for reachability
    reachable = set(EXTERNAL_ENTRY)
    for e in EXTERNAL_ENTRY:
        if e in g:
            reachable |= nx.descendants(g, e)

    feats = extract_interface_features(arch)
    ranked = RiskModel().train().rank_interfaces(feats)  # learned mode, matches the report
    idx = {i.id: i for i in arch.interfaces}

    rows = []
    for rank, (iid, score) in enumerate(ranked, 1):
        src = idx[iid].source
        rows.append({"rank": rank, "id": iid, "from": src, "to": idx[iid].target,
                     "risk": round(float(score), 2), "band": _band(score),
                     "externally_reachable": src in reachable})

    n_if = len(rows)
    high = [r for r in rows if r["band"] == "HIGH"]
    ext_if = [r for r in rows if r["externally_reachable"]]
    top10 = rows[:10]
    obs_high_reach = sum(1 for r in high if r["externally_reachable"])
    obs_top10_reach = sum(1 for r in top10 if r["externally_reachable"])

    # --- base-rate / chance analysis -------------------------------------------------------
    # Most interfaces in a connected-vehicle design are reachable from an off-board entry point,
    # so a high *count* of reachable hotspots is expected by the base rate alone. We therefore
    # test whether the overlap EXCEEDS chance, and whether risk scores actually rank higher on
    # reachable interfaces, rather than reporting the raw overlap as if it were a finding.
    from scipy.stats import hypergeom, mannwhitneyu
    base_rate = len(ext_if) / n_if
    exp_high_reach = base_rate * len(high)
    exp_top10_reach = base_rate * len(top10)
    # P(X >= observed) drawing len(high) interfaces from n_if of which len(ext_if) are reachable
    p_high = float(hypergeom.sf(obs_high_reach - 1, n_if, len(ext_if), len(high))) if high else None
    p_top10 = float(hypergeom.sf(obs_top10_reach - 1, n_if, len(ext_if), len(top10)))
    # does risk rank higher on reachable interfaces? (Mann-Whitney, one-sided 'greater')
    risk_reach = [r["risk"] for r in rows if r["externally_reachable"]]
    risk_non = [r["risk"] for r in rows if not r["externally_reachable"]]
    if risk_reach and risk_non:
        u, p_rank = mannwhitneyu(risk_reach, risk_non, alternative="greater")
        rank_auc = float(u) / (len(risk_reach) * len(risk_non))
    else:
        p_rank, rank_auc = None, None

    res = {
        "architecture": arch.name,
        "n_components": len(arch.components),
        "n_interfaces": n_if,
        "external_entry_points": sorted(EXTERNAL_ENTRY),
        "n_externally_reachable_interfaces": len(ext_if),
        "base_rate_reachable": round(base_rate, 3),
        "n_high": len(high),
        "n_high_and_externally_reachable": obs_high_reach,
        "expected_high_reachable_by_base_rate": round(exp_high_reach, 2),
        "hypergeom_p_high_ge_observed": p_high,
        "top10_externally_reachable": obs_top10_reach,
        "expected_top10_reachable_by_base_rate": round(exp_top10_reach, 2),
        "hypergeom_p_top10_ge_observed": p_top10,
        "mean_risk_reachable": round(float(np.mean(risk_reach)), 3) if risk_reach else None,
        "mean_risk_non_reachable": round(float(np.mean(risk_non)), 3) if risk_non else None,
        "rank_test_p_reachable_gt_non": (round(float(p_rank), 3) if p_rank is not None else None),
        "rank_test_auc": (round(rank_auc, 3) if rank_auc is not None else None),
        "rows": rows,
    }

    L = ["# SafeShift — External Attack-Surface Overlap (connected-vehicle example)\n",
         f"- Architecture: {arch.name}",
         f"- Components: {res['n_components']} | Interfaces: {n_if}",
         f"- External entry points (off-board connectivity): {', '.join(sorted(EXTERNAL_ENTRY))}",
         f"- Externally-reachable interfaces: **{len(ext_if)} / {n_if}** "
         f"(base rate **{base_rate:.0%}** of all interfaces)",
         (f"- HIGH-risk & externally reachable: **{obs_high_reach} / {len(high)}** "
          f"(chance expectation **{exp_high_reach:.1f}**; hypergeometric P(>={obs_high_reach}) = "
          f"**{p_high:.2f}**)") if high else "- (no HIGH interfaces)",
         f"- Top-10 hotspots externally reachable: **{obs_top10_reach} / 10** "
         f"(chance expectation **{exp_top10_reach:.1f}**; P(>={obs_top10_reach}) = **{p_top10:.2f}**)",
         f"- Mean risk on reachable vs non-reachable interfaces: "
         f"**{res['mean_risk_reachable']} vs {res['mean_risk_non_reachable']}** "
         f"(Mann-Whitney one-sided p = **{res['rank_test_p_reachable_gt_non']}**)\n",
         "| Rank | Interface | From -> To | Risk | Band | Externally reachable (R155 surface) |",
         "|-----:|-----------|-----------|-----:|------|:--:|"]
    for r in rows:
        L.append(f"| {r['rank']} | {r['id']} | {r['from']} -> {r['to']} | {r['risk']:.2f} "
                 f"| {r['band']} | {'yes' if r['externally_reachable'] else 'no'} |")
    L.append("\n_An interface is 'externally reachable' if its source component is reachable, in the "
             "directed dependency graph, from an off-board connectivity entry point (the cyber "
             "attack-propagation surface UNECE R155/R156 and ISO/SAE 21434 govern). Because most "
             "interfaces in this connected design are reachable, the high-risk interfaces are reachable "
             "at roughly the base rate (overlap not above chance), and the two highest-risk interfaces "
             "-- raw camera/radar inputs to fusion -- are NOT externally reachable. The honest reading: "
             "a single design-time pass *enumerates* the externally-reachable surface alongside "
             "integration risk, so one analysis feeds both the integration plan and the ISO/SAE 21434 "
             "TARA -- but it does not selectively concentrate risk on that surface in this hand-designed "
             "example._")
    _write("results_security", res, "\n".join(L))
    return res


# --------------------------------------------------------------------------------------
# 2) Dependency-map figures (risk-coloured)
# --------------------------------------------------------------------------------------
_BAND_COLOR = {"HIGH": "#C0392B", "MEDIUM": "#E08E0B", "LOW": "#2E8B57"}


def _draw_map(arch: Architecture, out_png: str, title: str) -> None:
    feats = extract_interface_features(arch)
    model = RiskModel().train()
    scores = {iid: model.predict(f) for iid, f in feats.items()}
    g = build_dependency_graph(arch)
    simple = nx.DiGraph(g)
    pos = nx.spring_layout(simple, seed=SEED, k=1.7, iterations=250)
    plt.figure(figsize=(9, 6.5))
    nx.draw_networkx_nodes(simple, pos, node_size=850, node_color="#D6E2F0",
                           edgecolors="#33495E", linewidths=1.2)
    nx.draw_networkx_labels(simple, pos, font_size=7.5, font_color="#10202E")
    idx = {i.id: i for i in arch.interfaces}
    for iid, sc in scores.items():
        itf = idx[iid]
        band = _band(sc)
        nx.draw_networkx_edges(g, pos, edgelist=[(itf.source, itf.target)],
                               edge_color=_BAND_COLOR[band],
                               width=1.0 + 3.4 * float(sc), alpha=0.85,
                               arrowsize=14, connectionstyle="arc3,rad=0.08",
                               node_size=850)
    handles = [plt.Line2D([0], [0], color=c, lw=3,
               label=f"{b} integration risk") for b, c in _BAND_COLOR.items()]
    plt.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
    plt.title(title, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def maps() -> dict:
    a1 = load_architecture(os.path.join(EX_DIR, "example_adas_architecture.yaml"))
    a2 = load_architecture(os.path.join(EX_DIR, "example_connected_vehicle_architecture.yaml"))
    p1 = os.path.join(FIG, "dependency_map_adas.png")
    p2 = os.path.join(FIG, "dependency_map_connected_vehicle.png")
    _draw_map(a1, p1, "SafeShift dependency map — reference ADAS architecture\n"
                      "(edge colour = integration-risk band, width is proportional to risk)")
    _draw_map(a2, p2, "SafeShift dependency map — connected-vehicle / SDV architecture\n"
                      "(edge colour = integration-risk band, width is proportional to risk)")
    return {"figures": [os.path.basename(p1), os.path.basename(p2)]}


# --------------------------------------------------------------------------------------
# 3) Robustness across independent, ALIGNED ground truths
# --------------------------------------------------------------------------------------
def _f(X):
    return {name: X[:, i] for i, name in enumerate(S.FEATURES)}


def _gt_interaction(X):  # the reference generator already used in the paper (synthetic.py)
    return S.true_logit(X)


def _gt_linear(X):       # purely additive, no interaction terms
    f = _f(X)
    return (-3.1 + 1.2 * f["safety_related"] + 0.8 * f["timing_critical"]
            + 0.7 * f["supplier_boundary"] + 1.4 * f["protocol_mismatch_risk"]
            + 0.25 * f["max_asil_rank"] + 0.9 * f["tgt_in_cycle"]
            + 0.03 * f["src_fan_out"] + 0.04 * f["tgt_fan_in"]
            + 0.02 * f["signals"] - 0.9 * f["min_maturity"])


def _gt_threshold(X):    # rule/tree-like margins rather than a smooth sum
    f = _f(X)
    z = np.full(X.shape[0], -2.4)
    z += 2.6 * ((f["safety_related"] > 0) &
                ((f["tgt_in_cycle"] > 0) |
                 ((f["supplier_boundary"] > 0) & (f["protocol_mismatch_risk"] >= 0.7))))
    z += 2.1 * ((f["max_asil_rank"] >= 3) & (f["min_maturity"] < 0.4))
    z += 1.4 * ((f["protocol_mismatch_risk"] >= 0.8) & (f["supplier_boundary"] > 0))
    z += 0.9 * (f["signals"] > 22)
    return z


def _gt_structure(X):    # structure-emphasis but safety-aware (a realistic risk model is
    f = _f(X)            # never safety-blind), so it stays aligned with the engineering factors
    return (-3.6 + 1.6 * f["tgt_in_cycle"] + 0.12 * f["tgt_fan_in"] + 0.11 * f["src_fan_out"]
            + 0.04 * f["signals"] + 1.4 * f["safety_related"] + 0.7 * f["timing_critical"]
            + 0.7 * f["protocol_mismatch_risk"]
            + 0.9 * f["supplier_boundary"] * f["protocol_mismatch_risk"]
            - 0.6 * f["min_maturity"])


_GTS = {"interaction (reference)": _gt_interaction, "linear-additive": _gt_linear,
        "threshold / rule-like": _gt_threshold, "structure-emphasis": _gt_structure}


def _gt_confounder(X, hidden, beta):  # off-feature latent driver: part of the risk comes from a
    f = _f(X)                          # hidden factor the models never observe (team experience,
    feat = (1.0 * f["safety_related"] + 0.8 * f["protocol_mismatch_risk"]   # tooling, schedule...),
            + 0.6 * f["supplier_boundary"] + 0.5 * f["timing_critical"]      # so recovery is bounded
            - 0.5 * f["min_maturity"])                                       # below the aligned ceiling.
    return -2.4 + feat + beta * hidden


def _fit_eval(X, y, rng):
    """Train heuristic/LogReg/RF on a 70% split and return held-out ROC-AUC for each + random."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    heur = np.array([heuristic_score({FN[i]: r[i] for i in range(len(FN))}) for r in Xte])
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED)\
        .fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    rnd = rng.random(len(yte))
    return {"positive_rate": round(float(y.mean()), 3),
            "Random": round(float(roc_auc_score(yte, rnd)), 3),
            "Heuristic": round(float(roc_auc_score(yte, heur)), 3),
            "LogReg": round(float(roc_auc_score(yte, lr)), 3),
            "RandomForest": round(float(roc_auc_score(yte, rf)), 3)}


def robustness() -> dict:
    rng = np.random.default_rng(SEED)
    X = S.sample_features(8000, np.random.default_rng(SEED))
    out = {}
    # (a) Four ALIGNED generators: each is a different functional form (linear, interaction,
    #     threshold, structure) over the SAME features with the SAME signs as the scorer. They
    #     test robustness to functional form/coefficients -- NOT to feature directions -- so they
    #     cannot, by construction, detect sign-alignment circularity.
    for gt_name, gt in _GTS.items():
        p = 1.0 / (1.0 + np.exp(-gt(X)))
        y = (np.random.default_rng(7).random(len(p)) < p).astype(int)
        out[gt_name] = _fit_eval(X, y, rng)
    # (b) One OFF-FEATURE generator: ~half the systematic risk comes from a latent factor the
    #     models never see. This is the genuinely harder, more independent test; performance drops
    #     toward (but stays above) chance, bounding what the synthetic study can claim when real
    #     risk has drivers outside the feature set.
    hidden = np.random.default_rng(101).standard_normal(len(X))
    feat_only = _gt_confounder(X, np.zeros(len(X)), 0.0)   # logit with no hidden contribution
    beta = float(np.std(feat_only))                        # hidden ~ half the systematic variance
    p = 1.0 / (1.0 + np.exp(-_gt_confounder(X, hidden, beta)))
    y = (np.random.default_rng(7).random(len(p)) < p).astype(int)
    out["off-feature latent driver"] = _fit_eval(X, y, rng)

    L = ["# SafeShift — Robustness across alternative ground truths\n",
         "The first four rows are *aligned* latent risk functions: each is a different functional "
         "form (linear, interaction, threshold, structure-emphasis) over the SAME engineering "
         "features with the SAME signs as SafeShift's scorer. They test robustness to functional "
         "form and coefficients -- not to the choice of feature directions -- so informed models are "
         "expected to do well, and these rows do NOT establish independence from the scorer's "
         "assumptions. The final row, 'off-feature latent driver', is the genuinely harder test: "
         "about half the systematic risk comes from a hidden factor the models never observe, so "
         "performance drops toward chance, bounding what the synthetic study can claim.\n",
         "| Ground truth | pos. rate | Random | Heuristic | LogReg | RandomForest |",
         "|--------------|----------:|-------:|----------:|-------:|-------------:|"]
    for gt, m in out.items():
        L.append(f"| {gt} | {m['positive_rate']:.3f} | {m['Random']:.3f} | {m['Heuristic']:.3f} "
                 f"| {m['LogReg']:.3f} | {m['RandomForest']:.3f} |")
    _write("results_robustness", out, "\n".join(L))
    return out


# --------------------------------------------------------------------------------------
# 4) Scalability: analysis runtime vs architecture size
# --------------------------------------------------------------------------------------
def _synth_arch(n_comp: int, seed: int, back_edge_p: float = 0.03,
                cyclic: bool = False) -> Architecture:
    """Layered, sparse, mostly feed-forward architecture with back-edges (realistic E/E sparsity).
    Defaults (back_edge_p=0.03, cyclic=False) reproduce the near-acyclic sweep exactly. Set
    cyclic=True with a higher back_edge_p to inject real directed cycles (back-edges to any earlier
    layer) for the cyclic stress test."""
    rng = np.random.default_rng(seed)
    n_layers = max(4, int(np.sqrt(n_comp)))
    layer = {i: int(rng.integers(0, n_layers)) for i in range(n_comp)}
    comps = [Component(id=f"c{i}", name=f"c{i}", kind="ecu",
                       supplier=f"S{int(rng.integers(0, 8))}",
                       asil=["QM", "A", "B", "C", "D"][int(rng.integers(0, 5))],
                       maturity=float(round(rng.random(), 2))) for i in range(n_comp)]
    itfs, eid = [], 0
    protos = ["CAN", "CAN-FD", "Ethernet", "FlexRay", "LIN"]
    for i in range(n_comp):
        for _ in range(int(rng.integers(1, 4))):  # ~1-3 outgoing edges -> sparse (~2.5N)
            nxt = [t for t in range(n_comp) if layer[t] > layer[i]]
            if not nxt:
                continue
            t = int(rng.choice(nxt))
            itfs.append(Interface(id=f"e{eid}", source=f"c{i}", target=f"c{t}",
                                  protocol=protos[int(rng.integers(0, len(protos)))],
                                  signals=int(rng.integers(1, 40)),
                                  safety_related=bool(rng.integers(0, 2)),
                                  timing_critical=bool(rng.integers(0, 2))))
            eid += 1
    # back-edges introduce directed cycles. Default: a few short edges to the immediately
    # previous layer (bounded). cyclic=True: edges to any earlier layer at rate back_edge_p,
    # producing genuine multi-node cycles for the stress test.
    for i in range(n_comp):
        if rng.random() < back_edge_p:
            prev = ([t for t in range(n_comp) if layer[t] < layer[i]] if cyclic
                    else [t for t in range(n_comp) if layer[t] == layer[i] - 1])
            if prev:
                t = int(rng.choice(prev))
                itfs.append(Interface(id=f"e{eid}", source=f"c{i}", target=f"c{t}",
                                      protocol="CAN", signals=4))
                eid += 1
    return Architecture(name=f"synthetic-{n_comp}", components=comps, interfaces=itfs)


def scalability(sizes=(20, 50, 100, 200, 500), repeats: int = 3) -> dict:
    rows = []
    for n in sizes:
        arch = _synth_arch(n, seed=SEED + n)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            feats = extract_interface_features(arch)        # graph build + structural metrics
            _ = [heuristic_score(f) for f in feats.values()]  # scoring
            ts.append(time.perf_counter() - t0)
        rows.append({"components": n, "interfaces": len(arch.interfaces),
                     "median_seconds": round(float(np.median(ts)), 4)})
    res = {"rows": rows, "repeats": repeats}
    # Cyclic stress test: largest size but dense back-edges to any earlier layer -> genuine
    # multi-node directed cycles. With SCC-based membership (O(V+E)) this stays fast, empirically
    # retiring the former "bounding cycle enumeration is future work" caveat.
    big = _synth_arch(500, seed=SEED + 500, back_edge_p=0.15, cyclic=True)
    gbig = nx.DiGraph(build_dependency_graph(big))
    cyc_nodes = sum(len(s) for s in nx.strongly_connected_components(gbig) if len(s) > 1)
    tsc = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        feats = extract_interface_features(big)
        _ = [heuristic_score(f) for f in feats.values()]
        tsc.append(time.perf_counter() - t0)
    res["cyclic_stress"] = {"components": 500, "interfaces": len(big.interfaces),
                            "cycle_member_nodes": int(cyc_nodes),
                            "median_seconds": round(float(np.median(tsc)), 4)}
    plt.figure(figsize=(5.2, 4))
    xs = [r["components"] for r in rows]
    ys = [r["median_seconds"] for r in rows]
    plt.plot(xs, ys, "o-", color="#2E5FA3")
    plt.xlabel("Components"); plt.ylabel("Median analysis time (s)")
    plt.title("SafeShift analysis runtime vs architecture size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "scalability.png"), dpi=150); plt.close()
    cs = res["cyclic_stress"]
    L = ["# SafeShift — Scalability (analysis runtime vs size)\n",
         f"Median of {repeats} runs; sparse layered architectures (interfaces ~ 2.5x components). "
         "The main sweep uses near-acyclic graphs (short back-edges); the cyclic stress row below "
         "uses dense back-edges to any earlier layer to force genuine directed cycles.\n",
         "| Components | Interfaces | Median analysis time (s) |",
         "|-----------:|-----------:|-------------------------:|"]
    for r in rows:
        L.append(f"| {r['components']} | {r['interfaces']} | {r['median_seconds']:.4f} |")
    L.append(f"\n**Cyclic stress (SCC-based cycle membership):** {cs['components']} components, "
             f"{cs['interfaces']} interfaces, {cs['cycle_member_nodes']} nodes on directed cycles -- "
             f"analysed in **{cs['median_seconds']:.4f} s**. Because membership uses "
             f"strongly-connected components (O(V+E)) rather than cycle enumeration, dense cyclic "
             f"graphs stay fast.\n")
    L.append("Figure: `figures/scalability.png`.")
    _write("results_scalability", res, "\n".join(L))
    return res


def _write(stem: str, obj: dict, md: str) -> None:
    with open(os.path.join(HERE, f"{stem}.json"), "w") as fh:
        json.dump(obj, fh, indent=2)
    with open(os.path.join(HERE, f"{stem}.md"), "w") as fh:
        fh.write(md)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("security", "all"):
        r = security()
        print(f"[security] top-10 externally reachable: {r['top10_externally_reachable']}/10; "
              f"HIGH & external: {r['n_high_and_externally_reachable']}/{r['n_high']}")
    if cmd in ("maps", "all"):
        print("[maps]", maps()["figures"])
    if cmd in ("robustness", "all"):
        rb = robustness()
        print("[robustness] heuristic AUC by GT:",
              {k: v["Heuristic"] for k, v in rb.items()})
    if cmd in ("scalability", "all"):
        sc = scalability()
        print("[scalability]", [(r["components"], r["median_seconds"]) for r in sc["rows"]])
    print("Done. New artifacts written alongside run_eval.py outputs (frozen files untouched).")
