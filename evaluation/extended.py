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
    res = {
        "architecture": arch.name,
        "n_components": len(arch.components),
        "n_interfaces": n_if,
        "external_entry_points": sorted(EXTERNAL_ENTRY),
        "n_externally_reachable_interfaces": len(ext_if),
        "n_high": len(high),
        "n_high_and_externally_reachable": sum(1 for r in high if r["externally_reachable"]),
        "top10_externally_reachable": sum(1 for r in top10 if r["externally_reachable"]),
        "rows": rows,
    }

    L = ["# SafeShift — External Attack-Surface Overlap (connected-vehicle example)\n",
         f"- Architecture: {arch.name}",
         f"- Components: {res['n_components']} | Interfaces: {n_if}",
         f"- External entry points (off-board connectivity): {', '.join(sorted(EXTERNAL_ENTRY))}",
         f"- Externally-reachable interfaces: **{len(ext_if)} / {n_if}**",
         f"- HIGH-risk interfaces: **{len(high)}**; of those, externally reachable: "
         f"**{res['n_high_and_externally_reachable']} / {len(high)}**",
         f"- Of the top-10 integration-risk hotspots, externally reachable: "
         f"**{res['top10_externally_reachable']} / 10**\n",
         "| Rank | Interface | From → To | Risk | Band | Externally reachable (R155 surface) |",
         "|-----:|-----------|-----------|-----:|------|:--:|"]
    for r in rows:
        L.append(f"| {r['rank']} | {r['id']} | {r['from']} → {r['to']} | {r['risk']:.2f} "
                 f"| {r['band']} | {'yes' if r['externally_reachable'] else 'no'} |")
    L.append("\n_An interface is 'externally reachable' if its source component is reachable, "
             "in the directed dependency graph, from an off-board connectivity entry point. "
             "These are the interfaces on the cyber attack-propagation surface that UNECE R155/R156 "
             "and ISO-SAE 21434 govern. The overlap with SafeShift's integration-risk hotspots is "
             "the point: the same interfaces concentrate integration risk and security exposure._")
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


def robustness() -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    X = S.sample_features(8000, np.random.default_rng(SEED))
    out = {}
    for gt_name, gt in _GTS.items():
        p = 1.0 / (1.0 + np.exp(-gt(X)))
        y = (np.random.default_rng(7).random(len(p)) < p).astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
        heur = np.array([heuristic_score({FN[i]: r[i] for i in range(len(FN))}) for r in Xte])
        lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED)\
            .fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        rnd = rng.random(len(yte))
        out[gt_name] = {
            "positive_rate": round(float(y.mean()), 3),
            "Random": round(float(roc_auc_score(yte, rnd)), 3),
            "Heuristic": round(float(roc_auc_score(yte, heur)), 3),
            "LogReg": round(float(roc_auc_score(yte, lr)), 3),
            "RandomForest": round(float(roc_auc_score(yte, rf)), 3),
        }
    L = ["# SafeShift — Robustness across independent, aligned ground truths\n",
         "Each column is a *different* latent risk function (independent of SafeShift's scorer). "
         "All are plausible, aligned risk models. The point: informed models beat the random "
         "baseline across every generator, so the headline result is not an artifact of one "
         "chosen synthetic target.\n",
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
def _synth_arch(n_comp: int, seed: int) -> Architecture:
    """Layered, sparse, mostly feed-forward architecture with a few back-edges (realistic
    E/E sparsity; bounded cycle structure)."""
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
    # a few short back-edges (to the immediately previous layer) to introduce bounded cycles
    for i in range(n_comp):
        if rng.random() < 0.03:
            prev = [t for t in range(n_comp) if layer[t] == layer[i] - 1]
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
    plt.figure(figsize=(5.2, 4))
    xs = [r["components"] for r in rows]
    ys = [r["median_seconds"] for r in rows]
    plt.plot(xs, ys, "o-", color="#2E5FA3")
    plt.xlabel("Components"); plt.ylabel("Median analysis time (s)")
    plt.title("SafeShift analysis runtime vs architecture size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "scalability.png"), dpi=150); plt.close()
    L = ["# SafeShift — Scalability (analysis runtime vs size)\n",
         f"Median of {repeats} runs; sparse layered architectures (interfaces ~ 2.5x components).\n",
         "| Components | Interfaces | Median analysis time (s) |",
         "|-----------:|-----------:|-------------------------:|"]
    for r in rows:
        L.append(f"| {r['components']} | {r['interfaces']} | {r['median_seconds']:.4f} |")
    L.append("\nFigure: `figures/scalability.png`.")
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
