"""Generate a human-readable risk report (Markdown) from scored interfaces."""
from __future__ import annotations
from .schema import Architecture
from .features import extract_interface_features
from .model import RiskModel


def _band(score: float) -> str:
    return "HIGH" if score >= 0.66 else ("MEDIUM" if score >= 0.33 else "LOW")


def generate_report(arch: Architecture, model: RiskModel | None = None, top: int = 10) -> str:
    model = model or RiskModel()
    feats = extract_interface_features(arch)
    ranked = model.rank_interfaces(feats)
    idx = {i.id: i for i in arch.interfaces}
    lines = []
    lines.append(f"# SafeShift Risk Report — {arch.name}")
    lines.append("")
    lines.append(f"- Components: {len(arch.components)}  |  Interfaces: {len(arch.interfaces)}")
    lines.append(f"- Model mode: **{model.mode}**")
    n_high = sum(1 for _, s in ranked if s >= 0.66)
    lines.append(f"- Interfaces flagged HIGH risk: **{n_high}**")
    lines.append("")
    lines.append("## Ranked integration-risk hotspots")
    lines.append("")
    lines.append("| Rank | Interface | From → To | Protocol | Risk | Band |")
    lines.append("|-----:|-----------|-----------|----------|-----:|------|")
    for rank, (iid, score) in enumerate(ranked[:top], 1):
        itf = idx[iid]
        lines.append(f"| {rank} | {iid} | {itf.source} → {itf.target} | {itf.protocol} "
                    f"| {score:.2f} | {_band(score)} |")
    lines.append("")
    lines.append("## Why these were flagged")
    lines.append("")
    for iid, score in ranked[:min(5, top)]:
        f = feats[iid]
        reasons = []
        if f["safety_related"]: reasons.append("safety-related")
        if f["timing_critical"]: reasons.append("timing-critical")
        if f["supplier_boundary"]: reasons.append("crosses a supplier boundary")
        if f["tgt_in_cycle"]: reasons.append("target sits in a dependency cycle")
        if f["protocol_mismatch_risk"] >= 0.7: reasons.append("higher-complexity protocol")
        if f["min_maturity"] < 0.4: reasons.append("involves an immature component")
        if f["max_asil_rank"] >= 3: reasons.append("high ASIL safety level")
        lines.append(f"- **{iid}** (risk {score:.2f}): " +
                    (", ".join(reasons) if reasons else "structural exposure in the graph") + ".")
    lines.append("")
    lines.append("_Scores are relative indicators of where integration review should be focused "
                "first; they are produced from the architecture description and (in learned mode) "
                "a synthetic training set. They are decision-support signals, not guarantees._")
    return "\n".join(lines)
