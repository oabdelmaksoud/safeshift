"""Open input schema for describing an automotive software / E-E architecture.

The schema is intentionally simple and vendor-neutral (JSON or YAML). It captures the
minimum needed to reason about integration risk: components (e.g., ECUs, software modules),
their attributes (supplier, ASIL safety level, maturity), and the interfaces/signals that
connect them (protocol, direction, safety relevance).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import json
import os

ASIL_LEVELS = {"QM": 0, "A": 1, "B": 2, "C": 3, "D": 4}


@dataclass
class Component:
    id: str
    name: str = ""
    kind: str = "ecu"            # ecu | software_module | sensor | domain_controller | gateway
    supplier: str = "unknown"
    asil: str = "QM"             # QM, A, B, C, D
    maturity: float = 0.5        # 0..1 (0 = new/unproven, 1 = mature/reused)

    def asil_rank(self) -> int:
        return ASIL_LEVELS.get(str(self.asil).upper().replace("ASIL", "").strip(), 0)


@dataclass
class Interface:
    id: str
    source: str                  # component id
    target: str                  # component id
    protocol: str = "CAN"        # CAN | CAN-FD | Ethernet | FlexRay | LIN | SPI | internal
    signals: int = 1             # number of signals carried
    safety_related: bool = False
    timing_critical: bool = False


@dataclass
class Architecture:
    name: str = "unnamed-architecture"
    components: list[Component] = field(default_factory=list)
    interfaces: list[Interface] = field(default_factory=list)

    def component_index(self) -> dict[str, Component]:
        return {c.id: c for c in self.components}

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty = valid)."""
        problems: list[str] = []
        ids = {c.id for c in self.components}
        if len(ids) != len(self.components):
            problems.append("Duplicate component ids detected.")
        for itf in self.interfaces:
            if itf.source not in ids:
                problems.append(f"Interface {itf.id}: unknown source '{itf.source}'.")
            if itf.target not in ids:
                problems.append(f"Interface {itf.id}: unknown target '{itf.target}'.")
        return problems


def _coerce(arch_dict: dict[str, Any]) -> Architecture:
    comps = [Component(**c) for c in arch_dict.get("components", [])]
    itfs = [Interface(**i) for i in arch_dict.get("interfaces", [])]
    return Architecture(name=arch_dict.get("name", "unnamed-architecture"),
                        components=comps, interfaces=itfs)


def load_architecture(path: str) -> Architecture:
    """Load an architecture description from a .json, .yaml or .yml file."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            import yaml  # optional dependency
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    arch = _coerce(data)
    problems = arch.validate()
    if problems:
        raise ValueError("Invalid architecture:\n  - " + "\n  - ".join(problems))
    return arch
