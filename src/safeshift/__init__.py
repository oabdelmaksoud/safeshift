"""SafeShift: open, vendor-neutral shift-left risk prediction for automotive software architectures.

SafeShift analyzes a description of a vehicle's software/E-E architecture and predicts where
integration and architectural defects are most likely to arise, so they can be addressed during
virtual design rather than during physical prototyping. The goal is to bring "shift-left"
validation to the software-defined vehicle in an open, reusable form.
"""
__version__ = "0.4.0"

from .schema import Architecture, Component, Interface, load_architecture
from .graph import build_dependency_graph
from .features import extract_interface_features, extract_component_features
from .model import RiskModel
from .gnn import RiskGNN
from .report import generate_report

__all__ = [
    "Architecture", "Component", "Interface", "load_architecture",
    "build_dependency_graph", "extract_interface_features",
    "extract_component_features", "RiskModel", "RiskGNN", "generate_report", "__version__",
]
