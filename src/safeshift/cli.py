"""Command-line interface for SafeShift.

Usage:
    python -m safeshift analyze examples/example_adas_architecture.yaml --train --out report.md
"""
from __future__ import annotations
import argparse
import sys
from .schema import load_architecture
from .model import RiskModel
from .report import generate_report
from . import __version__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="safeshift",
                                description="Shift-left integration-risk prediction for "
                                            "automotive software architectures.")
    p.add_argument("--version", action="version", version=f"SafeShift {__version__}")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze", help="Analyze an architecture file and print a risk report.")
    a.add_argument("path", help="Path to architecture .json/.yaml/.yml")
    a.add_argument("--train", action="store_true",
                help="Train the optional ML model on synthetic data (else use heuristic).")
    a.add_argument("--top", type=int, default=10, help="Number of hotspots to list.")
    a.add_argument("--out", default=None, help="Write the Markdown report to this path.")

    args = p.parse_args(argv)
    if args.cmd != "analyze":
        p.print_help()
        return 1

    arch = load_architecture(args.path)
    model = RiskModel()
    if args.train:
        model.train()
    report = generate_report(arch, model, top=args.top)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Wrote report to {args.out} (model mode: {model.mode}).")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
