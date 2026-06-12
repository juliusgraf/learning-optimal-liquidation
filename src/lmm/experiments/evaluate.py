"""Frozen-policy evaluation with common random numbers (Phase 4).

Evaluates a checkpointed policy and the benchmarks on the SAME seed set
(CRN: identical env seed for learned policy and benchmark within an episode;
fixes AUDIT N10), under ONE reward definition for all policies (AUDIT C.4).
Reported returns are UNDISCOUNTED episode sums (stated in metadata).
Per-episode records are written to eval/ for regret.py and make_*.py.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from lmm.config import add_config_cli

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-evaluate",
        description="Evaluate a frozen policy vs benchmarks with CRN.",
    )
    add_config_cli(parser)
    parser.add_argument("--run-dir", required=True, help="training run directory")
    parser.add_argument("--checkpoint", default="final", help="checkpoint name")
    parser.add_argument("--n-episodes", type=int, default=100, help="evaluation episodes")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Phase 4 (parsed: run_dir={args.run_dir})")
