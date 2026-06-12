"""PRegret(T) from saved per-episode records (Phase 4).

PRegret(T) = sum_{e=1}^{E} (V_0^{pi_benchmark}(x_{0,e}) - V_0^{pi_e}(x_{0,e})),
T = (m+2)E, estimated per episode with common random numbers (same env seed
for the learned policy and the benchmark within an episode) — paper
`sec:learning` regret definition (the only part of Sec. 4 that is kept, D9).
Reads the per-episode records written by train.py / evaluate.py.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-regret",
        description="Compute PRegret(T) curves from saved per-episode records.",
    )
    parser.add_argument("--run-dir", required=True, help="training run directory")
    parser.add_argument("--benchmark", default="as", choices=["as", "twap"], help="pi_benchmark")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Phase 4 (parsed: run_dir={args.run_dir})")
