"""Regenerate all figures from saved run outputs (Phase 4+).

Figures are ALWAYS regenerated from results/<...>/metrics.csv and eval/
records — never produced inside training code (engineering conventions).
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-make-figures",
        description="Regenerate figures from saved run outputs.",
    )
    parser.add_argument("--run-dir", required=True, nargs="+", help="run directory/ies")
    parser.add_argument("--out", default=None, help="output dir (default: <run>/figures)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Phase 4 (parsed: run_dir={args.run_dir})")
