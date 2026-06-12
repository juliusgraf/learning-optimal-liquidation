"""Regenerate all tables from saved run outputs (Phase 4+).

Tables state in their metadata that reported returns are UNDISCOUNTED
episode sums (CLAUDE.md objective conventions).
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-make-tables",
        description="Regenerate tables from saved run outputs.",
    )
    parser.add_argument("--run-dir", required=True, nargs="+", help="run directory/ies")
    parser.add_argument("--out", default=None, help="output dir (default: <run>/tables)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Phase 4 (parsed: run_dir={args.run_dir})")
