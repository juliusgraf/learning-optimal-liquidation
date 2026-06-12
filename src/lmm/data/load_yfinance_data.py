"""CLI regenerating the historical mid-price CSV (ruling D13; Phase 6).

The committed ``legacy/data.csv`` IS real data: S&P 500 1-minute mid prices,
Dec 31 2025, 2:30-5:00 pm EST, normalized to 100 at session start — treated
as the FROZEN experimental input. This loader documents and reproduces the
construction from yfinance (``--normalize first=100`` default).
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-load-data",
        description="Download 1-minute mid prices via yfinance and write the "
        "normalized CSV (ruling D13).",
    )
    parser.add_argument("--symbols", nargs="+", required=True, help="ticker symbols")
    parser.add_argument("--start", required=True, help="session date, e.g. 2025-12-31")
    parser.add_argument("--session", default="14:30-17:00", help="EST time window")
    parser.add_argument(
        "--normalize",
        default="first=100",
        help="normalization rule (default: first row scaled to 100)",
    )
    parser.add_argument("--out", required=True, help="output CSV path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Phase 6 (parsed: {vars(args)})")
