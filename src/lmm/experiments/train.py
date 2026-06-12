"""Train any agent on any setting from config (Phase 4).

Writes results/<experiment_name>/<run_name>/ with config_resolved.yaml,
seed.txt, git_sha.txt, metrics.csv (per-episode), checkpoints/, logs/run.log
(engineering conventions, CLAUDE.md). Figures/tables are produced separately
by make_figures.py / make_tables.py from these saved outputs.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from lmm.config import add_config_cli

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-train",
        description="Train an agent (configs select setting and algorithm).",
    )
    add_config_cli(parser)
    parser.add_argument("--run-name", required=True, help="results subdirectory name")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.master_seed")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"Phase 4 (parsed: configs={args.config})")
