"""Run-directory creation and experiment metadata dumping (fully functional).

Every run writes ``results/<experiment_name>/<run_name>/`` containing
``config_resolved.yaml``, ``seed.txt``, ``git_sha.txt``, ``metrics.csv``,
``eval/``, ``checkpoints/``, ``logs/run.log``, ``figures/``, ``tables/``
(engineering conventions, CLAUDE.md). Figures and tables are always
regenerated from saved outputs by separate scripts.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lmm.config import ExperimentConfig, save_resolved

__all__ = ["RunPaths", "create_run_dir", "write_run_metadata", "get_run_logger"]


@dataclass(frozen=True)
class RunPaths:
    """Canonical layout of one run directory."""

    run_dir: Path
    checkpoints: Path
    eval: Path
    figures: Path
    tables: Path
    logs: Path
    metrics_csv: Path

    @property
    def config_resolved(self) -> Path:
        return self.run_dir / "config_resolved.yaml"

    @property
    def seed_txt(self) -> Path:
        return self.run_dir / "seed.txt"

    @property
    def git_sha_txt(self) -> Path:
        return self.run_dir / "git_sha.txt"

    @property
    def run_log(self) -> Path:
        return self.logs / "run.log"


def create_run_dir(results_root: str | Path, experiment_name: str, run_name: str) -> RunPaths:
    """Create ``<results_root>/<experiment_name>/<run_name>/`` and subdirs."""
    run_dir = Path(results_root) / experiment_name / run_name
    paths = RunPaths(
        run_dir=run_dir,
        checkpoints=run_dir / "checkpoints",
        eval=run_dir / "eval",
        figures=run_dir / "figures",
        tables=run_dir / "tables",
        logs=run_dir / "logs",
        metrics_csv=run_dir / "metrics.csv",
    )
    for d in (paths.run_dir, paths.checkpoints, paths.eval, paths.figures, paths.tables, paths.logs):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def _git_sha() -> str:
    """Current commit SHA (+ '-dirty' if the tree has changes); 'unknown' outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_metadata(paths: RunPaths, cfg: ExperimentConfig, master_seed: int) -> None:
    """Dump config_resolved.yaml, seed.txt and git_sha.txt into the run dir."""
    save_resolved(cfg, paths.config_resolved)
    paths.seed_txt.write_text(f"{master_seed}\n")
    paths.git_sha_txt.write_text(f"{_git_sha()}\n")


def get_run_logger(paths: RunPaths, name: str = "lmm") -> logging.Logger:
    """Logger writing to ``logs/run.log`` and the console (idempotent per run dir)."""
    logger = logging.getLogger(f"{name}.{paths.run_dir}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        fh = logging.FileHandler(paths.run_log)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
    return logger
