"""Run-directory creation and experiment metadata dumping (fully functional).

Each run writes ``<results_root>/<experiment_name>/<run_name>/`` containing
``config_resolved.yaml``, ``seed.txt``, ``git_sha.txt``, ``metrics.csv``,
``eval/``, ``checkpoints/``, ``logs/run.log``, ``figures/``, ``tables/``. Figures and tables are always
regenerated from saved outputs by separate scripts.
"""

from __future__ import annotations

import logging
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path

from lmm.config import environment_contract
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
    """Dump complete static provenance into the run directory."""
    if cfg.experiment.master_seed != master_seed:
        raise ValueError(
            "resolved experiment.master_seed and the executed master seed disagree: "
            f"{cfg.experiment.master_seed} != {master_seed}"
        )
    save_resolved(cfg, paths.config_resolved)
    paths.seed_txt.write_text(f"{master_seed}\n")
    paths.git_sha_txt.write_text(f"{_git_sha()}\n")
    historical = cfg.midprice.historical
    if historical is not None:
        from lmm.data.historical_artifact import validate_historical_artifact

        data_manifest = validate_historical_artifact(
            historical,
            Path.cwd(),
            horizon=cfg.grid.tau_op,
        )
        (paths.run_dir / "historical_data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2, sort_keys=True) + "\n"
        )
    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "PyYAML",
        "scipy",
        "matplotlib",
        "certifi",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    (paths.run_dir / "runtime_versions.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "artifact_schema_version": cfg.experiment.artifact_schema_version,
                "environment_contract": environment_contract(cfg),
                "clearing_mechanism": cfg.auction_flow.clearing_mechanism,
                "time_unit": cfg.grid.time_unit,
                "tau_op": cfg.grid.tau_op,
                "tau_cl": cfg.grid.tau_cl,
                "ablation_label": cfg.experiment.ablation_label,
                "auction_anchor": cfg.actions.auction_anchor,
                "dqn_equal_q_tie_breaking": "first action in lexicographic grid order",
                "torch_threads": _torch_thread_counts(),
                "runtime_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "LMM_TORCH_INTRAOP_THREADS",
                        "LMM_TORCH_INTEROP_THREADS",
                        "MPLCONFIGDIR",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _torch_thread_counts() -> dict[str, int | str]:
    """Record the effective Torch pools without making Torch a hard import here."""

    try:
        import torch

        return {
            "intra_op": int(torch.get_num_threads()),
            "inter_op": int(torch.get_num_interop_threads()),
        }
    except (ImportError, RuntimeError) as exc:
        return {"unavailable": type(exc).__name__}


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
