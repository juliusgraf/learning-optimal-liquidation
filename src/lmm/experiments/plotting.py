"""Shared plotting helpers for Phase 7 figures.

One matplotlib style file (``paper.mplstyle``), one colorblind-safe palette
with consistent per-policy colors/labels, a ``save_fig`` that always writes both
PDF (LaTeX) and PNG, and thin readers for the saved run-directory artifacts
(``make_figures`` / ``make_tables`` regenerate from these only — no env
stepping). ``collect_runs`` aggregates several run dirs for the cross-algorithm
figure/tables, keying the learned policy by each run's ``algo.name`` (the
records always label the learned policy ``"dqn"``, so identity MUST come from
the resolved config).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # file-output only; never interactive (headless CI)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from lmm.config import ExperimentConfig, load_config

__all__ = [
    "apply_style",
    "save_fig",
    "mark_auction",
    "POLICY_COLORS",
    "POLICY_LABELS",
    "POLICY_ORDER",
    "ALGO_ORDER",
    "read_metrics",
    "read_records",
    "read_regret",
    "read_trace",
    "read_metadata",
    "read_config",
    "RunInfo",
    "collect_runs",
]

_STYLE_PATH = Path(__file__).with_name("paper.mplstyle")

# Okabe-Ito colorblind-safe palette, one stable color per policy.
POLICY_COLORS = {
    "dqn": "#0072B2",      # blue
    "initial": "#999999",  # grey
    "as": "#E69F00",       # orange
    "twap": "#009E73",     # green
    "ddpg": "#D55E00",     # vermillion
    "td3": "#56B4E9",      # sky blue
    "sac": "#CC79A7",      # purple
}
POLICY_LABELS = {
    "dqn": "DQN",
    "initial": "Initial DQN",
    "as": "AS",
    "twap": "TWAP",
    "ddpg": "DDPG",
    "td3": "TD3",
    "sac": "SAC",
}
# Column order for eval-summary tables / distribution plots.
POLICY_ORDER = ["initial", "as", "twap", "dqn", "ddpg", "td3", "sac"]
# Learned algorithms compared in figure (f) / per-ticker tables.
ALGO_ORDER = ["dqn", "ddpg", "td3", "sac"]


def apply_style() -> None:
    """Apply the shared paper style (idempotent)."""
    plt.style.use(str(_STYLE_PATH))


def save_fig(fig, out_dir: str | Path, name: str) -> list[Path]:
    """Save ``fig`` to ``out_dir/name.{pdf,png}`` and close it. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def mark_auction(ax, tau_op: float, tau_cl: Optional[float] = None) -> None:
    """Mark the auction-open boundary (and optionally the clearing time) on a
    time-axis panel."""
    ax.axvline(tau_op, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
    if tau_cl is not None:
        ax.axvline(tau_cl, color="0.4", linestyle=":", linewidth=1.0, zorder=0)


# -- readers (saved outputs only) -------------------------------------------


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        warnings.warn(f"missing {path}", stacklevel=2)
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def read_metrics(run_dir: str | Path) -> Optional[pd.DataFrame]:
    return _read_csv(Path(run_dir) / "metrics.csv")


def read_records(run_dir: str | Path) -> Optional[pd.DataFrame]:
    return _read_csv(Path(run_dir) / "eval" / "records.csv")


def read_regret(run_dir: str | Path, benchmark: str) -> Optional[pd.DataFrame]:
    return _read_csv(Path(run_dir) / "eval" / f"regret_{benchmark}.csv")


def read_trace(run_dir: str | Path, policy: str, episode: int = 0) -> Optional[pd.DataFrame]:
    return _read_csv(Path(run_dir) / "eval" / "traces" / f"{policy}_ep{episode}.csv")


def read_metadata(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "eval" / "metadata.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def read_config(run_dir: str | Path) -> ExperimentConfig:
    return load_config(Path(run_dir) / "config_resolved.yaml")


# -- multi-run aggregation ---------------------------------------------------


@dataclass
class RunInfo:
    """One finished run directory, with its identity resolved from config."""

    run_dir: Path
    cfg: ExperimentConfig
    setting: str        # cfg.experiment.name ("synthetic_rough_heston" | "historical_sp500")
    algo: str           # cfg.algo.name ("dqn" | "ddpg" | "td3" | "sac")
    symbol: Optional[str]
    records: Optional[pd.DataFrame]
    metadata: dict
    seed: Optional[int]  # master seed from seed.txt (cross-seed aggregation key)


def collect_runs(run_dirs) -> list[RunInfo]:
    """Resolve each run dir into a :class:`RunInfo` (config + records +
    metadata). Skips dirs without a resolvable config (with a warning)."""
    runs: list[RunInfo] = []
    for rd in run_dirs:
        rd = Path(rd)
        cfg_path = rd / "config_resolved.yaml"
        if not cfg_path.exists():
            warnings.warn(f"skipping {rd}: no config_resolved.yaml", stacklevel=2)
            continue
        cfg = load_config(cfg_path)
        meta = read_metadata(rd)
        algo = cfg.algo.name if cfg.algo is not None else "unknown"
        seed_file = rd / "seed.txt"
        seed = int(seed_file.read_text().strip()) if seed_file.exists() else None
        runs.append(
            RunInfo(
                run_dir=rd,
                cfg=cfg,
                setting=cfg.experiment.name,
                algo=algo,
                symbol=meta.get("symbol"),
                records=read_records(rd),
                metadata=meta,
                seed=seed,
            )
        )
    return runs
