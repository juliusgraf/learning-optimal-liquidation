"""Shared plotting helpers for Phase 7 figures.

One matplotlib style file (``paper.mplstyle``), one colorblind-safe palette
with consistent per-policy colors/labels, a ``save_fig`` that always writes both
PDF (LaTeX) and PNG, and thin readers for the saved run-directory artifacts
(``make_figures`` / ``make_tables`` regenerate from these only — no env
stepping). ``collect_runs`` aggregates several run dirs for the cross-algorithm
figure/tables, keying the learned policy by each run's ``algo.name`` (the
records label the learned policy with the resolved ``algo.name``).
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

from lmm.agents.base import ENVIRONMENT_CONTRACT
from lmm.config import (
    ACTIVE_ARTIFACT_SCHEMA_VERSION,
    ExperimentConfig,
    load_config,
)

__all__ = [
    "apply_style",
    "save_fig",
    "mark_auction",
    "POLICY_COLORS",
    "POLICY_LABELS",
    "POLICY_ORDER",
    "ALGO_ORDER",
    "HISTORICAL_SETTING",
    "is_historical_setting",
    "read_metrics",
    "read_records",
    "read_policy_difference",
    "read_trace",
    "read_metadata",
    "read_config",
    "RunInfo",
    "collect_runs",
    "validate_evaluation_records",
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
    # The saved ``initial`` policy is the untrained instance of whichever
    # learned algorithm owns the run, not necessarily DQN (and never NFQ).
    "initial": "Initial policy",
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
HISTORICAL_SETTING = "historical_sp500_midquotes"


def is_historical_setting(setting: str) -> bool:
    """Return whether a run uses the publication true-midquote setting."""
    return setting == HISTORICAL_SETTING


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


def read_policy_difference(run_dir: str | Path, benchmark: str) -> Optional[pd.DataFrame]:
    return _read_csv(Path(run_dir) / "eval" / f"policy_difference_{benchmark}.csv")


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
    setting: str        # cfg.experiment.name (synthetic or true-midquote historical)
    algo: str           # cfg.algo.name ("dqn" | "ddpg" | "td3" | "sac")
    symbol: Optional[str]
    records: Optional[pd.DataFrame]
    metadata: dict
    seed: int  # mandatory master seed from seed.txt (cross-seed aggregation key)


def validate_evaluation_records(
    run_dir: str | Path,
    cfg: ExperimentConfig,
    metadata: dict,
    records: pd.DataFrame | None,
) -> None:
    """Fail closed unless records.csv is the complete metadata-declared CRN matrix."""
    run_dir = Path(run_dir)
    if records is None:
        raise ValueError(f"{run_dir}: eval/records.csv is missing or empty")
    if cfg.algo is None:
        raise ValueError(f"{run_dir}: evaluated run has no configured learned algorithm")

    required_metadata = (
        "n_episodes",
        "policies",
        "learned_policy_label",
        "evaluation_episode_seeds",
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(
            f"{run_dir}: eval metadata is missing completeness fields {missing}"
        )

    try:
        n_episodes = int(metadata["n_episodes"])
        policies = [str(value) for value in metadata["policies"]]
        evaluation_seeds = [int(value) for value in metadata["evaluation_episode_seeds"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{run_dir}: malformed evaluation completeness metadata") from exc
    if n_episodes <= 0:
        raise ValueError(f"{run_dir}: metadata.n_episodes must be positive")
    if len(policies) != len(set(policies)):
        raise ValueError(f"{run_dir}: metadata.policies contains duplicates")
    expected_policies = {cfg.algo.name, "initial", "as", "twap"}
    if set(policies) != expected_policies:
        raise ValueError(
            f"{run_dir}: metadata.policies must equal {sorted(expected_policies)}; "
            f"got {sorted(policies)}"
        )
    if metadata["learned_policy_label"] != cfg.algo.name:
        raise ValueError(
            f"{run_dir}: metadata.learned_policy_label disagrees with configured algo"
        )
    if len(evaluation_seeds) != n_episodes:
        raise ValueError(
            f"{run_dir}: metadata evaluation seed count {len(evaluation_seeds)} "
            f"does not equal n_episodes={n_episodes}"
        )

    required_columns = {"policy", "episode", "env_seed"}
    missing_columns = required_columns.difference(records.columns)
    if missing_columns:
        raise ValueError(
            f"{run_dir}: records.csv is missing columns {sorted(missing_columns)}"
        )
    if records[list(required_columns)].isna().any().any():
        raise ValueError(f"{run_dir}: records.csv has missing policy/episode/env_seed values")

    observed_pairs: list[tuple[str, int]] = []
    observed_seeds: list[int] = []
    try:
        for row in records[["policy", "episode", "env_seed"]].itertuples(index=False):
            episode_float = float(row.episode)
            seed_float = float(row.env_seed)
            if not episode_float.is_integer() or not seed_float.is_integer():
                raise ValueError("non-integral episode or env_seed")
            observed_pairs.append((str(row.policy), int(episode_float)))
            observed_seeds.append(int(seed_float))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{run_dir}: records.csv contains malformed episode/env_seed values"
        ) from exc

    expected_pairs = {
        (policy, episode)
        for policy in policies
        for episode in range(n_episodes)
    }
    if len(observed_pairs) != len(set(observed_pairs)):
        raise ValueError(f"{run_dir}: records.csv contains duplicate policy/episode rows")
    if set(observed_pairs) != expected_pairs:
        missing_pairs = expected_pairs.difference(observed_pairs)
        extra_pairs = set(observed_pairs).difference(expected_pairs)
        raise ValueError(
            f"{run_dir}: records.csv is not the complete metadata-declared matrix "
            f"(missing={len(missing_pairs)}, extra={len(extra_pairs)})"
        )
    for (policy, episode), observed_seed in zip(observed_pairs, observed_seeds):
        expected_seed = evaluation_seeds[episode]
        if observed_seed != expected_seed:
            raise ValueError(
                f"{run_dir}: CRN seed mismatch for policy={policy!r}, "
                f"episode={episode}: records={observed_seed}, metadata={expected_seed}"
            )


def collect_runs(run_dirs) -> list[RunInfo]:
    """Resolve each run dir into a :class:`RunInfo` (config + records +
    metadata). Skips dirs without a resolvable config (with a warning)."""
    runs: list[RunInfo] = []
    identities: dict[tuple[str, Optional[str], str, int], Path] = {}
    for rd in run_dirs:
        rd = Path(rd)
        cfg_path = rd / "config_resolved.yaml"
        if not cfg_path.exists():
            warnings.warn(f"skipping {rd}: no config_resolved.yaml", stacklevel=2)
            continue
        cfg = load_config(cfg_path)
        meta = read_metadata(rd)
        if meta.get("environment_contract") != ENVIRONMENT_CONTRACT:
            raise ValueError(
                f"{rd}: result artifact does not match {ENVIRONMENT_CONTRACT!r}; "
                "old result files are not accepted by the revised pipeline"
            )
        metadata_schema = meta.get("artifact_schema_version")
        if metadata_schema is None or int(metadata_schema) != ACTIVE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"{rd}: eval/metadata.yaml artifact_schema_version must equal "
                f"the active schema {ACTIVE_ARTIFACT_SCHEMA_VERSION}; got "
                f"{metadata_schema!r}"
            )
        if int(metadata_schema) != cfg.experiment.artifact_schema_version:
            raise ValueError(
                f"{rd}: eval/metadata.yaml artifact_schema_version "
                f"({metadata_schema}) disagrees with config_resolved.yaml "
                f"({cfg.experiment.artifact_schema_version})"
            )
        algo = cfg.algo.name if cfg.algo is not None else "unknown"
        seed_file = rd / "seed.txt"
        if not seed_file.exists():
            raise ValueError(f"{rd}: result artifact is missing seed.txt")
        seed = int(seed_file.read_text().strip())
        if seed != cfg.experiment.master_seed:
            raise ValueError(
                f"{rd}: seed.txt ({seed}) disagrees with "
                f"config_resolved.yaml experiment.master_seed "
                f"({cfg.experiment.master_seed})"
            )
        metadata_seed = meta.get("master_seed")
        if metadata_seed is not None and int(metadata_seed) != seed:
            raise ValueError(
                f"{rd}: eval/metadata.yaml master_seed ({metadata_seed}) "
                f"disagrees with seed.txt ({seed})"
            )
        symbol = meta.get("symbol")
        if cfg.experiment.name == HISTORICAL_SETTING:
            configured_symbols = (
                cfg.midprice.historical.symbols
                if cfg.midprice.historical is not None
                else ()
            )
            if symbol not in configured_symbols:
                raise ValueError(
                    f"{rd}: historical result metadata must identify one configured "
                    f"symbol; got {symbol!r}"
                )
        elif symbol is not None:
            raise ValueError(
                f"{rd}: non-historical result metadata must have symbol=null; "
                f"got {symbol!r}"
            )
        identity = (cfg.experiment.name, symbol, algo, seed)
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(
                "duplicate run identity "
                f"(setting={identity[0]!r}, symbol={identity[1]!r}, "
                f"algo={identity[2]!r}, seed={identity[3]}): "
                f"{previous} and {rd}"
            )
        identities[identity] = rd
        records = read_records(rd)
        validate_evaluation_records(rd, cfg, meta, records)
        runs.append(
            RunInfo(
                run_dir=rd,
                cfg=cfg,
                setting=cfg.experiment.name,
                algo=algo,
                symbol=symbol,
                records=records,
                metadata=meta,
                seed=seed,
            )
        )
    return runs
