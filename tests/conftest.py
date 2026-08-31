"""Session fixtures for the Phase 3 market/env tests (helpers in helpers.py)."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")  # headless figure tests (Phase 7)

from helpers import load_algo_cfg, load_dqn_cfg, load_historical_cfg, load_synthetic_cfg
from lmm.agents.base import ENVIRONMENT_CONTRACT
from lmm.config import save_resolved

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _upgrade_artifact_fixture(source: Path, target: Path, algo: str) -> Path:
    """Copy legacy-shaped sample numbers into the revised artifact envelope.

    The committed directories remain deliberately stale so rejection can be
    tested.  Figure/table tests receive a temporary schema-2 artifact with the
    current contract and current primary outcome columns.
    """
    shutil.copytree(source, target)
    cfg = load_dqn_cfg() if algo == "dqn" else load_algo_cfg(algo)
    save_resolved(cfg, target / "config_resolved.yaml")

    metadata_path = target / "eval" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text()) or {}
    metadata.update(
        environment_contract=ENVIRONMENT_CONTRACT,
        artifact_schema_version=cfg.experiment.artifact_schema_version,
        checkpoint_selection={"metric": "risk_adjusted_pnl"},
    )
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    records_path = target / "eval" / "records.csv"
    records = pd.read_csv(records_path)
    if algo != "dqn":
        records.loc[records["policy"] == "dqn", "policy"] = algo
    inventory_penalty = 0.5 * records["I_final"].astype(float) ** 2
    records["pnl"] = records["return_undisc"].astype(float) - 10_000.0
    records["risk_adjusted_pnl"] = records["pnl"] - inventory_penalty
    records["economic_objective"] = records["risk_adjusted_pnl"]
    records["liquidation_pnl_gross"] = records["pnl"]
    records["cancel_cost"] = 0.0
    records["inventory_penalty"] = inventory_penalty
    records["pnl_per_initial_notional"] = records["pnl"] / 10_000.0
    records["risk_adjusted_pnl_per_initial_notional"] = (
        records["risk_adjusted_pnl"] / 10_000.0
    )
    records["pnl_bps"] = 10_000.0 * records["pnl_per_initial_notional"]
    records["risk_adjusted_pnl_bps"] = (
        10_000.0 * records["risk_adjusted_pnl_per_initial_notional"]
    )
    records.to_csv(records_path, index=False)

    metrics_path = target / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics["pnl"] = metrics["return_undisc"].astype(float) - 10_000.0
        metrics["risk_adjusted_pnl"] = (
            metrics["pnl"] - 0.5 * metrics["I_final"].astype(float) ** 2
        )
        metrics["eval_pnl_mean"] = metrics["eval_return_mean"] - 10_000.0
        metrics["eval_risk_adjusted_pnl_mean"] = metrics["eval_pnl_mean"]
        metrics.to_csv(metrics_path, index=False)

    learned = records[records["policy"] == algo]
    for benchmark in ("as", "twap"):
        bench = records[records["policy"] == benchmark]
        merged = learned.merge(bench, on="episode", suffixes=("_p", "_b"))
        difference = merged["risk_adjusted_pnl_p"] - merged["risk_adjusted_pnl_b"]
        pd.DataFrame(
            {
                "episode": merged["episode"],
                "env_seed": merged["env_seed_p"],
                "benchmark_value": merged["risk_adjusted_pnl_b"],
                "policy_value": merged["risk_adjusted_pnl_p"],
                "policy_minus_benchmark": difference,
                "cumulative_policy_minus_benchmark": difference.cumsum(),
            }
        ).to_csv(target / "eval" / f"policy_difference_{benchmark}.csv", index=False)
    return target


@pytest.fixture
def fixture_run_dir(tmp_path) -> Path:
    return _upgrade_artifact_fixture(FIXTURES / "run_dir", tmp_path / "run_dqn", "dqn")


@pytest.fixture
def fixture_run_dir_ddpg(tmp_path) -> Path:
    return _upgrade_artifact_fixture(
        FIXTURES / "run_dir_ddpg", tmp_path / "run_ddpg", "ddpg"
    )


@pytest.fixture(scope="session")
def synthetic_cfg():
    return load_synthetic_cfg()


@pytest.fixture(scope="session")
def historical_cfg():
    return load_historical_cfg()


@pytest.fixture(scope="session")
def crn_synthetic_cfg():
    """Synthetic config with the conditional auction draws disabled (see
    helpers.py docstring) — bit-identical streams across an injection pair."""
    return load_synthetic_cfg("auction_flow.p2=0.0", "auction_flow.p4=0.0")
