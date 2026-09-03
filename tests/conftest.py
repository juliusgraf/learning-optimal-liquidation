"""Session fixtures for the Phase 3 market/env tests (helpers in helpers.py)."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
import numpy as np
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
    tested. Figure/table tests receive a temporary current-schema artifact with the
    current contract and current primary outcome columns.
    """
    shutil.copytree(source, target)
    cfg = load_dqn_cfg() if algo == "dqn" else load_algo_cfg(algo)
    save_resolved(cfg, target / "config_resolved.yaml")
    (target / "seed.txt").write_text(f"{cfg.experiment.master_seed}\n")

    metadata_path = target / "eval" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text()) or {}
    metadata.update(
        master_seed=cfg.experiment.master_seed,
        environment_contract=ENVIRONMENT_CONTRACT,
        artifact_schema_version=cfg.experiment.artifact_schema_version,
        auction_anchor=cfg.actions.auction_anchor,
        checkpoint_selection={"metric": "risk_adjusted_pnl"},
    )

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
    learned_records = records[records["policy"] == algo].sort_values("episode")
    metadata.update(
        n_episodes=len(learned_records),
        policies=[algo, "initial", "as", "twap"],
        learned_policy_label=algo,
        evaluation_episode_seeds=[
            int(value) for value in learned_records["env_seed"].tolist()
        ],
    )
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    records.to_csv(records_path, index=False)

    # Test-only migration of deliberately stale pre-v12 trace fixtures. The
    # production reader remains fail-closed and requires explicit act_ell/b.
    traces_dir = target / "eval" / "traces"
    if traces_dir.exists():
        for trace_path in traces_dir.glob("*.csv"):
            trace = pd.read_csv(trace_path)
            if "act_offset" not in trace:
                continue
            trace = trace.rename(columns={"act_offset": "act_b"})
            policy = trace_path.stem.split("_ep", 1)[0]
            if policy in (algo, "initial"):
                auction = trace["phase"].eq("auction")
                if cfg.actions.auction_anchor == "indicative":
                    center_b = np.floor(
                        (trace.loc[auction, "h_cl"].astype(float)
                         - trace.loc[auction, "s_mid"].astype(float))
                        / float(cfg.grid.alpha)
                        + 0.5
                    ).astype(int)
                else:
                    center_b = pd.Series(0, index=trace.index[auction], dtype=int)
                slope_index = np.floor(
                    trace.loc[auction, "act_Ka"].astype(float)
                    / float(cfg.actions.beta)
                    + 0.5
                ).clip(0, cfg.actions.K_max).astype(int)
                trace.loc[auction, "act_Ka"] = (
                    slope_index * float(cfg.actions.beta)
                )
                ell = np.floor(
                    trace.loc[auction, "act_b"].astype(float) - center_b + 0.5
                ).clip(-cfg.actions.B_max, cfg.actions.B_max).astype(int)
                ell.loc[slope_index.eq(0)] = 0
                trace["act_ell"] = pd.Series(pd.NA, index=trace.index, dtype="Int64")
                trace.loc[auction, "act_ell"] = ell
                trace.loc[auction, "act_b"] = center_b + ell
                trace.loc[auction, "S_a"] = (
                    trace.loc[auction, "s_mid"].astype(float)
                    + float(cfg.grid.alpha) * trace.loc[auction, "act_b"].astype(float)
                )
            else:
                trace["act_ell"] = pd.NA
            trace.to_csv(trace_path, index=False)

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
