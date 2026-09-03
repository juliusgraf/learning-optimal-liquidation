"""Benchmark/environment integration and common-random-number contracts."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import yaml

from helpers import drive_to_auction, load_dqn_cfg, new_env
from lmm.agents.base import ENVIRONMENT_CONTRACT
from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.experiments import evaluate as evaluate_mod
from lmm.experiments import policy_differences as differences_mod
from lmm.experiments import train as train_mod
from lmm.rl.loops import run_episode


SMALL_OVERRIDES = (
    "benchmark.as_n_samples=500",
)


@pytest.fixture(scope="module")
def cfg():
    return load_dqn_cfg(*SMALL_OVERRIDES)


def make_as_agent(cfg, env):
    agent = ASBenchmarkAgent(cfg)
    agent.bind(env)
    agent.calibrate(
        rng_k=np.random.default_rng(1),
    )
    return agent


@pytest.mark.parametrize("name", ["as", "twap"])
def test_benchmarks_complete_real_environment_episode(cfg, name):
    env = new_env(cfg)
    agent = make_as_agent(cfg, env) if name == "as" else TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    result = run_episode(env, agent, 11, chi=cfg.rl.chi, train=False)
    assert result.n_steps == result.n_clob_steps + cfg.grid.h
    assert np.isfinite(result.risk_adjusted_pnl)
    assert result.self_trade_count == 0


def test_as_delta_table_matches_closed_form(cfg):
    env = new_env(cfg)
    agent = make_as_agent(cfg, env)
    A, k, T, alpha = agent.A, agent.k, agent.T_as, cfg.grid.alpha
    for t, q in ((0, 1), (10, 3), (T - 1, 7), (T, 2)):
        x = (A / math.e) * (T - t)
        values = [
            sum(x**j / math.factorial(j) for j in range(qq + 1))
            for qq in (q - 1, q)
        ]
        expected = (1.0 / (k * alpha)) * (1.0 + math.log(values[1] / values[0]))
        assert agent.delta_ticks[t, q] == pytest.approx(expected, rel=1e-12)


def test_as_k_uses_order_size_tail_exponent_not_price_tick(cfg, monkeypatch):
    agent = ASBenchmarkAgent(cfg)
    monkeypatch.setattr(agent, "_estimate_K_hat", lambda rng: 3.0)
    calibration = agent.calibrate(
        rng_k=np.random.default_rng(1),
    )
    assert set(calibration) == {"A", "k"}
    assert not hasattr(agent, "sigma")
    assert calibration["A"] == pytest.approx(
        cfg.clob_flow.lambda0 / cfg.clob_flow.gamma_m
    )
    assert calibration["k"] == pytest.approx(cfg.clob_flow.gamma_m * 3.0)
    assert calibration["k"] != pytest.approx(cfg.grid.alpha * 3.0)


def test_as_persistence_contains_only_policy_relevant_calibration(cfg, tmp_path):
    env = new_env(cfg)
    agent = make_as_agent(cfg, env)
    path = tmp_path / "as_calibration.npz"
    agent.save(path)

    with np.load(path) as saved:
        assert set(saved.files) == {"A", "k", "delta_ticks"}

    restored = ASBenchmarkAgent(cfg)
    restored.load(path)
    assert restored.A == pytest.approx(agent.A)
    assert restored.k == pytest.approx(agent.k)
    np.testing.assert_array_equal(restored.delta_ticks, agent.delta_ticks)


def test_benchmark_without_clob_fills_uses_H_fallback_and_submits_once(cfg):
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    drive_to_auction(env, seed=5)
    assert agent._exec_prices == []
    action = agent._auction_action()
    assert not hasattr(action, "offset")
    assert action.reference_price == pytest.approx(env.h_cl)
    assert action.quantity_cap == pytest.approx(env.inventory)
    assert agent._auction_action().K_a == 0.0


def test_presampled_clob_tape_is_identical_across_different_policies(cfg):
    twap_env, noop_env = new_env(cfg), new_env(cfg)
    twap = TWAPBenchmarkAgent(cfg)
    twap.bind(twap_env)
    twap.start_episode(0)

    # Reset both with the same environment seed before either policy acts.
    twap_env.reset(seed=31337)
    noop_env.reset(seed=31337)
    a = twap_env._generator._realization
    b = noop_env._generator._realization
    assert a is not None and b is not None
    for name in (
        "buy_arrival_times",
        "sell_arrival_times",
        "buy_volumes",
        "sell_volumes",
        "ask_tops",
        "bid_tops",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.grid.all_times, b.grid.all_times)


EXPECTED_FILES = (
    "config_resolved.yaml",
    "seed.txt",
    "git_sha.txt",
    "metrics.csv",
    "feature_normalizer.yaml",
    "logs/run.log",
    "checkpoints/initial.pt",
    "checkpoints/initial_validation.yaml",
    "checkpoints/best_mature.pt",
    "checkpoints/best_mature_selection.yaml",
    "checkpoints/best.pt",
    "checkpoints/final.pt",
)


def _train(tmp_path, run_name: str) -> str:
    argv = [
        "--config", "configs/base.yaml",
        "--config", "configs/synthetic_rough_heston.yaml",
        "--config", "configs/algo/dqn.yaml",
        "--run-name", run_name,
        "-o", f"experiment.results_root={tmp_path}",
        "-o", "experiment.episodes=3",
        "-o", "rl.normalizer_fit_episodes=1",
        "-o", "rl.validation_frequency_episodes=1",
        "-o", "rl.validation_size=1",
        "-o", "rl.validation_patience_evals=10",
        "-o", "rl.checkpoint_min_clob_updates=0",
        "-o", "rl.checkpoint_min_auction_updates=0",
        "-o", "rl.checkpoint_require_initial_improvement=false",
        "-o", "rl.test_size=2",
        "-o", "algo.hyperparams.checkpoint_interval_episodes=2",
        "-o", "algo.hyperparams.min_buffer=200",
        "-o", "algo.hyperparams.min_buffer_clob=200",
        "-o", "algo.hyperparams.min_buffer_auction=90",
        "-o", "algo.hyperparams.batch_size=32",
        "-o", "benchmark.as_n_samples=500",
    ]
    assert train_mod.main(argv) == 0
    return str(tmp_path / "synthetic_rough_heston" / run_name)


@pytest.mark.slow
def test_optional_initial_improvement_gate_rejects_worse_mature_policy(
    tmp_path, monkeypatch
):
    scores = iter((10.0, -5.0, -5.0))

    def fixed_eval(*_args, **_kwargs):
        score = next(scores)
        return {
            "return_mean": score,
            "pnl_mean": score,
            "risk_adjusted_pnl_mean": score,
            "checkpoint_score": score,
        }

    monkeypatch.setattr(train_mod, "_run_eval", fixed_eval)
    argv = [
        "--config", "configs/base.yaml",
        "--config", "configs/synthetic_rough_heston.yaml",
        "--config", "configs/algo/dqn.yaml",
        "--run-name", "selection_failure",
        "-o", f"experiment.results_root={tmp_path}",
        "-o", "experiment.episodes=1",
        "-o", "rl.normalizer_fit_episodes=1",
        "-o", "rl.validation_frequency_episodes=1",
        "-o", "rl.validation_size=1",
        "-o", "rl.validation_patience_evals=10",
        "-o", "rl.checkpoint_min_clob_updates=0",
        "-o", "rl.checkpoint_min_auction_updates=0",
        "-o", "rl.checkpoint_require_initial_improvement=true",
        "-o", "algo.hyperparams.checkpoint_interval_episodes=10",
    ]
    with pytest.raises(RuntimeError, match="initial-policy improvement gate"):
        train_mod.main(argv)
    checkpoints = (
        tmp_path / "synthetic_rough_heston" / "selection_failure" / "checkpoints"
    )
    assert not (checkpoints / "best.pt").exists()
    assert (checkpoints / "best_mature.pt").exists()
    assert (checkpoints / "selection_failure.yaml").exists()


@pytest.mark.slow
def test_default_selection_reports_best_mature_policy_even_below_initial_diagnostic(
    tmp_path, monkeypatch
):
    scores = iter((10.0, -5.0))

    def fixed_eval(*_args, **_kwargs):
        score = next(scores)
        return {
            "return_mean": score,
            "pnl_mean": score,
            "risk_adjusted_pnl_mean": score,
            "checkpoint_score": score,
        }

    monkeypatch.setattr(train_mod, "_run_eval", fixed_eval)
    argv = [
        "--config", "configs/base.yaml",
        "--config", "configs/synthetic_rough_heston.yaml",
        "--config", "configs/algo/dqn.yaml",
        "--run-name", "diagnostic_not_gate",
        "-o", f"experiment.results_root={tmp_path}",
        "-o", "experiment.episodes=1",
        "-o", "rl.normalizer_fit_episodes=1",
        "-o", "rl.validation_frequency_episodes=1",
        "-o", "rl.validation_size=1",
        "-o", "rl.validation_patience_evals=10",
        "-o", "rl.checkpoint_min_clob_updates=0",
        "-o", "rl.checkpoint_min_auction_updates=0",
        "-o", "algo.hyperparams.checkpoint_interval_episodes=10",
    ]
    assert train_mod.main(argv) == 0
    checkpoints = (
        tmp_path / "synthetic_rough_heston" / "diagnostic_not_gate" / "checkpoints"
    )
    selection = yaml.safe_load((checkpoints / "best_selection.yaml").read_text())
    assert (checkpoints / "best.pt").exists()
    assert selection["value"] == -5.0
    assert not selection["economic_safety"]["require_improvement_over_initial"]
    assert not selection["economic_safety"]["candidate_beats_initial"]
    assert selection["economic_safety"]["reportable"]
    assert not (checkpoints / "selection_failure.yaml").exists()


@pytest.mark.slow
def test_short_end_to_end_pipeline_uses_revised_artifacts(tmp_path):
    run = _train(tmp_path, "acceptance")
    for rel in EXPECTED_FILES:
        assert (tmp_path / "synthetic_rough_heston" / "acceptance" / rel).exists(), rel

    assert evaluate_mod.main(["--run-dir", run, "--n-episodes", "2"]) == 0
    rows = list(csv.DictReader(open(f"{run}/eval/records.csv")))
    expected_policies = {"dqn", *evaluate_mod.POLICIES}
    assert len(rows) == 2 * len(expected_policies)
    assert {row["policy"] for row in rows} == expected_policies
    assert all(np.isfinite(float(row["risk_adjusted_pnl"])) for row in rows)

    metadata = yaml.safe_load(open(f"{run}/eval/metadata.yaml"))
    assert metadata["environment_contract"] == ENVIRONMENT_CONTRACT
    assert metadata["normalization_fit_split"] == "train"
    assert metadata["as_impact_calibration_source"] == (
        "seeded configured CLOB-flow simulation"
    )
    assert set(metadata["as_calibration"]) == {"A", "k"}
    assert metadata["as_specification"] == {
        "risk_aversion_gamma": 0.0,
        "volatility_calibrated": False,
    }
    assert metadata["checkpoint_selection_split"] == "validation"
    assert metadata["policy_evaluation_split"] == "test"
    selection = metadata["checkpoint_selection"]
    assert selection["eligibility"]["eligible"]
    assert selection["economic_safety"]["reportable"]
    for benchmark in ("as", "twap"):
        assert differences_mod.main(["--run-dir", run, "--benchmark", benchmark]) == 0
        path = f"{run}/eval/policy_difference_{benchmark}.csv"
        diff_rows = list(csv.DictReader(open(path)))
        assert len(diff_rows) == 2
        expected = sum(float(row["policy_minus_benchmark"]) for row in diff_rows)
        assert float(diff_rows[-1]["cumulative_policy_minus_benchmark"]) == pytest.approx(expected)
