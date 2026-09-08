"""Numerical refinement must never become a market event or learning step."""

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from helpers import load_synthetic_cfg, new_env, NOOP_CLOB, NOOP_AUCTION
from legacy_episode_reference import (
    REFERENCE,
    capture_episode,
    reference_episode,
    validate_reference,
)
from lmm.config import ConfigError, environment_contract, artifact_asdict
from lmm.market.midprice import (
    RoughHestonMidPrice,
    aggregate_brownian_increments,
    internal_time_grid,
)


def refined_cfg(step=0.5):
    # Empty weights are permitted before a new training-only forecast fit.
    return load_synthetic_cfg(
        "algo1.clob_forecast_weights=[]",
        f"midprice.rough_heston.rough_heston_max_step_minutes={step}",
    )


@pytest.mark.parametrize("step", [0, -1, float("nan"), float("inf"), True])
def test_invalid_steps(step):
    with pytest.raises(ValueError):
        internal_time_grid([0, 1], 120, step)
    with pytest.raises(ValueError):
        replace(
            load_synthetic_cfg().midprice.rough_heston,
            rough_heston_max_step_minutes=step,
        )


def test_grid_inclusion_bounds_and_nesting():
    anchors = [0.0, 1.0, 1.0, 2.134512, 7.9999, 119.0, 120.0]
    previous = internal_time_grid(anchors, 120.0, None)
    for step in [2.0, 1.0, 0.7, 0.5, 0.25]:
        grid = internal_time_grid(anchors, 120.0, step)
        assert np.all(np.diff(grid) > 0)
        assert np.max(np.diff(grid)) <= step + 1e-13
        assert np.isin(previous, grid).all()
        assert np.isin(anchors, grid).all()
        assert grid[-1] == 120
        previous = grid


def test_brownian_time_conversion_and_aggregation():
    fine = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
    normals = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0], [2.0, -3.0]])
    dw = normals * np.sqrt(np.diff(fine)[:, None] / 98280.0)
    actual = aggregate_brownian_increments(fine, dw, [0.0, 0.5, 2.0])
    np.testing.assert_array_equal(actual, [dw[0] + dw[1], dw[2] + dw[3]])
    with pytest.raises(ValueError):
        aggregate_brownian_increments(fine, dw, [0.0, 0.6, 2.0])


def test_constant_variance_analytic_log_price():
    cfg = refined_cfg()
    p = replace(cfg.midprice.rough_heston, theta=0.0, varsigma=0.0, nu=0.0)
    times = internal_time_grid([0, 1.3, 2.7], 4.0, 0.5)
    dw = np.random.default_rng(4).normal(size=(len(times) - 1, 2)) * np.sqrt(
        np.diff(times)[:, None] / 98280.0
    )
    model = RoughHestonMidPrice(p, cfg.grid)
    model.reset(np.random.default_rng(2))
    model.prepare_grid([0, 1.3, 2.7], 4.0, increments=dw)
    for t in [1.3, 2.7, 4.0]:
        model.advance_to(t)
    expected = (
        math.log(cfg.grid.S0)
        + np.r_[
            0.0,
            np.cumsum(
                -0.5 * p.v0 * np.diff(times) / 98280.0
                + math.sqrt(p.v0)
                * (p.rho * dw[:, 0] + math.sqrt(1 - p.rho**2) * dw[:, 1])
            ),
        ]
    )
    np.testing.assert_allclose(
        np.log(model.raw_price_path), expected, atol=2e-15, rtol=2e-15
    )
    np.testing.assert_array_equal(model.variance_path, np.full(len(times), p.v0))
    np.testing.assert_allclose(model._times, times / 98280.0, atol=1e-19)


@pytest.mark.parametrize("method", ["naive", "vectorized"])
def test_full_history_reference_with_negative_variance(method):
    cfg = refined_cfg(0.5)
    p = cfg.midprice.rough_heston
    times = internal_time_grid([0, 1.1, 2.0], 3.0, 0.5)
    dt = np.diff(times) / 98280.0
    dw = np.random.default_rng(5).normal(size=(len(dt), 2)) * np.sqrt(dt[:, None])
    dw[0, 0] = -0.1  # Force a negative PRE-truncation value.
    model = RoughHestonMidPrice(p, cfg.grid, method=method)
    model.reset(np.random.default_rng(123))
    model.prepare_grid([0, 1.1, 2.0], 3.0, increments=dw)
    for t in [1.1, 2.0, 3.0]:
        model.advance_to(t)
    years = times / 98280.0
    v = [p.v0]
    y = [math.log(cfg.grid.S0)]
    for j in range(1, len(times)):
        vp = max(v[-1], 0.0)
        y.append(
            y[-1]
            - 0.5 * vp * dt[j - 1]
            + math.sqrt(vp)
            * (p.rho * dw[j - 1, 0] + math.sqrt(1 - p.rho**2) * dw[j - 1, 1])
        )
        v.append(
            p.v0
            + sum(
                (years[j] - years[i]) ** (p.H - 0.5)
                / math.gamma(p.H + 0.5)
                * (
                    (p.theta - p.kappa * max(v[i], 0)) * dt[i]
                    + p.xi * math.sqrt(max(v[i], 0)) * dw[i, 0]
                )
                for i in range(j)
            )
        )
    np.testing.assert_allclose(model.variance_path, v, atol=2e-15)
    np.testing.assert_allclose(model.raw_price_path, np.exp(y), atol=2e-12)
    diag = model.variance_diagnostics()
    assert diag["finite"] and diag["minimum_variance"] < 0
    assert diag["negative_node_fraction"] == np.mean(np.asarray(v) < 0)
    assert diag["negative_time_fraction"] == pytest.approx(
        np.dot(np.asarray(v[:-1]) < 0, dt) / dt.sum()
    )


def test_legacy_golden_episode():
    # Original source is frozen, so both sides use this runtime's floating-point
    # operations without depending on one platform's JSON-float fingerprint.
    actual = capture_episode(Path(__file__).resolve().parents[1])
    expected = reference_episode()
    assert len(actual["rows"]) == len(expected["rows"]) == 71
    assert actual["rng_state"] == expected["rng_state"]
    assert json.dumps(actual["rows"], sort_keys=True, allow_nan=False) == json.dumps(
        expected["rows"], sort_keys=True, allow_nan=False
    ), "legacy episode differs from frozen pre-refinement source on the same runtime"


def test_legacy_reference_detects_one_ulp_change(monkeypatch):
    from lmm.env.mdp import MarketMakingEnv

    original_step = MarketMakingEnv.step

    def changed_step(self, action):
        obs, reward, done, truncated, info = original_step(self, action)
        info = dict(info, H_next=math.nextafter(info["H_next"], math.inf))
        return obs, reward, done, truncated, info

    monkeypatch.setattr(MarketMakingEnv, "step", changed_step)
    with pytest.raises(AssertionError, match="legacy episode differs"):
        test_legacy_golden_episode()


def test_legacy_reference_rejects_modified_source(tmp_path):
    reference = tmp_path / "reference"
    shutil.copytree(REFERENCE, reference, ignore=shutil.ignore_patterns("__pycache__"))
    source = reference / "src/lmm/env/mdp.py"
    source.write_bytes(source.read_bytes() + b"\n# changed fixture\n")
    with pytest.raises(ValueError, match="frozen reference source changed"):
        validate_reference(reference)


def test_market_interface_streams_rounding_and_freezing():
    from lmm.market.clearing import round_half_up_to_tick

    legacy = new_env(load_synthetic_cfg("algo1.clob_forecast_weights=[]"))
    refined = new_env(refined_cfg(0.25))
    for env in [legacy, refined]:
        env.reset(seed=53)
    np.testing.assert_array_equal(
        legacy._episode_grid.all_times, refined._episode_grid.all_times
    )
    assert legacy.np_random.bit_generator.state == refined.np_random.bit_generator.state
    for field in (
        "buy_arrival_times",
        "sell_arrival_times",
        "buy_volumes",
        "sell_volumes",
        "ask_tops",
        "bid_tops",
    ):
        np.testing.assert_array_equal(
            getattr(legacy._generator._realization, field),
            getattr(refined._generator._realization, field),
        )
    model = refined._midprice
    indices = np.searchsorted(
        model._internal_grid, np.r_[refined._episode_grid.clob_times, 120.0]
    )
    rounded = [
        round_half_up_to_tick(x, refined.grid.alpha)
        for x in model.raw_price_path[indices]
    ]
    np.testing.assert_array_equal(
        rounded, np.r_[refined._clob_mid_values, refined._prepared_frozen_mid]
    )
    assert len(model.variance_path) > len(rounded)
    for env in [legacy, refined]:
        count = 0
        prior = env.h_cl
        v_count = len(env._midprice.variance_path)
        while True:
            phase = env.phase
            _, _, done, _, info = env.step(
                NOOP_CLOB if phase == "clob" else NOOP_AUCTION
            )
            assert info["H_used"] == prior
            if phase == "clob" and env.phase == "auction":
                assert info["H_next"] == info["H_used"]  # opening forecast inheritance
            prior = info["H_next"]
            count += 1
            assert len(env.paper_state()["X3"]) == count + 1
            assert len(env._midprice.variance_path) == v_count
            if env.phase == "auction":
                assert env.s_mid == env._prepared_frozen_mid
            if done:
                break
        assert count == len(env._episode_grid.decision_times)


def test_refinement_reproducibility_and_price_stream_isolation():
    a, b = new_env(refined_cfg()), new_env(refined_cfg())
    a.reset(seed=12)
    b.reset(seed=12)
    np.testing.assert_array_equal(a._clob_mid_values, b._clob_mid_values)
    np.testing.assert_array_equal(a._midprice.variance_path, b._midprice.variance_path)
    assert a._midprice._rng is not a.np_random


def test_artifact_and_forecast_compatibility():
    legacy = load_synthetic_cfg()
    fine = refined_cfg()
    assert environment_contract(legacy) != environment_contract(fine)
    assert environment_contract(fine) != environment_contract(refined_cfg(0.25))
    assert (
        "rough_heston_max_step_minutes"
        not in artifact_asdict(legacy.midprice)["rough_heston"]
    )
    assert "clob_forecast_price_generator" not in artifact_asdict(legacy.algo1)
    with pytest.raises(ConfigError, match="forecast weights"):
        load_synthetic_cfg("midprice.rough_heston.rough_heston_max_step_minutes=.5")
    with pytest.raises(ConfigError, match="s_star"):
        load_synthetic_cfg(
            "algo1.clob_forecast_weights=[]",
            "midprice.rough_heston.rough_heston_max_step_minutes=.5",
            "midprice.rough_heston.s_star=252",
        )


def test_diagnostic_smoke_and_coupling():
    from lmm.experiments.refinement import diagnose

    result = diagnose(load_synthetic_cfg(), episodes=2, seed=3, steps=[1.0, 0.5, 0.25])
    assert result["summary"]["0.25m"]["price_rmse_vs_finest"] == 0
    assert result["summary"]["legacy"]["price_rmse_vs_finest"] > 0
    counts = [
        [r["decision_count"] for r in rows] for rows in result["per_episode"].values()
    ]
    assert all(c == counts[0] for c in counts)
    json.dumps(result, allow_nan=False)


def test_learning_transition_and_event_counts():
    from lmm.agents.benchmarks import TWAPBenchmarkAgent
    from lmm.rl.loops import run_episode

    class CountingPolicy(TWAPBenchmarkAgent):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.transitions = []
            self.updates = []
            self.calls = 0

        def act(self, obs, mask, phase, *, eval_mode=False):
            self.calls += 1
            return 0

        def observe(self, transition):
            self.transitions.append(transition)

        def update(self, phase=None):
            self.updates.append(phase)
            return {}

    counts = []
    for cfg in [
        load_synthetic_cfg("algo1.clob_forecast_weights=[]"),
        refined_cfg(0.25),
    ]:
        cfg = replace(
            cfg,
            rl=replace(
                cfg.rl,
                structured_warmup_episodes=0,
                market_return_control_variate=False,
                n_step=1,
            ),
        )
        env = new_env(cfg)
        agent = CountingPolicy(cfg)
        result = run_episode(env, agent, 91, chi=1.0, train=True)
        expected = len(env._episode_grid.decision_times)
        assert (
            result.n_steps
            == len(agent.transitions)
            == len(agent.updates)
            == agent.calls
            == expected
        )
        np.testing.assert_array_equal(
            [t.info["t"] for t in agent.transitions], env._episode_grid.decision_times
        )
        counts.append(
            [
                (t.info.get("n_buy_step"), t.info.get("n_sell_step"))
                for t in agent.transitions
            ]
        )
    assert counts[0] == counts[1]


def test_auction_proposal_coupling_survives_eligibility_differences():
    from lmm.experiments.refinement import CoupledEnvironment

    cfg = load_synthetic_cfg("algo1.clob_forecast_weights=[]", "auction_flow.p2=1.0")
    envs = [
        CoupledEnvironment(
            cfg, RoughHestonMidPrice(cfg.midprice.rough_heston, cfg.grid)
        )
        for _ in range(2)
    ]
    for env in envs:
        env.reset(seed=2)
        while env.phase == "clob":
            env.step(NOOP_CLOB)
    # Make cancellation eligibility differ while preserving valid books.
    envs[1]._generator.auction_flow.inject_market_maker(1.0, envs[1].s_mid)
    for _ in range(cfg.grid.h):
        outcomes = [env.step(NOOP_AUCTION)[4]["events"].as_dict() for env in envs]
        assert {k: v != "not_proposed" for k, v in outcomes[0].items()} == {
            k: v != "not_proposed" for k, v in outcomes[1].items()
        }
