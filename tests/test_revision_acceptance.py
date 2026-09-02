"""Acceptance checklist for the revised auction/CLOB simulator.

These tests follow Section 17 of ``paper/main.tex``.  They intentionally test
observable contracts at subsystem boundaries instead of preserving legacy
implementation details from the pre-revision simulator.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import (
    REPO_ROOT,
    load_algo_cfg,
    load_dqn_cfg,
    load_historical_cfg,
    load_synthetic_cfg,
    new_env,
)
from lmm.agents.base import ENVIRONMENT_CONTRACT, REWARD_SCALE, Transition
from lmm.agents.benchmarks import TWAPBenchmarkAgent
from lmm.agents.ddpg import DDPGAgent
from lmm.agents.dqn import DQNAgent
from lmm.agents.sac import SACAgent
from lmm.agents.td3 import TD3Agent
from lmm.config import load_config, save_resolved
from lmm.data.historical_artifact import validate_historical_artifact
from lmm.env import mdp as mdp_module
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.features import COMMON_FEATURES
from lmm.experiments.plotting import collect_runs
from lmm.market.auction import AgentOrderLedger, ExogenousAuctionFlow
from lmm.market.clearing import (
    Algo1Estimator,
    CappedPositivePartSchedule,
    TerminalAllocation,
    allocate_pro_rata,
)
from lmm.market.clob import BookSnapshot
from lmm.market.generator import EpisodeGrid, EpisodeRealization, MarketGenerator
from lmm.market.midprice import RoughHestonMidPrice
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.rl.replay import ReplayBatch
from lmm.utils.seeding import seed_everything


def _small_cfg(*overrides: str):
    return load_synthetic_cfg(
        "grid.tau_op=6",
        "grid.tau_cl=9",
        "grid.h=3",
        "grid.T_physical=9.0",
        *overrides,
    )


def _small_dqn_cfg(*overrides: str):
    return load_dqn_cfg(
        "grid.tau_op=6",
        "grid.tau_cl=9",
        "grid.h=3",
        "grid.T_physical=9.0",
        *overrides,
    )


def _snapshot(k_mid: int, ask: list[float], bid: list[float]) -> BookSnapshot:
    ask_values = np.zeros(12)
    bid_values = np.zeros(12)
    ask_values[: len(ask)] = ask
    bid_values[: len(bid)] = bid
    return BookSnapshot(k_mid, ask_values, bid_values)


def _drive_to_auction(env, seed: int = 7) -> dict:
    env.reset(seed=seed)
    last = {}
    while env.phase == "clob":
        _, _, _, _, last = env.step(ClobAction(0.0, 0))
    return last


class _ScriptedRng:
    """Small deterministic RNG for one or more six-proposal auction steps."""

    def __init__(self, proposal_rows, scalar_randoms=(), integer_value: int = 0):
        self._rows = iter(np.asarray(row, dtype=float) for row in proposal_rows)
        self._scalars = iter(float(x) for x in scalar_randoms)
        self._integer_value = int(integer_value)

    def random(self, size=None):
        if size is not None:
            assert int(size) == 6
            return next(self._rows).copy()
        return next(self._scalars)

    def uniform(self, low, high):
        return 0.5 * (float(low) + float(high))

    def integers(self, low, high=None):
        upper = int(high if high is not None else low)
        lower = int(low if high is not None else 0)
        return min(max(self._integer_value, lower), upper - 1)


def _flow(cfg, **probabilities) -> ExogenousAuctionFlow:
    params = replace(cfg.auction_flow, **probabilities)
    return ExogenousAuctionFlow(params, cfg.grid, cfg.clob_flow)


def test_synthetic_and_historical_differ_only_in_midprice_and_identity():
    for algo in ("dqn", "ddpg", "td3", "sac"):
        synthetic = load_config(
            REPO_ROOT / "configs/base.yaml",
            REPO_ROOT / "configs/synthetic_rough_heston.yaml",
            REPO_ROOT / f"configs/algo/{algo}.yaml",
        )
        historical = load_config(
            REPO_ROOT / "configs/base.yaml",
            REPO_ROOT / "configs/historical_sp500_midquotes.yaml",
            REPO_ROOT / f"configs/algo/{algo}.yaml",
        )
        for cfg in (synthetic, historical):
            assert cfg.grid.time_unit == "minutes"
            assert cfg.grid.physical_time_per_grid_unit == pytest.approx(1.0)
            assert cfg.grid.tau_op == 120
            assert cfg.grid.tau_cl == 150
            assert cfg.grid.h == 30
        assert synthetic.midprice.rough_heston.s_star == pytest.approx(252 * 6.5 * 60)
        assert synthetic.experiment.episodes == historical.experiment.episodes == 800
        for section in (
            "grid",
            "clob_flow",
            "auction_flow",
            "algo1",
            "reward",
            "rl",
            "actions",
            "features",
            "benchmark",
            "algo",
        ):
            assert getattr(synthetic, section) == getattr(historical, section), (
                algo,
                section,
            )


def test_historical_dataset_has_verified_disjoint_nonempty_pools():
    cfg = load_historical_cfg()
    params = cfg.midprice.historical
    assert params is not None
    manifest = validate_historical_artifact(
        params,
        REPO_ROOT,
        horizon=cfg.grid.tau_op,
    )
    assert manifest["dataset_id"] == params.split_id
    assert manifest["tickers"] == ["CAT", "PG", "GOOGL", "JPM", "MSFT"]
    assert manifest["n_sessions"] == 20
    split_dates = manifest["split_session_dates"]
    assert {name: len(dates) for name, dates in split_dates.items()} == {
        "train": 10,
        "validation": 5,
        "test": 5,
    }
    pools = [set(split_dates[name]) for name in ("train", "validation", "test")]
    assert pools[0].isdisjoint(pools[1])
    assert pools[0].isdisjoint(pools[2])
    assert pools[1].isdisjoint(pools[2])

    for split, expected_count in (("train", 10), ("validation", 5), ("test", 5)):
        env = mdp_module.make_env(
            cfg,
            symbol="MSFT",
            repo_root=REPO_ROOT,
            data_split=split,
        )
        assert env._midprice._paths.shape == (expected_count, cfg.grid.tau_op + 1)


def test_public_environment_does_not_expose_presampled_future_paths_or_grid():
    env = mdp_module.make_env(
        load_historical_cfg(),
        symbol="MSFT",
        repo_root=REPO_ROOT,
        data_split="train",
    )
    env.reset(seed=20260831)

    assert not hasattr(env, "midprice")
    assert not hasattr(env, "episode_grid")
    assert not hasattr(env._midprice, "paths")
    assert not hasattr(env._midprice, "path")
    assert not hasattr(env._midprice, "path_index")
    with pytest.raises(RuntimeError, match="unavailable until the episode completes"):
        _ = env.completed_episode_grid


def test_historical_mid_is_observed_through_open_then_frozen_during_call():
    cfg = load_historical_cfg()
    env = mdp_module.make_env(
        cfg,
        symbol="MSFT",
        repo_root=REPO_ROOT,
        data_split="test",
    )
    env.reset(seed=20260831)
    while env.phase == "clob":
        env.step(ClobAction(0.0, 0))
    open_mid = env.s_mid
    assert env.t == cfg.grid.tau_op
    assert open_mid == pytest.approx(env._prepared_frozen_mid)
    terminated = False
    while not terminated:
        assert env.s_mid == open_mid
        _, _, terminated, _, _ = env.step(AuctionAction(0.0, 0, 0))
    assert env.s_mid == open_mid


def test_realized_grid_is_strict_and_arrivals_form_an_exact_partition():
    env = new_env(_small_cfg())
    env.reset(seed=20260831)
    with pytest.raises(RuntimeError, match="unavailable until the episode completes"):
        _ = env.completed_episode_grid
    grid = env._episode_grid
    assert np.all(np.diff(grid.all_times) > 0.0)
    assert grid.clob_times[-1] == env.grid.tau_op - 1
    assert grid.auction_times[0] == env.grid.tau_op
    assert grid.terminal_time == env.grid.tau_cl

    tape = env._generator._realization
    assert tape is not None
    for times, bounds in (
        (tape.buy_arrival_times, tape.buy_bounds),
        (tape.sell_arrival_times, tape.sell_bounds),
    ):
        seen: list[int] = []
        for i in range(len(grid.clob_times)):
            lo, hi = tape.interval_times(i)
            indices = list(range(int(bounds[i]), int(bounds[i + 1])))
            seen.extend(indices)
            assert all(lo < float(times[j]) <= hi for j in indices)
        assert seen == list(range(len(times)))


def test_policy_features_do_not_read_the_presampled_future():
    env = new_env(_small_cfg())
    obs, _ = env.reset(seed=11)
    assert tuple(env.cfg.features.clob) == COMMON_FEATURES
    assert len(obs) == 18

    tape = env._generator._realization
    assert tape is not None
    changed_asks = tape.ask_tops.copy()
    changed_bids = tape.bid_tops.copy()
    changed_asks[1:] += 10_000.0
    changed_bids[1:] += 20_000.0
    env._generator._realization = replace(
        tape,
        ask_tops=changed_asks,
        bid_tops=changed_bids,
    )
    np.testing.assert_array_equal(env.features.clob_features(env), obs)


def test_clob_order_expires_and_final_residual_uses_actual_matching():
    cfg = _small_cfg()
    gen = MarketGenerator(cfg.clob_flow, cfg.auction_flow, cfg.grid)
    grid = EpisodeGrid(
        clob_times=np.array([0.0, 5.0]),
        auction_times=np.array([6.0, 7.0, 8.0]),
        terminal_time=9.0,
    )
    gen._realization = EpisodeRealization(
        grid=grid,
        buy_arrival_times=np.array([5.5]),
        sell_arrival_times=np.array([]),
        buy_volumes=np.array([3.0]),
        sell_volumes=np.array([]),
        buy_bounds=np.array([0, 0, 1]),
        sell_bounds=np.array([0, 0, 0]),
        ask_tops=np.array([10.0, 10.0]),
        bid_tops=np.array([10.0, 10.0]),
    )
    gen.prepare_clob_decision(0, 10_000)
    gen.book.place_agent_order(5.0, 12)
    gen.step_clob_index(0)
    gen.book.clear_agent_order()
    gen.prepare_clob_decision(1, 10_000)
    assert gen.book.agent_remaining == 0.0

    gen.book.place_agent_order(5.0, 12)
    flow = gen.step_clob_index(1)
    assert flow.executed_agent == 0.0
    assert flow.residual_exogenous is not None
    assert flow.residual_exogenous.agent_remaining == 0.0
    assert flow.residual_exogenous.ask_volumes[0] == pytest.approx(7.0)


def test_carryover_recalibration_replaces_only_O_n():
    cfg = _small_cfg()
    est = Algo1Estimator(cfg.algo1, cfg.grid)
    est.reset()
    est.observe(1, _snapshot(10_000, [4.0], []))  # tick 10001
    est.observe(2, _snapshot(10_000, [6.0, 9.0], []))
    calibration = est.final_replacement_calibration(
        _snapshot(10_000, [2.0], []), n=2
    )

    expected_e = (4.0 + 2.0) / 2.0
    expected_second = (4.0**2 + 2.0**2) / 2.0
    expected_k = (2.0 * expected_e - expected_second / expected_e) / cfg.grid.alpha
    assert calibration.e_hat[10_001] == pytest.approx(expected_e)
    assert calibration.varsigma_hat[10_001] == pytest.approx(expected_second)
    assert calibration.K_hat[10_001] == pytest.approx(expected_k)
    assert 10_002 not in calibration.K_hat
    np.testing.assert_array_equal(calibration.ticks, [10_001])


def test_initialized_slope_floor_and_persistent_schedule_survive_cancellation():
    cfg = _small_cfg()
    flow = _flow(cfg, p1=0.0, p2=0.5, p3=0.0, p4=0.0)
    flow.reset(100.0)
    persistent_id = next(r.schedule_id for r in flow.active_schedules if r.persistent)
    flow.inject_market_maker(1.0, 100.0)
    assert sum(r.slope for r in flow.active_schedules) >= cfg.auction_flow.D_mu

    # Only D is proposed (the second indicator); it may cancel the injected
    # schedule, but the designated schedule is never eligible.
    rng = _ScriptedRng([[1, 0, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]])
    assert flow.step(rng).schedule_cancel == "accepted"
    assert flow.step(rng).schedule_cancel == "ineligible"
    survivor = next(r for r in flow.active_schedules if r.schedule_id == persistent_id)
    assert survivor.persistent and survivor.active
    flow.assert_valid()


def test_invalid_proposal_rolls_back_and_is_counted_as_rejected():
    cfg = _small_cfg()
    flow = _flow(cfg, p1=0.5, p2=0.5, p3=0.5, p4=0.5)
    flow.reset(0.0)
    before = (
        [(r.schedule_id, r.slope, r.reference, r.active) for r in flow.schedules],
        flow.buy_volumes.copy(),
        flow.sell_volumes.copy(),
    )
    # Only J- (sell arrival, fifth indicator) is proposed.  At a zero-price
    # fallback book every positive sell volume makes J+I negative.
    event = flow.step(_ScriptedRng([[1, 1, 1, 1, 0, 1]], scalar_randoms=[0.0]))
    after = (
        [(r.schedule_id, r.slope, r.reference, r.active) for r in flow.schedules],
        flow.buy_volumes.copy(),
        flow.sell_volumes.copy(),
    )
    assert event.sell_arrival == "rejected"
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[2], after[2])
    assert flow.proposal_counts["sell_arrival"] == {
        "proposed": 1,
        "accepted": 0,
        "rejected": 1,
        "ineligible": 0,
    }


def test_market_order_indices_count_accepted_arrivals_not_live_orders():
    cfg = _small_cfg()
    flow = _flow(cfg, p1=0.5, p2=0.5, p3=0.5, p4=0.5)
    flow.reset(100.0)
    rng = _ScriptedRng(
        [
            [1, 1, 0, 1, 1, 1],  # J+ only
            [1, 1, 1, 0, 1, 1],  # G+ only
        ],
        scalar_randoms=[0.0],
    )
    assert flow.step(rng).buy_arrival == "accepted"
    assert flow.n_buy == 1 and flow.buy_volumes[0] > 0.0
    assert flow.step(rng).buy_cancel == "accepted"
    assert flow.n_buy == 1 and flow.buy_volumes[0] == 0.0


def test_lagged_indicative_price_and_strict_before_agent_cancellation():
    cfg = _small_cfg(
        "auction_flow.p1=0.0",
        "auction_flow.p2=0.0",
        "auction_flow.p3=0.0",
        "auction_flow.p4=0.0",
    )
    env = new_env(cfg)
    last_clob = _drive_to_auction(env)
    opening_h = env.h_cl
    assert opening_h == last_clob["H_next"] == last_clob["H_used"]

    env._generator.auction_flow.inject_market_maker(20.0, env.s_mid + 1.0)
    _, _, _, _, info = env.step(AuctionAction(2.0, 0, 0))
    assert info["H_used"] == opening_h
    assert env.h_cl == info["H_next"]
    assert info["H_next"] != opening_h

    ledger = AgentOrderLedger(cfg.grid)
    ledger.submit(cfg.grid.tau_op, 2.0, 100.0)
    ledger.submit(cfg.grid.tau_op + 1, 3.0, 100.0)
    ledger.apply_cancel_all(cfg.grid.tau_op + 1)
    assert not ledger.live[0] and ledger.live[1]
    ledger.submit(cfg.grid.tau_op + 2, 4.0, 100.0)
    ledger.apply_cancel_all(cfg.grid.tau_op + 2)
    assert not ledger.live[1] and ledger.live[2]


def test_dqn_action_counts_and_canonical_no_order_rules():
    cfg = _small_dqn_cfg()
    env = new_env(cfg)
    assert len(env.clob_grid) == 1 + 30 * 13 == 391
    assert len(env.auction_grid) == 2 + 32 * 21 * 2 == 1346
    zero = [a for a in env.auction_grid.actions if a.K_a == 0.0]
    assert [(a.K_a, a.ell, a.cancel) for a in zero] == [
        (0.0, 0, 0),
        (0.0, 0, 1),
    ]

    _drive_to_auction(env)
    assert env.action_mask().sum() == 673
    env.step(AuctionAction(cfg.actions.beta, 0, 0))
    assert env.action_mask().sum() == len(env.auction_grid)

    no_cancel = new_env(
        _small_dqn_cfg(
            "actions.auction_cancel_mode=never",
        )
    )
    assert len(no_cancel.auction_grid) == 1 + 32 * 21 == 673
    no_order = [a for a in no_cancel.auction_grid.actions if a.K_a == 0.0]
    assert [(a.K_a, a.ell, a.cancel) for a in no_order] == [
        (0.0, 0, 0)
    ]


def _constant_output(network, value: float) -> None:
    with torch.no_grad():
        linear = [m for m in network.modules() if isinstance(m, torch.nn.Linear)]
        for layer in linear:
            layer.weight.zero_()
            layer.bias.zero_()
        linear[-1].bias.fill_(float(value))


def test_final_clob_transition_bootstraps_from_auction_target_network():
    cfg = _small_dqn_cfg("algo.hyperparams.double_q=false")
    agent = DQNAgent(cfg, seed_everything(10, SEED_COMPONENTS, seed_torch=True))
    _constant_output(agent.q_target["clob"], 1.0)
    _constant_output(agent.q_target["auction"], 7.0)
    batch = ReplayBatch(
        obs=np.zeros((1, len(COMMON_FEATURES)), np.float32),
        action=np.zeros(1, np.int64),
        reward=np.array([2.0], np.float32),
        next_obs=np.zeros((1, len(COMMON_FEATURES)), np.float32),
        done=np.array([False]),
        junction=np.array([True]),
        next_mask=np.ones((1, len(agent.auction_grid)), bool),
    )
    assert agent.compute_targets("clob", batch).item() == pytest.approx(9.0)


_AGENTS = {
    "dqn": DQNAgent,
    "ddpg": DDPGAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
}


@pytest.mark.parametrize("algo", tuple(_AGENTS))
def test_one_update_opportunity_and_common_unclipped_scaled_reward(algo):
    loader = _small_dqn_cfg if algo == "dqn" else lambda *o: load_algo_cfg(
        algo,
        "grid.tau_op=6",
        "grid.tau_cl=9",
        "grid.h=3",
        "grid.T_physical=9.0",
        *o,
    )
    cfg = loader(
        "algo.hyperparams.min_buffer=1",
        "algo.hyperparams.min_buffer_clob=1",
        "algo.hyperparams.min_buffer_auction=1",
        "algo.hyperparams.batch_size=1",
    )
    agent = _AGENTS[algo](cfg, seed_everything(12, SEED_COMPONENTS, seed_torch=True))
    action = 0 if algo == "dqn" else np.zeros(agent._act_dim["clob"], np.float32)
    mask = np.ones(len(agent.clob_grid), bool) if algo == "dqn" else None
    raw_reward = 1.0e8
    agent.observe(
        Transition(
            obs=np.zeros(18, np.float32),
            action=action,
            reward=raw_reward,
            next_obs=np.zeros(18, np.float32),
            done=False,
            phase="clob",
            next_phase="clob",
            next_mask=mask,
        )
    )
    assert agent.hp.reward_scale == REWARD_SCALE
    assert agent.replay["clob"]._reward[0] == pytest.approx(
        raw_reward * REWARD_SCALE, rel=1e-6
    )
    assert agent.update("clob").get("n_grad_steps_clob") == 1.0
    assert agent.update("clob") == {}


def test_terminal_prorata_balances_and_environment_uses_actual_fill(monkeypatch):
    allocation = allocate_pro_rata(
        exogenous_schedule_values=np.array([10.0]),
        agent_net_quantity=10.0,
        buy_market_volume=5.0,
        sell_market_volume=0.0,
    )
    assert allocation.requested_agent == 10.0
    assert allocation.actual_agent == pytest.approx(2.5)
    assert allocation.executed_supply == pytest.approx(allocation.executed_demand)
    assert allocation.self_trade_count == 0

    cfg = load_synthetic_cfg(
        "grid.tau_op=3",
        "grid.tau_cl=4",
        "grid.h=1",
        "grid.T_physical=4.0",
        "auction_flow.p1=0.0",
        "auction_flow.p2=0.0",
        "auction_flow.p3=0.0",
        "auction_flow.p4=0.0",
    )
    env = new_env(cfg)
    _drive_to_auction(env)
    i_open = env.inventory
    fake = TerminalAllocation(
        requested_agent=50.0,
        actual_agent=2.5,
        Q_supply=10.0,
        Q_demand=10.0,
        rho_supply=0.25,
        rho_demand=1.0,
        executed_supply=10.0,
        executed_demand=10.0,
        self_trade_count=0,
    )
    monkeypatch.setattr(mdp_module, "allocate_terminal", lambda *a, **k: fake)
    _, _, done, _, info = env.step(AuctionAction(0.0, 0, 0))
    assert done
    assert info["requested_Z"] == 50.0
    assert info["Z"] == 2.5
    assert info["I_final"] == pytest.approx(i_open - 2.5)
    assert info["auction_economic_cash"] == pytest.approx(info["S_cl"] * 2.5)


def test_auction_has_no_inventory_bound_or_terminal_clipping():
    cfg = load_synthetic_cfg(
        "grid.tau_op=3",
        "grid.tau_cl=4",
        "grid.h=1",
        "grid.T_physical=4.0",
        "auction_flow.p1=0.0",
        "auction_flow.p2=0.0",
        "auction_flow.p3=0.0",
        "auction_flow.p4=0.0",
    )
    env = new_env(cfg)
    _drive_to_auction(env)
    env._generator.auction_flow.inject_taker(+1, 10_000.0)
    _, _, done, _, info = env.step(
        AuctionAction(cfg.actions.auction_K_grid_max, -cfg.actions.B_max, 0)
    )
    assert done
    assert info["Z"] > cfg.grid.I0
    assert info["I_final"] == pytest.approx(cfg.grid.I0 - info["Z"])
    assert info["I_final"] < 0.0
    assert info["self_trade_count"] == 0


def test_unshaped_return_equals_risk_adjusted_pnl_plus_initial_notional():
    cfg = _small_cfg(
        "reward.shaping_enabled=false",
        "reward.center_initial_inventory_value=false",
    )
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    result = run_episode(env, agent, 77, chi=cfg.rl.chi, train=False)
    assert result.return_undisc == pytest.approx(
        result.risk_adjusted_pnl
        + result.initial_mid * result.initial_inventory,
        abs=1e-8,
    )


def test_centered_unshaped_return_equals_risk_adjusted_pnl():
    cfg = _small_cfg(
        "reward.shaping_enabled=false",
        "reward.center_initial_inventory_value=true",
    )
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    result = run_episode(env, agent, 77, chi=cfg.rl.chi, train=False)
    assert result.reward_baseline_adjustment == pytest.approx(
        -result.initial_mid * result.initial_inventory, abs=1e-8
    )
    assert result.return_undisc == pytest.approx(
        result.risk_adjusted_pnl, abs=1e-8
    )


def test_no_auction_comparator_terminates_at_open_with_exogenous_mid_mark():
    cfg = _small_cfg(
        "experiment.auction_enabled=false",
        "reward.shaping_enabled=false",
        "rl.h_cl_feature_enabled=false",
    )
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    result = run_episode(env, agent, 91, chi=cfg.rl.chi, train=False)
    assert result.n_steps == len(env.completed_episode_grid.clob_times)
    assert env.t == cfg.grid.tau_op
    assert result.z_tau_cl == 0.0
    assert result.s_cl == pytest.approx(result.residual_liquidation_price)
    assert result.return_undisc == pytest.approx(
        result.risk_adjusted_pnl,
        abs=1e-8,
    )


def test_rough_heston_uses_annualized_delta_bar_t():
    cfg = _small_cfg()
    params = cfg.midprice.rough_heston
    model = RoughHestonMidPrice(params, cfg.grid)
    model.reset(np.random.default_rng(3))
    model.advance_to(2.5)
    assert model._times[1] == pytest.approx(2.5 / params.s_star, rel=1e-15)
    assert model._times[1] != pytest.approx(2.5)
    assert cfg.grid.time_unit == "minutes"


def test_benchmark_schedule_keeps_positive_part_and_inventory_cap():
    schedule = CappedPositivePartSchedule(slope=4.0, reference=100.0, cap=7.0)
    assert schedule.value(99.0) == 0.0
    assert schedule.value(101.0) == 4.0
    assert schedule.value(1_000.0) == 7.0

    cfg = _small_dqn_cfg()
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    _drive_to_auction(env)
    action = agent._auction_action()
    assert not isinstance(action, AuctionAction)
    assert action.reference_price == pytest.approx(env.h_cl)
    assert action.quantity_cap == pytest.approx(env.inventory)
    external = CappedPositivePartSchedule(
        action.K_a, action.reference_price, action.quantity_cap
    )
    assert external.value(action.reference_price - 1.0) == 0.0
    assert external.value(action.reference_price + 1.0e6) == pytest.approx(env.inventory)


def test_old_checkpoints_and_result_directories_are_rejected(tmp_path: Path):
    cfg = _small_dqn_cfg()
    agent = DQNAgent(cfg, seed_everything(19, SEED_COMPONENTS, seed_torch=True))
    old_checkpoint = tmp_path / "old.pt"
    torch.save(
        {
            "artifact_schema_version": 9,
            "environment_contract": "unified-minute-auction-mdp-2026-09-01-v6",
        },
        old_checkpoint,
    )
    with pytest.raises(ValueError, match="artifact_schema_version mismatch"):
        agent.load(old_checkpoint)

    torch.save(
        {
            "artifact_schema_version": cfg.experiment.artifact_schema_version,
            "environment_contract": "unified-minute-auction-mdp-2026-09-01-v6",
        },
        old_checkpoint,
    )
    with pytest.raises(ValueError, match="environment contract mismatch"):
        agent.load(old_checkpoint)

    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    save_resolved(cfg, run_dir / "config_resolved.yaml")
    (run_dir / "seed.txt").write_text("19\n")
    (run_dir / "eval").mkdir()
    (run_dir / "eval" / "metadata.yaml").write_text(
        "environment_contract: pre-revision\n"
    )
    with pytest.raises(ValueError, match=ENVIRONMENT_CONTRACT):
        collect_runs([run_dir])
