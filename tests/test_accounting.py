"""Economic P&L accounting and primary-validation metric tests."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from helpers import load_dqn_cfg, new_env
from lmm.agents.benchmarks import TWAPBenchmarkAgent
from lmm.env.action_spaces import AuctionAction, ClobAction, FiveCoordinateAction
from lmm.experiments.accounting import compute_liquidation_accounting
from lmm.experiments import train as train_mod
from lmm.rl.loops import run_episode


def test_liquidation_accounting_hand_case_and_inventory_conservation():
    out = compute_liquidation_accounting(
        initial_inventory=100.0,
        initial_mid=100.0,
        clob_exec_qty=60.0,
        clob_cash=6012.0,
        auction_exec_qty=30.0,
        auction_price=100.2,
        final_inventory=10.0,
        residual_liquidation_price=100.1,
        cancel_cost=1.5,
        inventory_penalty=4.0,
    )
    assert out.auction_cash == pytest.approx(3006.0)
    assert out.residual_liquidation_cash == pytest.approx(1001.0)
    assert out.liquidation_pnl_gross == pytest.approx(19.0)
    assert out.liquidation_pnl_net == pytest.approx(17.5)
    assert out.economic_objective == pytest.approx(13.5)


def test_wrong_side_auction_purchase_is_signed_cash_outflow():
    out = compute_liquidation_accounting(
        initial_inventory=10.0,
        initial_mid=100.0,
        clob_exec_qty=0.0,
        clob_cash=0.0,
        auction_exec_qty=-2.0,
        auction_price=101.0,
        final_inventory=12.0,
        residual_liquidation_price=100.0,
    )
    assert out.auction_cash == -202.0
    assert out.residual_liquidation_cash == 1200.0
    assert out.liquidation_pnl_gross == -2.0


def test_accounting_rejects_broken_inventory_identity():
    with pytest.raises(ValueError, match="inventory conservation"):
        compute_liquidation_accounting(
            initial_inventory=10.0,
            initial_mid=100.0,
            clob_exec_qty=1.0,
            clob_cash=100.0,
            auction_exec_qty=2.0,
            auction_price=100.0,
            final_inventory=8.0,
            residual_liquidation_price=100.0,
        )


def test_validation_summary_keeps_shaped_return_but_scores_risk_adjusted_pnl(monkeypatch):
    episodes = {
        1: SimpleNamespace(return_undisc=1000.0, return_disc=900.0,
                           liquidation_pnl_gross=-5.0, liquidation_pnl_net=-6.0,
                           economic_objective=-10.0, pnl=-6.0,
                           risk_adjusted_pnl=-10.0),
        2: SimpleNamespace(return_undisc=-1000.0, return_disc=-900.0,
                           liquidation_pnl_gross=15.0, liquidation_pnl_net=14.0,
                           economic_objective=6.0, pnl=14.0,
                           risk_adjusted_pnl=6.0),
    }

    monkeypatch.setattr(train_mod, "run_episode", lambda env, agent, seed, **kw: episodes[seed])
    summary = train_mod._run_eval(None, None, [1, 2], 0.99, "risk_adjusted_pnl")
    assert summary["return_mean"] == 0.0
    assert summary["pnl_mean"] == 4.0
    assert summary["risk_adjusted_pnl_mean"] == -2.0
    assert summary["checkpoint_score"] == -2.0


@pytest.mark.parametrize("cost_name", ["cancel_cost", "inventory_penalty"])
def test_accounting_rejects_negative_economic_cost(cost_name):
    kwargs = {"cancel_cost": 0.0, "inventory_penalty": 0.0}
    kwargs[cost_name] = -0.1
    with pytest.raises(ValueError, match="economic costs must be non-negative"):
        compute_liquidation_accounting(
            initial_inventory=10.0,
            initial_mid=100.0,
            clob_exec_qty=5.0,
            clob_cash=500.0,
            auction_exec_qty=5.0,
            auction_price=100.0,
            final_inventory=0.0,
            residual_liquidation_price=100.0,
            **kwargs,
        )


def test_benchmark_auction_action_uses_capped_positive_part_schedule():
    cfg = load_dqn_cfg()
    env = new_env(cfg)
    env.reset(seed=7)
    while env.phase == "clob":
        env.step(ClobAction(0.0, 0))
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    agent._exec_prices = [env.s_mid - 100.0]  # force an extreme raw quote target
    action = agent._auction_action()
    assert 0.0 <= action.K_a <= cfg.actions.auction_K_grid_max
    assert not isinstance(action, AuctionAction)
    assert action.quantity_cap == pytest.approx(env.inventory)
    assert action.reference_price == pytest.approx(agent._exec_prices[0])


def test_public_auction_action_has_only_manuscript_coordinates():
    assert tuple(field.name for field in fields(AuctionAction)) == (
        "K_a",
        "ell",
        "cancel",
    )
    assert tuple(field.name for field in fields(FiveCoordinateAction)) == (
        "volume",
        "delta",
        "K_a",
        "ell",
        "cancel",
    )
    with pytest.raises(TypeError):
        AuctionAction(1.0, 0, 0, reference_price=100.0)


def test_benchmark_uses_formal_zero_inventory_rule_without_dust_cutoff():
    cfg = load_dqn_cfg()
    env = new_env(cfg)
    env.reset(seed=7)
    while env.phase == "clob":
        env.step(ClobAction(0.0, 0))
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)

    env._inventory = 0.005
    agent.start_episode(0)
    tiny = agent._auction_action()
    assert tiny.K_a > 0.0
    assert tiny.quantity_cap == pytest.approx(0.005)

    env._inventory = 0.0
    agent.start_episode(1)
    assert agent._auction_action() == AuctionAction(0.0, 0, 0)


def test_raw_auction_actions_cannot_bypass_common_bounds():
    cfg = load_dqn_cfg()
    env = new_env(cfg)
    env.reset(seed=7)
    while env.phase == "clob":
        env.step(ClobAction(0.0, 0))
    with pytest.raises(ValueError, match=r"K\^a exceeds"):
        env.step(AuctionAction(cfg.actions.auction_K_grid_max + 1.0, 0, 0))
    with pytest.raises(ValueError, match="auction ell"):
        env.step(AuctionAction(1.0, cfg.actions.B_max + 1, 0))


def test_dqn_auction_grid_has_unique_canonical_zero_slope_actions():
    cfg = load_dqn_cfg()
    env = new_env(cfg)
    grid = env.auction_grid
    assert len(grid) == 1346
    zero = [a for a in grid.actions if a.K_a == 0.0]
    assert [(a.K_a, a.ell, a.cancel) for a in zero] == [
        (0.0, 0, 0),
        (0.0, 0, 1),
    ]
    semantics = {
        (a.K_a, a.ell if a.K_a > 0.0 else 0, a.cancel)
        for a in grid.actions
    }
    assert len(semantics) == len(grid)


def test_no_cancel_treatment_has_genuine_673_action_grid_and_rejects_cancel():
    cfg = load_dqn_cfg(
        "actions.auction_cancel_mode=never",
    )
    env = new_env(cfg)
    grid = env.auction_grid
    assert len(grid) == 673
    assert (
        grid.actions[0].K_a,
        grid.actions[0].ell,
        grid.actions[0].cancel,
    ) == (0.0, 0, 0)
    assert all(a.cancel == 0 for a in grid.actions)
    env.reset(seed=7)
    while env.phase == "clob":
        env.step(ClobAction(0.0, 0))
    with pytest.raises(ValueError, match="cancellation is disabled"):
        env.step(AuctionAction(1.0, 0, 1))
    env.step(AuctionAction(1.0, 0, 0))
    assert env._ledger.cancel_admissible()
    assert not env.cancel_admissible


def test_completed_episode_economic_objective_decomposes_exactly():
    cfg = load_dqn_cfg()
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    out = run_episode(env, agent, env_seed=20260828, chi=cfg.rl.chi, train=False)
    assert out.economic_objective == pytest.approx(
        out.liquidation_pnl_gross - out.cancel_cost - out.inventory_penalty
    )


def test_no_cancel_treatment_has_zero_cancellations_and_fees():
    cfg = load_dqn_cfg(
        "actions.auction_cancel_mode=never",
    )
    env = new_env(cfg)
    agent = TWAPBenchmarkAgent(cfg)
    agent.bind(env)
    agent.start_episode(0)
    out = run_episode(env, agent, env_seed=20260828, chi=cfg.rl.chi, train=False)
    assert out.cancel_count == 0
    assert out.cancel_cost == 0.0
