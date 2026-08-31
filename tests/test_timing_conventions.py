"""Chronology tests for pre-action CLOB snapshots and lagged auction prices."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import NOOP_CLOB, drive_to_auction, load_synthetic_cfg, new_env
from lmm.env.action_spaces import AuctionAction, ClobAction


def quiet_cfg(*overrides):
    return load_synthetic_cfg(
        "auction_flow.p1=0.0",
        "auction_flow.p2=0.0",
        "auction_flow.p3=0.0",
        "auction_flow.p4=0.0",
        *overrides,
    )


def test_auction_opens_with_lagged_algorithm1_value_and_current_proposals(synthetic_cfg):
    env = new_env(synthetic_cfg)
    clob_steps = drive_to_auction(env, seed=260828)
    last = clob_steps[-1][1]
    assert env.t == synthetic_cfg.grid.tau_op
    assert env.h_cl == last["H_used"] == last["H_next"]
    # p1=1: the current auction proposal has already been accepted before the
    # opening observation, in addition to the persistent carry-over/fallback.
    assert env.n_mm >= 2
    assert env.exogenous_slope >= synthetic_cfg.auction_flow.D_mu


def test_current_book_changes_only_the_next_indicative_price():
    cfg = quiet_cfg()
    env = new_env(cfg)
    drive_to_auction(env, seed=5)
    h_open = env.h_cl
    env.generator.auction_flow.inject_market_maker(50.0, env.s_mid + 1.0)
    _, _, _, _, info = env.step(AuctionAction(2.0, 0, 0))
    assert info["H_used"] == h_open
    assert info["H_next"] != h_open
    assert env.h_cl == info["H_next"]
    _, _, _, _, next_info = env.step(AuctionAction(2.0, 0, 0))
    assert next_info["H_used"] == info["H_next"]


def test_current_agent_schedule_is_invisible_to_current_H_but_enters_next_H():
    cfg = quiet_cfg()
    a, b = new_env(cfg), new_env(cfg)
    drive_to_auction(a, seed=9)
    drive_to_auction(b, seed=9)
    h = a.h_cl
    _, _, _, _, ia = a.step(AuctionAction(0.0, 0, 0))
    _, _, _, _, ib = b.step(AuctionAction(10.0, 20, 0))
    assert ia["H_used"] == ib["H_used"] == h
    assert ia["H_next"] != ib["H_next"]


def test_cancel_removes_prior_schedules_but_not_current_schedule():
    cfg = quiet_cfg()
    env = new_env(cfg)
    drive_to_auction(env, seed=21)
    env.step(AuctionAction(3.0, 1, 0))
    env.step(AuctionAction(5.0, 4, 1))
    K, S = env._ledger.live_orders()
    np.testing.assert_allclose(K, [5.0])
    np.testing.assert_allclose(S, [env.s_mid + 4 * env.grid.alpha])


def test_final_schedule_survives_same_step_cancel_and_is_cleared():
    cfg = quiet_cfg()
    env = new_env(cfg)
    drive_to_auction(env, seed=33)
    info = None
    while True:
        final = int(env.t) == env.grid.tau_cl - 1
        cancel = int(final and env.cancel_admissible)
        _, _, done, _, info = env.step(AuctionAction(2.0, 2, cancel))
        if done:
            break
    assert info is not None and info["action"].cancel == 1
    K, S = env._ledger.live_orders()
    np.testing.assert_allclose(K, [2.0])
    np.testing.assert_allclose(S, [env.s_mid + 2 * env.grid.alpha])
    assert info["H_next"] == info["S_cl"]
    assert info["Z"] == pytest.approx(env._terminal_allocation.actual_agent)


def test_strategic_clob_action_never_enters_algorithm1_snapshot(synthetic_cfg):
    a, b = new_env(synthetic_cfg), new_env(synthetic_cfg)
    a.reset(seed=3)
    b.reset(seed=3)
    _, _, _, _, ia = a.step(NOOP_CLOB)
    _, _, _, _, ib = b.step(ClobAction(30.0, 12))
    assert ia["H_used"] == ib["H_used"] == synthetic_cfg.algo1.H0
    assert ia["H_next"] == ib["H_next"]
    assert b.generator.book.agent_remaining == 0.0


def test_h_cache_chain_across_phase_boundary_and_terminal(synthetic_cfg):
    env = new_env(synthetic_cfg)
    env.reset(seed=8)
    previous = env.h_cl
    while True:
        if env.phase == "clob":
            volume = float(min(5, int(env.inventory)))
            action = ClobAction(volume, 2 if volume else 0)
        else:
            action = AuctionAction(2.0, 2, 0)
        _, _, done, _, info = env.step(action)
        assert info["H_used"] == previous
        previous = info["H_next"]
        if done:
            break
    assert info["S_cl"] == previous


def test_h0_is_used_at_the_first_clob_decision(synthetic_cfg):
    env = new_env(synthetic_cfg)
    env.reset(seed=1)
    _, _, _, _, info = env.step(ClobAction(5.0, 2))
    assert info["t"] == 0.0
    assert info["H_used"] == synthetic_cfg.algo1.H0
