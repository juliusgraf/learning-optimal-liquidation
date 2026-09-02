"""Sign-convention tests (ruling D3; CLAUDE.md BINDING conventions).

zeta = + is the BUY side for market orders: buy market volume weakly RAISES
the clearing price, sell volume weakly LOWERS it; N^+ counts BUYING market
orders end to end (naming audit of the new code — the legacy `N_plus`
counted sells). Also pins the paper's aggregate E_t execution formula
against the sequential book walk (Sec. 2.1.1, agent priority at her level).
"""

from __future__ import annotations

import numpy as np
import pytest

from lmm.env.action_spaces import AuctionAction
from lmm.market.clearing import ClearingInputs, solve_linear_clearing
from lmm.market.clob import OrderBook, executed_volume

from helpers import drive_to_auction, new_env, step_pair

AUCTION_HOLD = AuctionAction(0.0, 0, 0)


# ---------------------------------------------------------------------------
# Closed-form monotonicity (corrected Prop. linear)
# ---------------------------------------------------------------------------


def make_inputs(net: float) -> ClearingInputs:
    return ClearingInputs(
        K_exo=np.array([1.0, 2.0]),
        S_exo=np.array([100.0, 100.5]),
        K_agent=np.array([0.5]),
        S_agent=np.array([99.5]),
        net_market_volume=net,
        fallback_mid=100.0,
    )


def test_buy_volume_raises_sell_volume_lowers_closed_form():
    p0, _ = solve_linear_clearing(make_inputs(0.0))
    p_buy, _ = solve_linear_clearing(make_inputs(+10.0))
    p_sell, _ = solve_linear_clearing(make_inputs(-10.0))
    assert p_sell < p0 < p_buy


@pytest.mark.parametrize("trial", range(20))
def test_monotone_in_net_market_volume_random(trial):
    """p* is (weakly) nondecreasing in the net buy volume on random
    instances; strict whenever the aggregate slope is positive."""
    rng = np.random.default_rng(100 + trial)
    n_exo, n_agent = int(rng.integers(1, 6)), int(rng.integers(0, 4))
    base = dict(
        K_exo=rng.uniform(0.1, 5.0, n_exo),
        S_exo=rng.uniform(95.0, 105.0, n_exo),
        K_agent=rng.uniform(0.1, 5.0, n_agent),
        S_agent=rng.uniform(95.0, 105.0, n_agent),
        fallback_mid=100.0,
    )
    net = float(rng.uniform(-50.0, 50.0))
    p_lo, _ = solve_linear_clearing(ClearingInputs(net_market_volume=net, **base))
    p_hi, _ = solve_linear_clearing(ClearingInputs(net_market_volume=net + 5.0, **base))
    assert p_hi > p_lo


# ---------------------------------------------------------------------------
# Env-level: injected takers move the estimate the right way
# ---------------------------------------------------------------------------


def taker_twins(cfg, seed: int, side: int, volume: float):
    env_a, env_b = new_env(cfg), new_env(cfg)
    drive_to_auction(env_a, seed=seed)
    drive_to_auction(env_b, seed=seed)
    for _ in range(2):
        step_pair(env_a, env_b, AUCTION_HOLD)
    env_b._generator.auction_flow.inject_taker(side, volume)
    return step_pair(env_a, env_b, AUCTION_HOLD)


def test_injected_buy_taker_raises_estimate(crn_synthetic_cfg):
    (_, ia), (_, ib) = taker_twins(crn_synthetic_cfg, seed=17, side=+1, volume=100.0)
    assert ib["H_next"] > ia["H_next"]


def test_injected_sell_taker_lowers_estimate(crn_synthetic_cfg):
    (_, ia), (_, ib) = taker_twins(crn_synthetic_cfg, seed=17, side=-1, volume=100.0)
    assert ib["H_next"] < ia["H_next"]


def test_injected_buy_taker_raises_terminal_clearing(crn_synthetic_cfg):
    """Buy market volume weakly raises the FINAL clearing price S_cl too."""
    env_a, env_b = new_env(crn_synthetic_cfg), new_env(crn_synthetic_cfg)
    drive_to_auction(env_a, seed=29)
    drive_to_auction(env_b, seed=29)
    env_b._generator.auction_flow.inject_taker(+1, 100.0)
    info_a = info_b = None
    while True:
        (_, info_a), (_, info_b) = step_pair(env_a, env_b, AUCTION_HOLD)
        if "S_cl" in info_a:
            break
    assert info_b["S_cl"] > info_a["S_cl"]


# ---------------------------------------------------------------------------
# Naming audit: N^+ counts BUYING market orders end to end
# ---------------------------------------------------------------------------


def test_n_buy_counts_buy_taker_events(synthetic_cfg):
    """Paper X7/X8 are cumulative accepted-arrival counts.

    The next decision's proposals are processed before ``step`` returns, so
    ``info['proposal_counts']`` is the snapshot for the action just taken while
    ``env.n_buy``/``env.n_sell`` already describe the returned next state.
    """
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=11)
    while True:
        n_buy_before, n_sell_before = env.n_buy, env.n_sell
        _, _, term, _, info = env.step(AUCTION_HOLD)
        counts = info["proposal_counts"]
        assert counts["buy_arrival"]["accepted"] == n_buy_before
        assert counts["sell_arrival"]["accepted"] == n_sell_before
        assert env.n_buy >= n_buy_before
        assert env.n_sell >= n_sell_before
        ps = env.paper_state()
        assert ps["X7"] == env.n_buy and ps["X8"] == env.n_sell
        if term:
            break


# ---------------------------------------------------------------------------
# E_t: sequential book walk == the paper's aggregate formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(30))
def test_executed_volume_formula_equals_sequential_walk(synthetic_cfg, trial):
    """E_t = max(0, min(v_t, sum nu^+ - sum_{j<delta} V^{+,j})) over one
    step's buy market orders and the start-of-step ask book equals the
    total filled by sequential `process_buy_market_order` calls (agent
    priority at her own level; AUDIT A.2)."""
    rng = np.random.default_rng(7000 + trial)
    book = OrderBook(synthetic_cfg.clob_flow, synthetic_cfg.grid)
    lc = synthetic_cfg.clob_flow.Lc
    ask0 = rng.uniform(0.0, 8.0, lc)
    book.ask_volumes = ask0.copy()
    book.bid_volumes = rng.uniform(0.0, 8.0, lc)

    v_agent = float(rng.uniform(0.0, 25.0))
    delta = int(rng.integers(0, lc + 1))  # incl. level Lc (one past the book)
    book.place_agent_order(v_agent, delta)

    buys = rng.uniform(0.0, 6.0, size=int(rng.integers(0, 6)))
    executed = sum(book.process_buy_market_order(v) for v in buys)

    assert executed == pytest.approx(
        executed_volume(v_agent, delta, ask0, buys), abs=1e-12
    )


def test_agent_priority_at_her_level(synthetic_cfg):
    """At the agent's own tick the agent fills BEFORE the exogenous volume
    (execution priority, Sec. 2.1.1)."""
    book = OrderBook(synthetic_cfg.clob_flow, synthetic_cfg.grid)
    book.ask_volumes = np.zeros(synthetic_cfg.clob_flow.Lc)
    book.ask_volumes[0] = 5.0  # exogenous volume at the agent's level
    book.bid_volumes = np.zeros(synthetic_cfg.clob_flow.Lc)
    book.place_agent_order(3.0, 0)
    executed = book.process_buy_market_order(2.0)
    assert executed == 2.0  # all 2.0 to the agent, none to the exogenous level
    assert book.ask_volumes[0] == 5.0
