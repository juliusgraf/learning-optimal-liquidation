"""H_cl timing-convention tests (rulings D1, D2 — CLAUDE.md "CRITICAL").

Strategy: CRN twin envs (same seed; the env's draws are policy-independent,
ruling D10) driven identically except for ONE difference at time t — an
injected exogenous order "arriving at t", a different agent action, or a
flipped cancel bit. The time-t reward and the H^cl in the time-t state must
be IDENTICAL across the twins; only the time-(t+1) values may move.

Twin tests in the auction use the ``crn_synthetic_cfg`` fixture
(p2 = p4 = 0), which removes the only draws conditional on ledger contents
so injections cannot desynchronize the streams (see tests/helpers.py).
"""

from __future__ import annotations

import math

import pytest

from lmm.env.action_spaces import AuctionAction, ClobAction

from helpers import NOOP_CLOB, drive_to_auction, new_env, step_pair

AUCTION_HOLD = AuctionAction(2.0, 2, 0)


def auction_twins(cfg, seed: int, warmup_steps: int = 0):
    env_a, env_b = new_env(cfg), new_env(cfg)
    drive_to_auction(env_a, seed=seed)
    drive_to_auction(env_b, seed=seed)
    for _ in range(warmup_steps):
        step_pair(env_a, env_b, AUCTION_HOLD)
    return env_a, env_b


# ---------------------------------------------------------------------------
# D1: the auction estimate is end-of-(t-1); nothing sampled/decided at t
# enters the time-t state or reward
# ---------------------------------------------------------------------------


def test_d1_exogenous_arrival_at_t_enters_only_t_plus_1(crn_synthetic_cfg):
    """A large exogenous MM arriving at time t leaves the time-t H_cl and
    reward unchanged; it moves the time-(t+1) H_cl and reward (ruling D1)."""
    env_a, env_b = auction_twins(crn_synthetic_cfg, seed=5, warmup_steps=3)

    # The order "arrives at time t": present during step t, absent at the
    # end of t-1 when the time-t estimate was cached.
    env_b.generator.auction_flow.inject_market_maker(K=50.0, S=env_b.s_mid - 0.10)

    (r_a, ia), (r_b, ib) = step_pair(env_a, env_b, AUCTION_HOLD)
    assert ia["H_used"] == ib["H_used"]  # time-t state identical
    assert r_a == r_b  # time-t reward identical
    assert ia["H_next"] != ib["H_next"]  # end-of-t recompute sees the arrival

    (r_a2, ia2), (r_b2, ib2) = step_pair(env_a, env_b, AUCTION_HOLD)
    assert ia2["H_used"] == ia["H_next"] and ib2["H_used"] == ib["H_next"]
    assert ia2["H_used"] != ib2["H_used"]  # the arrival is in the t+1 state
    assert r_a2 != r_b2  # and in the t+1 reward


def test_d1_buy_taker_arrival_at_t_enters_only_t_plus_1(crn_synthetic_cfg):
    env_a, env_b = auction_twins(crn_synthetic_cfg, seed=6, warmup_steps=2)
    env_b.generator.auction_flow.inject_taker(+1, 200.0)
    (r_a, ia), (r_b, ib) = step_pair(env_a, env_b, AUCTION_HOLD)
    assert ia["H_used"] == ib["H_used"] and r_a == r_b
    assert ib["H_next"] > ia["H_next"]  # buy volume raises the estimate (D3)


def test_d1_cancel_changes_only_next_estimate(crn_synthetic_cfg):
    """theta predictability: flipping c_t leaves the time-t estimate
    unchanged (reward differs by exactly the cancel cost d_t) and changes
    the time-(t+1) estimate (rulings D1, D4)."""
    env_a, env_b = auction_twins(crn_synthetic_cfg, seed=9, warmup_steps=2)
    assert env_b._ledger.cancel_admissible()

    t = int(env_a.t)
    g = env_a.grid
    d_t = (t - g.tau_op) * env_a.cfg.reward.d
    assert d_t > 0.0

    (r_a, ia), (r_b, ib) = step_pair(
        env_a, env_b, AuctionAction(2.0, 2, 0), AuctionAction(2.0, 2, 1)
    )
    assert ia["H_used"] == ib["H_used"]  # time-t estimate unchanged by c_t
    assert r_a - r_b == pytest.approx(d_t, rel=1e-12)  # only the cost differs
    assert ia["H_next"] != ib["H_next"]  # theta_{t+1} embeds c_t

    (r_a2, ia2), (r_b2, ib2) = step_pair(env_a, env_b, AUCTION_HOLD)
    assert ia2["H_used"] != ib2["H_used"]
    assert r_a2 != r_b2


def test_order_at_t_minus_1_always_live_in_time_t_estimate(crn_synthetic_cfg):
    """Submit AND cancel-all at the same step t: the cancel kills orders
    strictly before t only, so the time-(t+1) estimate contains exactly the
    time-t order. Verified against an independent closed-form recompute
    (corrected Eq. (2), Prop. linear)."""
    env = new_env(crn_synthetic_cfg)
    drive_to_auction(env, seed=21)
    env.step(AuctionAction(3.0, 1, 0))  # t = tau_op: first order

    _, _, _, _, info = env.step(AuctionAction(5.0, 4, 1))  # submit + cancel

    g = env.grid
    flow = env.generator.auction_flow
    K_exo, S_exo = flow.supply_curves()
    s_a = g.alpha * (math.floor(env.s_mid / g.alpha) + 4)
    num = float(K_exo @ S_exo) + 5.0 * s_a + flow.net_market_volume()
    den = float(K_exo.sum()) + 5.0
    assert info["H_next"] == pytest.approx(num / den, rel=1e-12)


def test_final_order_uncancellable_and_eq1_is_terminal_estimate(synthetic_cfg):
    """The t_m order can never be cancelled (c_{t_m} kills s < m only), and
    the j = m+1 estimate (the end-of-t_m recompute) IS corrected Eq. (1):
    S_cl, Z verified against independent closed forms (rulings D1, D3)."""
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=33)
    m = env.grid.tau_cl - 1
    info = None
    while True:
        cancel = 1 if int(env.t) == m and env._ledger.cancel_admissible() else 0
        _, _, term, _, info = env.step(AuctionAction(2.0, 2, cancel))
        if term:
            break
    assert info["action"].cancel == 1  # the cancel variant actually ran

    K_live, S_live = env._ledger.live_orders()
    assert len(K_live) == 1  # only the t_m order survived its own cancel

    g = env.grid
    flow = env.generator.auction_flow
    K_exo, S_exo = flow.supply_curves()
    s_a = g.alpha * (math.floor(env.s_mid / g.alpha) + 2)
    expected_s_cl = (float(K_exo @ S_exo) + 2.0 * s_a + flow.net_market_volume()) / (
        float(K_exo.sum()) + 2.0
    )
    assert info["S_cl"] == pytest.approx(expected_s_cl, rel=1e-12)
    assert info["H_next"] == info["S_cl"]  # j = m+1 estimate == Eq. (1) output
    assert info["Z"] == pytest.approx(2.0 * (info["S_cl"] - s_a), rel=1e-12)


# ---------------------------------------------------------------------------
# D2: the CLOB H used at time t predates the time-t order flow and action
# ---------------------------------------------------------------------------


def test_d2_time_t_action_invisible_in_time_t_h(synthetic_cfg):
    """The agent's time-t order does not affect the H used in the time-t
    reward; its unexecuted remainder DOES enter the end-of-t Algorithm-1
    snapshot, hence the time-(t+1) H (ruling D2).

    The twins diverge at t = 0: a once-observed tick among n >= 2 snapshots
    has K_hat = max(0, v(2-n)/n)/alpha = 0 (clamped), so a single deep order
    placed later is invisible to Algorithm 1 — only at the FIRST snapshot
    does a fresh tick contribute (K_hat = v/alpha > 0)."""
    env_a, env_b = new_env(synthetic_cfg), new_env(synthetic_cfg)
    env_a.reset(seed=3)
    env_b.reset(seed=3)

    # Different time-0 actions: no order vs a large deep order (level Lc, one
    # past the exogenous book — never executed, so it survives as remainder).
    (_, ia), (_, ib) = step_pair(env_a, env_b, NOOP_CLOB, ClobAction(30.0, 12))
    assert ia["H_used"] == ib["H_used"] == synthetic_cfg.grid.S0
    assert ib["E_t"] == 0.0
    assert ia["H_next"] != ib["H_next"]

    (_, ia2), (_, ib2) = step_pair(env_a, env_b, NOOP_CLOB)
    assert ia2["H_used"] == ia["H_next"] and ib2["H_used"] == ib["H_next"]
    assert ia2["H_used"] != ib2["H_used"]


def test_d2_reward_at_t0_uses_h0(synthetic_cfg):
    """H_0 = configured initial value (= initial mid); the reward at t = 0
    uses H_0 (ruling D2)."""
    env = new_env(synthetic_cfg)
    env.reset(seed=1)
    _, _, _, _, info = env.step(ClobAction(5.0, 2))
    assert info["t"] == 0.0
    assert info["H_used"] == synthetic_cfg.grid.S0


def test_h_cache_chain_full_episode(synthetic_cfg):
    """Cache contract across the whole episode and the phase junction: the
    H used at every decision time t equals the H computed at the end of
    step t-1 (rulings D1, D2)."""
    env = new_env(synthetic_cfg)
    env.reset(seed=8)
    prev_next = env.h_cl  # == H_0
    while True:
        if env.phase == "clob":
            a = ClobAction(min(5.0, env.inventory), 2)
        else:
            a = AUCTION_HOLD
        _, _, term, _, info = env.step(a)
        assert info["H_used"] == prev_next
        prev_next = info["H_next"]
        if term:
            break
    assert info["S_cl"] == prev_next  # terminal clearing = last recompute
