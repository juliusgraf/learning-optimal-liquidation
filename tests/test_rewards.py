"""Reward tests: hand-computed closed-form cases for all three regimes and
an env-vs-formula sweep (rulings D1, D2, D4, D5, D8; paper `sec:MDP`).
"""

from __future__ import annotations

import numpy as np
import pytest

from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.rewards import auction_reward, clob_reward, f_a, f_c, terminal_reward
from lmm.market.auction import AgentOrderLedger

from helpers import load_synthetic_cfg, new_env

K_STAR, ALPHA, Q, D, LAMBDA = 1000, 0.01, 1.0, 0.1, 0.5  # base.yaml values


def test_config_carries_the_closed_form_constants(synthetic_cfg):
    r = synthetic_cfg.reward
    assert (r.k_star, synthetic_cfg.grid.alpha, r.q, r.d, r.lambda_inv) == (
        K_STAR, ALPHA, Q, D, LAMBDA
    )


# ---------------------------------------------------------------------------
# CLOB regime (ruling D5: paper EXACTLY, no clamp)
# ---------------------------------------------------------------------------


def test_f_c_closed_form():
    assert f_c(5.0, K_STAR, ALPHA) == 0.5  # u_+ / (k* alpha) = 5/10
    assert f_c(-3.0, K_STAR, ALPHA) == 0.0
    assert f_c(15.0, K_STAR, ALPHA) == 1.5  # NO clamp at 1 (D5)


def test_clob_reward_hand_case():
    # S = 100.02, E = 4, H = 100.00: u = k*alpha - (H - S) = 10.02
    # r = 100.02 * 4 * 10.02/10
    assert clob_reward(100.02, 4.0, 100.0, K_STAR, ALPHA) == pytest.approx(
        100.02 * 4.0 * 1.002, rel=1e-14
    )


def test_clob_reward_multiplier_exceeds_one_when_s_above_h():
    """S^bullet > H^cl => f_c(...) > 1 by definition; the legacy clamp at 1
    (main.py:545) was wrong (ruling D5)."""
    s, e, h = 100.50, 2.0, 100.0
    r = clob_reward(s, e, h, K_STAR, ALPHA)
    multiplier = r / (s * e)
    assert multiplier == pytest.approx(1.05, rel=1e-14)
    assert multiplier > 1.0


def test_clob_reward_zero_beyond_k_star_ticks():
    # H - S >= k*alpha => u <= 0 => reward 0.
    assert clob_reward(89.99, 5.0, 100.0, K_STAR, ALPHA) == 0.0


# ---------------------------------------------------------------------------
# Auction regime (ruling D4: scalar cancel, cost d_t * c_t)
# ---------------------------------------------------------------------------


def test_auction_reward_right_side():
    # u = K H (H - S) = 2*100*0.1 = 20 > 0 => f_a = 0.
    assert auction_reward(2.0, 99.9, 100.0, Q, 0.0, 0) == pytest.approx(20.0, rel=1e-12)


def test_auction_reward_wrong_side_penalty():
    # u = 2*100*(-0.2) = -40 => r = u + f_a(u) = -40 - 40 = -80 (q = 1).
    assert f_a(-40.0, Q) == -40.0
    assert auction_reward(2.0, 100.2, 100.0, Q, 0.0, 0) == pytest.approx(-80.0, rel=1e-12)


def test_auction_cancel_cost_is_d_t_times_c():
    """Cost = d_t * c_t with d_t = (t - n - 1) d — NOT d times the number of
    orders cancelled (legacy was wrong; ruling D4)."""
    t, tau_op = 135, 120
    d_t = (t - tau_op) * D
    assert d_t == pytest.approx(1.5)
    base = auction_reward(2.0, 99.9, 100.0, Q, d_t, 0)
    with_cancel = auction_reward(2.0, 99.9, 100.0, Q, d_t, 1)
    assert base - with_cancel == pytest.approx(d_t, rel=1e-12)


def test_abstain_with_cancel_costs_only_d_t():
    # K^a = 0 (abstain) with c = 1: r = -d_t.
    assert auction_reward(0.0, 0.0, 100.0, Q, 0.7, 1) == pytest.approx(-0.7)


# ---------------------------------------------------------------------------
# Terminal regime (rulings D3, D8) with a partially cancelled ledger
# ---------------------------------------------------------------------------


def test_terminal_reward_hand_case_with_partially_cancelled_ledger():
    """Ledger: orders at t = 120, 121, 122; cancel-all decided at t = 122
    kills 120 and 121 only... here instead: cancel at 121 kills 120; orders
    121, 122 live. theta = (1, 0, 0, ...). Hand-computed terminal value."""
    cfg = load_synthetic_cfg()
    ledger = AgentOrderLedger(cfg.grid)
    ledger.reset()
    ledger.submit(120, 1.0, 100.3)  # will be cancelled
    ledger.submit(121, 2.0, 99.9)
    ledger.apply_cancel_all(121)  # c_{121} = 1: kills the t=120 order only
    ledger.submit(122, 3.0, 100.1)

    theta = ledger.theta()
    np.testing.assert_array_equal(theta[:3], [1.0, 0.0, 0.0])
    K_live, S_live = ledger.live_orders()
    np.testing.assert_allclose(K_live, [2.0, 3.0])
    np.testing.assert_allclose(S_live, [99.9, 100.1])

    s_cl, i_tau_op = 100.0, 0.0
    # Z = 2(100-99.9) + 3(100-100.1) = -0.1 ; I_final = 0 - Z = 0.1.
    z = float(np.sum(K_live * (s_cl - S_live)))
    assert z == pytest.approx(-0.1, rel=1e-9)
    i_final = i_tau_op - z
    # u = (2*100*0.1, 3*100*(-0.1)) = (20, -30):
    # r = (20 - 30) - 0.5*0.1^2 + (0 + f_a(-30)) = -10 - 0.005 - 30.
    r = terminal_reward(K_live, S_live, s_cl, i_final, LAMBDA, Q)
    assert r == pytest.approx(-40.005, rel=1e-12)


def test_terminal_reward_no_inventory_clipping():
    """D8: the |I|^2 penalty applies to the RAW final inventory, however
    large (legacy clipped at +-1e6 scale guards; removed)."""
    huge = 1e7
    r = terminal_reward(np.array([]), np.array([]), 100.0, huge, LAMBDA, Q)
    assert r == -LAMBDA * huge**2


# ---------------------------------------------------------------------------
# Env-vs-formula: every step's reward equals the closed form on its own info
# ---------------------------------------------------------------------------


def fixed_policy(env, cancel_at: int | None):
    if env.phase == "clob":
        return ClobAction(min(5.0, env.inventory), 2)
    cancel = int(
        cancel_at is not None
        and int(env.t) == cancel_at
        and env._ledger.cancel_admissible()
    )
    return AuctionAction(2.0, 2, cancel)


@pytest.mark.parametrize("cancel_at", [None, 135])
def test_env_rewards_match_closed_forms(synthetic_cfg, cancel_at):
    env = new_env(synthetic_cfg)
    env.reset(seed=4)
    r_cfg = synthetic_cfg.reward
    alpha = synthetic_cfg.grid.alpha
    i_tau_op = None
    saw_cancel = False
    while True:
        a = fixed_policy(env, cancel_at)
        _, r, term, _, info = env.step(a)
        if info["phase"] == "clob":
            expected = clob_reward(
                info["S_bullet"], info["E_t"], info["H_used"], r_cfg.k_star, alpha
            )
        else:
            if i_tau_op is None:
                i_tau_op = env.inventory  # frozen during the auction
            expected = auction_reward(
                a.K_a, info["S_a"], info["H_used"], r_cfg.q, info["d_t"], a.cancel
            )
            saw_cancel = saw_cancel or a.cancel == 1
            if term:
                expected += info["terminal_reward"]
        assert r == expected  # identical float path
        if term:
            break
    assert saw_cancel == (cancel_at is not None)

    # Terminal diagnostics are internally consistent (D8: no clipping).
    assert info["I_final"] == pytest.approx(i_tau_op - info["Z"], rel=1e-12)
    K_live, S_live = env._ledger.live_orders()
    assert info["terminal_reward"] == terminal_reward(
        K_live, S_live, info["S_cl"], info["I_final"], r_cfg.lambda_inv, r_cfg.q
    )
    assert info["Z"] == pytest.approx(
        float(np.sum(K_live * (info["S_cl"] - S_live))), rel=1e-12
    )


def test_numerical_guard_never_binds_on_standard_runs():
    """CLAUDE.md (D8): with the guard ON at the default far-out bound, a
    seeded standard episode is bit-identical to the guard-OFF run."""
    cfg_off = load_synthetic_cfg()
    cfg_on = load_synthetic_cfg("reward.numerical_guard=true")

    def run(cfg):
        env = new_env(cfg)
        env.reset(seed=12)
        rewards = []
        while True:
            _, r, term, _, info = env.step(fixed_policy(env, cancel_at=None))
            rewards.append(r)
            if term:
                return rewards, info["I_final"]

    r_off, i_off = run(cfg_off)
    r_on, i_on = run(cfg_on)
    assert r_on == r_off and i_on == i_off


# ---------------------------------------------------------------------------
# One-sided benchmark order variants (ruling D16; reward treatment author-
# confirmed as ruling D18 — see audit/AUDIT.md)
# ---------------------------------------------------------------------------


def test_auction_reward_one_sided_positive_gap_matches_linear():
    # H > S^a: the hockey stick coincides with the linear curve above the kink
    linear = auction_reward(2.0, 100.0, 101.0, Q, 0.0, 0)
    one_sided = auction_reward(2.0, 100.0, 101.0, Q, 0.0, 0, one_sided=True)
    assert one_sided == pytest.approx(linear)
    assert one_sided == pytest.approx(2.0 * 101.0 * 1.0)  # u >= 0: f_a inactive


def test_auction_reward_one_sided_negative_gap_is_zero():
    # H < S^a: supply (H - S^a)_+ = 0 -> u = 0 and NO wrong-side penalty,
    # while the linear order would be penalized
    assert auction_reward(2.0, 102.0, 101.0, Q, 0.0, 0, one_sided=True) == 0.0
    assert auction_reward(2.0, 102.0, 101.0, Q, 0.0, 0) < 0.0


def test_terminal_reward_one_sided_mask():
    K = np.array([1.0, 3.0])
    S = np.array([99.0, 101.0])
    s_cl = 100.0
    # order 1 (one-sided) is below its kink: contributes nothing, no penalty
    r = terminal_reward(K, S, s_cl, 0.0, 0.0, Q, one_sided=np.array([False, True]))
    assert r == pytest.approx(1.0 * s_cl * 1.0)
    # linear treatment of order 1 would add u = 3*100*(-1) and f_a(u) = -Q*300
    r_linear = terminal_reward(K, S, s_cl, 0.0, 0.0, Q)
    assert r_linear == pytest.approx(100.0 - 300.0 - Q * 300.0)
