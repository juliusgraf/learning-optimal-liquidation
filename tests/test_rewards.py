"""Closed-form tests for the revised shaped and economic reward contracts."""

from __future__ import annotations

import pytest

from helpers import load_synthetic_cfg, new_env
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.rewards import auction_reward, clob_reward, f_a, f_c, terminal_reward


K_STAR, ALPHA, Q, D, LAMBDA = 1000, 0.01, 1.0, 0.1, 0.5


def test_config_carries_closed_form_constants(synthetic_cfg):
    r = synthetic_cfg.reward
    assert (r.k_star, synthetic_cfg.grid.alpha, r.q, r.d, r.lambda_inv) == (
        K_STAR,
        ALPHA,
        Q,
        D,
        LAMBDA,
    )


def test_f_c_is_positive_part_ratio_clamped_to_one():
    assert f_c(5.0, K_STAR, ALPHA) == 0.5
    assert f_c(-3.0, K_STAR, ALPHA) == 0.0
    assert f_c(15.0, K_STAR, ALPHA) == 1.0


def test_f_a_is_purchase_side_attenuation():
    assert f_a(10.0, 1.0) == 0.0
    assert f_a(-10.0, 0.25) == 2.5
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        f_a(-1.0, 1.1)


def test_clob_reward_hand_cases_and_unshaped_switch():
    # A quote above H receives cash, never a multiplier above one.
    cash = 100.02 * 4.0
    assert clob_reward(100.02, 4.0, 100.0, K_STAR, ALPHA) == pytest.approx(cash)
    # H-S=5 leaves half the k*alpha tolerance.
    assert clob_reward(95.0, 4.0, 100.0, K_STAR, ALPHA) == pytest.approx(95.0 * 4.0 * 0.5)
    assert clob_reward(80.0, 4.0, 100.0, K_STAR, ALPHA) == 0.0
    assert clob_reward(
        80.0, 4.0, 100.0, K_STAR, ALPHA, shaping_enabled=False
    ) == pytest.approx(320.0)


def test_auction_interim_reward_and_fee_hand_cases():
    positive = 2.0 * 100.0 * (100.0 - 99.9)
    assert auction_reward(2.0, 99.9, 100.0, Q, 0.0, 0) == pytest.approx(positive)
    # With q=1 a projected purchase is fully attenuated in the training signal.
    assert auction_reward(2.0, 100.1, 100.0, Q, 0.0, 0) == 0.0
    assert auction_reward(2.0, 99.9, 100.0, Q, 1.5, 1) == pytest.approx(
        positive - 1.5
    )
    # Unshaped and external-policy paths retain only the economic fee.
    assert auction_reward(
        2.0, 99.9, 100.0, Q, 1.5, 1, shaping_enabled=False
    ) == -1.5
    assert auction_reward(
        2.0, 99.9, 100.0, Q, 1.5, 1, external_policy=True
    ) == -1.5


def test_terminal_reward_uses_actual_aggregate_fill_once():
    # Actual Z=-2 is a purchase: cash=-202, f_a=202 when q=1.  Residual
    # inventory is marked at the frozen exogenous mid, not at clearing.
    out = terminal_reward(101.0, -2.0, 12.0, 100.0, 0.5, 1.0)
    assert out == pytest.approx(-202.0 + 202.0 + 1_200.0 - 0.5 * 12.0**2)
    unshaped = terminal_reward(
        101.0,
        -2.0,
        12.0,
        100.0,
        0.5,
        1.0,
        shaping_enabled=False,
    )
    assert unshaped == pytest.approx(-202.0 + 1_200.0 - 0.5 * 12.0**2)


def test_terminal_inventory_penalty_is_not_clipped():
    huge = 1.0e7
    out = terminal_reward(100.0, 0.0, huge, 0.0, LAMBDA, Q)
    assert out == -LAMBDA * huge**2


def test_environment_step_rewards_match_pure_formulas(synthetic_cfg):
    env = new_env(synthetic_cfg)
    env.reset(seed=4)
    while True:
        if env.phase == "clob":
            volume = float(min(5, int(env.inventory)))
            action = ClobAction(volume, 2 if volume else 0)
        else:
            action = AuctionAction(2.0, 2, 0)
        _, reward, done, _, info = env.step(action)
        if info["phase"] == "clob":
            expected = clob_reward(
                info["S_bullet"],
                info["E_t"],
                info["H_used"],
                synthetic_cfg.reward.k_star,
                synthetic_cfg.grid.alpha,
            )
        else:
            expected = auction_reward(
                action.K_a,
                info["S_a"],
                info["H_used"],
                synthetic_cfg.reward.q,
                info["d_t"],
                action.cancel,
            )
            if done:
                expected += info["terminal_reward"]
                assert info["terminal_reward"] == pytest.approx(
                    terminal_reward(
                        info["S_cl"],
                        info["Z"],
                        info["I_final"],
                        env.s_mid,
                        synthetic_cfg.reward.lambda_inv,
                        synthetic_cfg.reward.q,
                    )
                )
        assert reward == pytest.approx(expected)
        if done:
            break


def test_numerical_guard_is_diagnostic_only():
    cfg_off = load_synthetic_cfg()
    cfg_on = load_synthetic_cfg("reward.numerical_guard=true")

    def run(cfg):
        env = new_env(cfg)
        env.reset(seed=12)
        rewards = []
        while True:
            action = (
                ClobAction(0.0, 0)
                if env.phase == "clob"
                else AuctionAction(2.0, 2, 0)
            )
            _, reward, done, _, info = env.step(action)
            rewards.append(reward)
            if done:
                return rewards, info["I_final"]

    assert run(cfg_on) == run(cfg_off)
