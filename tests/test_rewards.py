"""Closed-form tests for the revised shaped and economic reward contracts."""

from __future__ import annotations

import pytest

from helpers import drive_to_auction, load_synthetic_cfg, new_env
from lmm.config import economic_evaluation_config
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.rewards import (
    auction_fictive_reward,
    auction_reward,
    clob_reward,
    f_a,
    f_c,
    terminal_reward,
)


K_STAR, ALPHA, Q, D, LAMBDA = 1000, 0.01, 1.0, 0.1, 0.5


def test_config_carries_closed_form_constants(synthetic_cfg):
    r = synthetic_cfg.reward
    assert (r.k_star, synthetic_cfg.grid.alpha, r.q, r.d, r.lambda_inv) == (
        K_STAR,
        ALPHA,
        Q,
        D,
        2.0,
    )
    assert r.shaping_enabled
    assert r.clawback_shaping
    assert r.center_initial_inventory_value


def test_phase_specific_shaping_switches_fall_back_to_shared_contract():
    baseline = load_synthetic_cfg()
    assert baseline.reward.effective_clob_shaping is True
    assert baseline.reward.effective_auction_shaping is True

    unshaped = load_synthetic_cfg("reward.shaping_enabled=false")
    assert unshaped.reward.effective_clob_shaping is False
    assert unshaped.reward.effective_auction_shaping is False

    split = load_synthetic_cfg(
        "reward.clob_shaping_enabled=false",
        "reward.auction_shaping_enabled=true",
        "reward.shaping_enabled=true",
    )
    assert split.reward.shaping_enabled is True
    assert split.reward.effective_clob_shaping is False
    assert split.reward.effective_auction_shaping is True


def test_economic_evaluation_contract_disables_all_shaping_overrides():
    training = load_synthetic_cfg(
        "reward.clob_shaping_enabled=true",
        "reward.auction_shaping_enabled=true",
    )
    evaluation = economic_evaluation_config(training)
    assert training.reward.effective_clob_shaping
    assert training.reward.effective_auction_shaping
    assert not evaluation.reward.effective_clob_shaping
    assert not evaluation.reward.effective_auction_shaping
    assert not evaluation.reward.clawback_shaping
    assert evaluation.reward.center_initial_inventory_value


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


def test_auction_reward_subtracts_the_exact_signed_cancelled_shaping():
    current = auction_fictive_reward(2.0, 99.9, 100.0, Q)
    cancelled = auction_fictive_reward(3.0, 99.8, 100.0, Q)
    assert auction_reward(
        2.0,
        99.9,
        100.0,
        Q,
        1.5,
        1,
        cancelled_interim_shaping=cancelled,
    ) == pytest.approx(current - cancelled - 1.5)
    with pytest.raises(ValueError, match="requires cancel=1"):
        auction_reward(
            2.0,
            99.9,
            100.0,
            Q,
            0.0,
            0,
            cancelled_interim_shaping=cancelled,
        )
    with pytest.raises(ValueError, match="shaping is disabled"):
        auction_reward(
            2.0,
            99.9,
            100.0,
            Q,
            1.5,
            1,
            shaping_enabled=False,
            cancelled_interim_shaping=cancelled,
        )


def test_single_replace_clawback_telescopes_to_the_surviving_credit_and_fees():
    cfg = load_synthetic_cfg(
        "reward.center_initial_inventory_value=false",
        "actions.auction_order_mode=single_replace",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=19)

    _, r1, _, _, i1 = env.step(AuctionAction(2.0, -10, 0))
    phi1 = i1["auction_interim_shaping"]
    assert phi1 > 0.0
    assert i1["auction_shaping_clawback"] == 0.0
    assert r1 == pytest.approx(phi1)

    _, r2, _, _, i2 = env.step(AuctionAction(4.0, -9, 1))
    phi2 = i2["auction_interim_shaping"]
    assert phi2 > 0.0
    assert i2["auction_shaping_clawback"] == pytest.approx(phi1)
    assert r2 == pytest.approx(phi2 - phi1 - i2["cancellation_fee"])
    assert env.own_slope == pytest.approx(4.0)  # current replacement remains live

    _, r3, _, _, i3 = env.step(AuctionAction(0.0, 0, 1))
    assert i3["auction_interim_shaping"] == 0.0
    assert i3["auction_shaping_clawback"] == pytest.approx(phi2)
    assert r3 == pytest.approx(-phi2 - i3["cancellation_fee"])
    assert not env.cancel_admissible

    # Every canceled phi appears once with + sign and once with - sign.
    assert r1 + r2 + r3 == pytest.approx(
        -i2["cancellation_fee"] - i3["cancellation_fee"]
    )


def test_cancel_all_claws_back_every_live_credit_in_multi_order_mode():
    cfg = load_synthetic_cfg(
        "reward.center_initial_inventory_value=false",
        "actions.auction_order_mode=multi",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    _, r1, _, _, i1 = env.step(AuctionAction(1.0, -10, 0))
    _, r2, _, _, i2 = env.step(AuctionAction(2.0, -9, 0))
    _, r3, _, _, i3 = env.step(AuctionAction(0.0, 0, 1))
    phi1 = i1["auction_interim_shaping"]
    phi2 = i2["auction_interim_shaping"]
    assert i3["auction_shaping_clawback"] == pytest.approx(phi1 + phi2)
    assert r1 + r2 + r3 == pytest.approx(-i3["cancellation_fee"])


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


def test_environment_step_rewards_match_pure_formulas():
    cfg = load_synthetic_cfg(
        "reward.shaping_enabled=true",
        "reward.center_initial_inventory_value=false",
        "actions.auction_order_mode=multi",
    )
    env = new_env(cfg)
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
                cfg.reward.k_star,
                cfg.grid.alpha,
            )
        else:
            expected = auction_reward(
                action.K_a,
                info["S_a"],
                info["H_used"],
                cfg.reward.q,
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
                        cfg.reward.lambda_inv,
                        cfg.reward.q,
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
                else AuctionAction(0.0, 0, 0)
            )
            _, reward, done, _, info = env.step(action)
            rewards.append(reward)
            if done:
                return rewards, info["I_final"]

    assert run(cfg_on) == run(cfg_off)
