"""Admissibility tests: Adm(x) and the cancel-all constraint C(x)
(CLAUDE.md, time-free state-only constraint; rulings D4, D11).

C(x) = max_i (1 - x^{9,(i)}) * 1{x^{17,(i)} > 0}: a cancel-all is admissible
iff a live prior order with K^a > 0 exists. The env's internal-ledger mask
must agree with C evaluated on paper_state() (CLAUDE.md implementation
note), and the env must REJECT inadmissible submissions (masking, never
projection — AUDIT N12).
"""

from __future__ import annotations

import numpy as np
import pytest

from lmm.env.action_spaces import AuctionAction, ClobAction

from helpers import NOOP_AUCTION, drive_to_auction, load_synthetic_cfg, new_env


def C_of_paper_state(ps) -> float:
    """The CLAUDE.md formula, evaluated literally on X^9 and X^17."""
    x9, x17 = ps["X9"], ps["X17"]
    vals = (1.0 - x9) * (x17 > 0.0)
    return float(np.max(vals)) if len(vals) else 0.0


# ---------------------------------------------------------------------------
# CLOB-phase masks (a^1 <= x^1; quote constraint structural)
# ---------------------------------------------------------------------------


def test_clob_mask_counts_volume_constraint(synthetic_cfg):
    env = new_env(synthetic_cfg)
    env.reset(seed=0)
    grid = env.clob_grid
    n_deltas = (
        synthetic_cfg.actions.clob_delta_max - synthetic_cfg.actions.clob_delta_min + 1
    )
    # Full inventory: everything admissible.
    assert env.action_mask().sum() == len(grid) == 1 + 30 * n_deltas
    # Fractional inventory 3.5: no-op + v in {1,2,3}.
    assert grid.mask(3.5).sum() == 1 + 3 * n_deltas
    # Zero inventory: only the no-op.
    assert grid.mask(0.0).sum() == 1


def test_env_rejects_inadmissible_clob_actions(synthetic_cfg):
    env = new_env(synthetic_cfg)
    env.reset(seed=0)
    env._inventory = 5.0
    with pytest.raises(ValueError, match="inadmissible CLOB volume"):
        env.step(ClobAction(6.0, 2))
    with pytest.raises(ValueError, match="inadmissible CLOB volume"):
        env.step(ClobAction(synthetic_cfg.actions.clob_volume_max + 1.0, 2))
    with pytest.raises(ValueError, match="CLOB offset must be"):
        env.step(ClobAction(1.0, -1))
    with pytest.raises(TypeError):
        env.step("not an action")


# ---------------------------------------------------------------------------
# Cancel-all constraint C(x) through the auction phase
# ---------------------------------------------------------------------------


def assert_cancel_admissible(env, expected: bool):
    mask = env.action_mask()
    cancel_rows = np.array([a.cancel == 1 for a in env.auction_grid.actions])
    assert mask[cancel_rows].any() == expected
    if not expected:
        assert mask[~cancel_rows].all()
    elif env.cfg.actions.auction_order_mode == "single_replace":
        stacking = np.array(
            [a.cancel == 0 and a.K_a > 0.0 for a in env.auction_grid.actions]
        )
        assert not mask[stacking].any()
        assert mask[~(cancel_rows | stacking)].all()
    else:
        assert mask[~cancel_rows].all()
    assert env._ledger.cancel_admissible() == expected
    assert env.cancel_admissible == expected
    assert (C_of_paper_state(env.paper_state()) > 0) == expected


def test_cancel_forbidden_at_auction_open_and_after_abstain(synthetic_cfg):
    """C(x) = 0 at the auction open (property (i) of CLAUDE.md) and stays 0
    while the agent only abstains (K^a = 0 submissions are no entries)."""
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=2)
    assert_cancel_admissible(env, False)  # t = n+1: c_{t_{n+1}} = 0 forced
    assert len(env.auction_grid) == 254
    assert env.action_mask().sum() == 127
    with pytest.raises(ValueError, match="cancel-all is ineligible"):
        env.step(AuctionAction(2.0, 2, 1))

    env.step(NOOP_AUCTION)  # abstain (K^a = 0)
    assert_cancel_admissible(env, False)
    env.step(NOOP_AUCTION)
    assert_cancel_admissible(env, False)


def test_cancel_allowed_after_live_submission_then_forbidden_again(synthetic_cfg):
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=2)
    env.step(AuctionAction(2.0, 1, 0))  # live prior order with K^a > 0
    assert_cancel_admissible(env, True)
    assert env.action_mask().sum() == 128

    # Cancel-all WITHOUT a new submission: ledger empty again afterwards.
    env.step(AuctionAction(0.0, 0, 1))
    assert_cancel_admissible(env, False)

    # Submit + cancel in one step: the same-step order survives (theta
    # recursion cutoff s < t), so the cancel stays admissible at t+1.
    env.step(AuctionAction(3.0, 2, 0))
    assert_cancel_admissible(env, True)
    env.step(AuctionAction(1.0, 0, 1))
    assert_cancel_admissible(env, True)  # the just-submitted K=1 order lives


def test_single_replace_mode_requires_cancellation_before_a_new_live_schedule():
    cfg = load_synthetic_cfg("actions.auction_order_mode=single_replace")
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    env.step(AuctionAction(cfg.actions.beta, 1, 0))

    mask = env.action_mask()
    stacking = np.array(
        [a.K_a > 0.0 and a.cancel == 0 for a in env.auction_grid.actions]
    )
    assert not mask[stacking].any()
    assert mask.sum() == 128  # two no-ops plus every cancel/replace action
    with pytest.raises(ValueError, match="single_replace mode requires cancel=1"):
        env.step(AuctionAction(cfg.actions.beta, 2, 0))

    env.step(AuctionAction(cfg.actions.beta, 2, 1))
    assert env.own_slope == pytest.approx(cfg.actions.beta)
    assert env.cancel_admissible


def test_indicative_centered_grid_maps_to_absolute_manuscript_offset():
    cfg = load_synthetic_cfg(
        "actions.beta=6.666666666666667",
        "actions.K_max=5",
        "actions.auction_slope_multipliers=[1,2,3,4,5]",
        "actions.B_max=150",
        "actions.auction_offset_center=indicative",
        "actions.auction_local_offset_max=10",
        "actions.auction_order_mode=single_replace",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    assert len(env.auction_grid) == 212

    # A template at local displacement -4 is executed at the manuscript
    # offset b=33 when the observed indicative price is 37 ticks above the
    # frozen mid.  Direct AuctionAction objects remain absolute b actions.
    env._h_cache = env.s_mid + 37 * cfg.grid.alpha
    idx = next(
        i
        for i, a in enumerate(env.auction_grid.actions)
        if a.K_a == pytest.approx(cfg.actions.beta)
        and a.offset == -4
        and a.cancel == 0
    )
    assert env.action_mask()[idx]
    assert env._decode_auction(idx) == AuctionAction(cfg.actions.beta, 33, 0)
    direct = AuctionAction(cfg.actions.beta, -4, 0)
    assert env._decode_auction(direct) is direct


def test_nonuniform_slope_subset_keeps_small_and_large_actions_compact():
    cfg = load_synthetic_cfg(
        "actions.beta=1.0",
        "actions.K_max=32",
        "actions.auction_slope_multipliers=[1,2,4,8,16,32]",
        "actions.B_max=150",
        "actions.auction_offset_center=indicative",
        "actions.auction_local_offset_max=10",
    )
    env = new_env(cfg)
    slopes = sorted({a.K_a for a in env.auction_grid.actions if a.K_a > 0.0})
    assert slopes == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert len(env.auction_grid) == 254


def test_indicative_centered_grid_masks_templates_outside_ambient_band():
    cfg = load_synthetic_cfg(
        "actions.B_max=30",
        "actions.auction_offset_center=indicative",
        "actions.auction_local_offset_max=10",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    env._h_cache = env.s_mid + 28 * cfg.grid.alpha
    mask = env.action_mask()
    outside = np.array(
        [a.K_a > 0.0 and a.offset > 2 for a in env.auction_grid.actions]
    )
    assert not mask[outside].any()


def test_cancel_admissibility_is_in_the_auction_observation(synthetic_cfg):
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=2)
    idx = synthetic_cfg.features.auction.index("cancel_admissible")
    assert env.features.auction_features(env)[idx] == 0.0
    obs, *_ = env.step(AuctionAction(2.0, 1, 0))
    assert obs[idx] == 1.0


def test_clob_phase_has_no_cancel_concept_and_C_is_zero(synthetic_cfg):
    """During the CLOB phase the auction ledger is empty, so C(x) = 0
    (property (i)); the CLOB grid has no cancel coordinate at all."""
    env = new_env(synthetic_cfg)
    env.reset(seed=5)
    while env.phase == "clob":
        assert C_of_paper_state(env.paper_state()) == 0.0
        assert not env._ledger.cancel_admissible()
        env.step(ClobAction(0.0, 0))


def test_env_rejects_inadmissible_auction_actions(synthetic_cfg):
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=3)
    with pytest.raises(ValueError, match="K\\^a must be finite and nonnegative"):
        env.step(AuctionAction(-1.0, 0, 0))
    with pytest.raises(ValueError, match="cancel must be 0 or 1"):
        env.step(AuctionAction(1.0, 0, 2))
    with pytest.raises(TypeError):
        env.step(ClobAction(1.0, 2))  # wrong phase's action type


# ---------------------------------------------------------------------------
# Internal-ledger mask == C(paper_state()) along a whole mixed episode
# ---------------------------------------------------------------------------


def test_ledger_mask_agrees_with_C_on_paper_state_throughout(synthetic_cfg):
    """CLAUDE.md implementation note: the env's internal liveness ledger is
    equivalent to evaluating C on paper_state() — asserted at EVERY auction
    decision of an episode with a mixed submit/abstain/cancel policy."""
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=13)
    rng = np.random.default_rng(99)
    checked = 0
    while True:
        admissible = env._ledger.cancel_admissible()
        assert (C_of_paper_state(env.paper_state()) > 0) == admissible
        mask = env.action_mask()
        cancel_rows = np.array([a.cancel == 1 for a in env.auction_grid.actions])
        assert mask[cancel_rows].any() == admissible
        checked += 1

        K = float(rng.choice([0.0, 0.0, 2.0, 5.0]))  # abstain half the time
        cancel = int(
            admissible
            and (K > 0.0 or rng.random() < 0.4)
        )  # single-replace requires cancellation for a new live schedule
        offset = 0 if K == 0.0 else int(rng.integers(-3, 4))
        _, _, term, _, _ = env.step(AuctionAction(K, offset, cancel))
        if term:
            break
    assert checked == env.grid.tau_cl - env.grid.tau_op  # every auction step
