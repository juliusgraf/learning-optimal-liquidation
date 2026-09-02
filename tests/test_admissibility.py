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
from lmm.market.clob import OrderBook

from helpers import NOOP_AUCTION, drive_to_auction, load_synthetic_cfg, new_env


def C_of_paper_state(ps) -> float:
    """The CLAUDE.md formula, evaluated literally on X^9 and X^17."""
    x9, x17 = ps["X9"], ps["X17"]
    vals = (1.0 - x9) * (x17 > 0.0)
    return float(np.max(vals)) if len(vals) else 0.0


def test_clob_depth_is_the_populated_level_count(synthetic_cfg):
    book = OrderBook(synthetic_cfg.clob_flow, synthetic_cfg.grid)
    full = np.ones(synthetic_cfg.clob_flow.Lc, dtype=float)
    book.ask_volumes = full.copy()
    book.bid_volumes = full.copy()
    assert book.depth(+1) == book.depth(-1) == synthetic_cfg.clob_flow.Lc

    book.ask_volumes[2:] = 0.0
    book.bid_volumes[:] = 0.0
    assert book.depth(+1) == 2
    assert book.depth(-1) == 0


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
    assert len(env.auction_grid) == 1346
    assert env.action_mask().sum() == 673
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
    assert env.action_mask().sum() == len(env.auction_grid)

    # Cancel-all WITHOUT a new submission: ledger empty again afterwards.
    env.step(AuctionAction(0.0, 0, 1))
    assert_cancel_admissible(env, False)

    # Submit + cancel in one step: the same-step order survives (theta
    # recursion cutoff s < t), so the cancel stays admissible at t+1.
    env.step(AuctionAction(3.0, 2, 0))
    assert_cancel_admissible(env, True)
    env.step(AuctionAction(1.0, 0, 1))
    assert_cancel_admissible(env, True)  # the just-submitted K=1 order lives


def test_multiple_live_schedules_are_allowed_without_cancellation():
    cfg = load_synthetic_cfg()
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    env.step(AuctionAction(cfg.actions.beta, 1, 0))

    mask = env.action_mask()
    stacking = np.array(
        [a.K_a > 0.0 and a.cancel == 0 for a in env.auction_grid.actions]
    )
    assert mask[stacking].all()
    assert mask.all()
    env.step(AuctionAction(2.0 * cfg.actions.beta, 2, 0))
    assert env.own_slope == pytest.approx(3.0 * cfg.actions.beta)
    assert env.cancel_admissible


def test_local_action_resolves_to_private_absolute_b():
    cfg = load_synthetic_cfg(
        "actions.beta=6.666666666666667",
        "actions.K_max=5",
        "actions.B_inf=150",
        "actions.B_max=10",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    assert len(env.auction_grid) == 212

    # Both indexed and direct public actions carry local ell=-4. The simulator
    # privately resolves it to absolute b=37-4=33.
    env._h_cache = env.s_mid + 37 * cfg.grid.alpha
    idx = next(
        i
        for i, a in enumerate(env.auction_grid.actions)
        if a.K_a == pytest.approx(cfg.actions.beta)
        and a.ell == -4
        and a.cancel == 0
    )
    assert env.action_mask()[idx]
    assert env._decode_auction(idx) == AuctionAction(cfg.actions.beta, -4, 0)
    direct = AuctionAction(cfg.actions.beta, -4, 0)
    assert env._decode_auction(direct) is direct
    assert env._resolve_auction_action(direct).b == 33


def test_index_and_decoded_template_execute_identically():
    cfg = load_synthetic_cfg(
        "actions.beta=1.0",
        "actions.K_max=5",
        "actions.B_inf=150",
        "actions.B_max=10",
    )
    indexed, templated = new_env(cfg), new_env(cfg)
    drive_to_auction(indexed, seed=12)
    drive_to_auction(templated, seed=12)
    indexed._h_cache = indexed.s_mid + 37 * cfg.grid.alpha
    templated._h_cache = templated.s_mid + 37 * cfg.grid.alpha
    index = next(
        i
        for i, template in enumerate(indexed.auction_grid.actions)
        if template.K_a == cfg.actions.beta
        and template.ell == -4
        and template.cancel == 0
    )
    template = templated.auction_grid.decode(index)

    obs_i, reward_i, done_i, _, info_i = indexed.step(index)
    obs_t, reward_t, done_t, _, info_t = templated.step(template)
    np.testing.assert_array_equal(obs_i, obs_t)
    assert reward_i == reward_t
    assert done_i == done_t
    assert info_i["action"] == info_t["action"] == AuctionAction(
        cfg.actions.beta, -4, 0
    )
    assert info_i["executed_b"] == info_t["executed_b"] == 33
    assert info_i["H_next"] == info_t["H_next"]


def test_indicative_templates_are_masked_at_the_ambient_absolute_boundary():
    cfg = load_synthetic_cfg(
        "auction_flow.B_inf=30",
        "actions.B_inf=30",
        "actions.B_max=10",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    env._h_cache = env.s_mid + 28 * cfg.grid.alpha
    mask = env.action_mask()
    outside = np.array(
        [a.K_a > 0.0 and a.ell > 2 for a in env.auction_grid.actions]
    )
    assert not mask[outside].any()
    with pytest.raises(ValueError, match="resolved auction b"):
        env.step(AuctionAction(cfg.actions.beta, 3, 0))


def test_slope_grid_is_the_complete_manuscript_lattice():
    cfg = load_synthetic_cfg("actions.beta=0.5", "actions.K_max=5")
    env = new_env(cfg)
    slopes = sorted({a.K_a for a in env.auction_grid.actions if a.K_a > 0.0})
    assert slopes == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert len(env.auction_grid) == 212


def test_common_B_inf_controls_strategic_and_exogenous_price_bands():
    cfg = load_synthetic_cfg(
        "auction_flow.B_inf=10",
        "actions.B_inf=10",
        "actions.B_max=10",
    )
    env = new_env(cfg)
    drive_to_auction(env, seed=2)
    assert (cfg.auction_flow.M1, cfg.auction_flow.M2) == (-10, 10)
    assert sorted(
        {a.ell for a in env.auction_grid.actions if a.K_a > 0.0}
    ) == list(range(-10, 11))
    action = env._decode_auction(
        next(
            i
            for i, a in enumerate(env.auction_grid.actions)
            if a.K_a > 0.0 and a.ell == -10 and a.cancel == 0
        )
    )
    assert action.ell == -10
    env._h_cache = env.s_mid
    assert env._resolve_auction_action(action).b == -10


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
    with pytest.raises(ValueError, match="beta lattice"):
        env.step(AuctionAction(0.5 * synthetic_cfg.actions.beta, 0, 0))
    with pytest.raises(ValueError, match="cancel must be 0 or 1"):
        env.step(AuctionAction(1.0, 0, 2))
    with pytest.raises(TypeError):
        env.step(ClobAction(1.0, 2))  # wrong phase's action type


def test_negative_discrete_action_indices_are_rejected(synthetic_cfg):
    env = new_env(synthetic_cfg)
    env.reset(seed=3)
    with pytest.raises(ValueError, match="CLOB action index -1"):
        env.step(-1)
    drive_to_auction(env, seed=3)
    with pytest.raises(ValueError, match="auction action index -1"):
        env.step(-1)


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
        )
        offset = 0 if K == 0.0 else int(rng.integers(-3, 4))
        _, _, term, _, _ = env.step(AuctionAction(K, offset, cancel))
        if term:
            break
    assert checked == env.grid.tau_cl - env.grid.tau_op  # every auction step
