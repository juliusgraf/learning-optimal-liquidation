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

from helpers import NOOP_AUCTION, drive_to_auction, new_env


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
    with pytest.raises(ValueError, match="volume .* > inventory"):
        env.step(ClobAction(synthetic_cfg.grid.I0 + 1.0, 2))
    with pytest.raises(ValueError, match="delta must be >= 0"):
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
    assert mask[~cancel_rows].all()  # c = 0 actions always admissible
    assert env._ledger.cancel_admissible() == expected
    assert (C_of_paper_state(env.paper_state()) > 0) == expected


def test_cancel_forbidden_at_auction_open_and_after_abstain(synthetic_cfg):
    """C(x) = 0 at the auction open (property (i) of CLAUDE.md) and stays 0
    while the agent only abstains (K^a = 0 submissions are no entries)."""
    env = new_env(synthetic_cfg)
    drive_to_auction(env, seed=2)
    assert_cancel_admissible(env, False)  # t = n+1: c_{t_{n+1}} = 0 forced
    with pytest.raises(ValueError, match="cancel-all with no live prior"):
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

    # Cancel-all WITHOUT a new submission: ledger empty again afterwards.
    env.step(AuctionAction(0.0, 0, 1))
    assert_cancel_admissible(env, False)

    # Submit + cancel in one step: the same-step order survives (theta
    # recursion cutoff s < t), so the cancel stays admissible at t+1.
    env.step(AuctionAction(3.0, 2, 0))
    assert_cancel_admissible(env, True)
    env.step(AuctionAction(1.0, 0, 1))
    assert_cancel_admissible(env, True)  # the just-submitted K=1 order lives


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
    with pytest.raises(ValueError, match="K\\^a must be >= 0"):
        env.step(AuctionAction(-1.0, 0, 0))
    with pytest.raises(ValueError, match="c_t must be 0 or 1"):
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
        cancel = int(admissible and rng.random() < 0.4)
        _, _, term, _, _ = env.step(AuctionAction(K, int(rng.integers(-3, 4)), cancel))
        if term:
            break
    assert checked == env.grid.tau_cl - env.grid.tau_op  # every auction step
