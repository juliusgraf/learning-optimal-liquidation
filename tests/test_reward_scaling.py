"""Archival large-notional reward stress tests (including native learners).

The DQN scale below is an explicit legacy fixture. Active weighted-reward,
potential and SB3 contracts are tested in test_learning_repair.py. The former
large-notional reward distribution is not the current replay distribution.

The paper's three-regime rewards are LARGE: auction/terminal terms
~K^a * H * (H - S^a) with H ~ 100 give per-transition rewards O(1e3-1e4) and
episode returns up to ~1e6. The learner scales them by ``reward_scale`` INSIDE
replay only (docs/rl_design.md; reported metrics stay in PAPER units). At the
old default ``reward_scale = 1.0`` the value targets blew up: the continuous
MSE critics reached losses ~1e18 (DDPG) and ~1e26 / grad ``inf`` (SAC), and the
DQN's clipped gradients were swamped by O(1e5) TD errors (none of the four
agents learned; see the seeded reproduce_all run analysed when this fix landed).

These tests pin the contract that keeps the Bellman targets the agents regress
on bounded:

1. the configured ``reward_scale`` maps a representative large paper reward into
   a trainable band (the regression guard against a return to 1.0);
2. ``observe`` actually applies the scale on the replay path;
3. ``compute_targets`` stays finite and bounded by the scaled-reward magnitude;
4. a short SEEDED training segment fed paper-scale rewards keeps the per-update
   losses / TD errors finite and non-divergent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import load_algo_cfg, load_dqn_cfg
from lmm.agents.base import Transition
from lmm.agents.ddpg import DDPGAgent
from lmm.agents.dqn import DQNAgent
from lmm.agents.sac import SACAgent
from lmm.agents.td3 import TD3Agent
from lmm.env.features import COMMON_FEATURES, FeatureNormalizer
from lmm.rl.continuous_replay import ContinuousReplayBatch
from lmm.rl.loops import SEED_COMPONENTS
from lmm.rl.replay import ReplayBatch
from lmm.utils.seeding import seed_everything

ALGOS = ["dqn", "ddpg", "td3", "sac"]
CONTINUOUS_ALGOS = ["ddpg", "td3", "sac"]
_AGENT_CLS = {"dqn": DQNAgent, "ddpg": DDPGAgent, "td3": TD3Agent, "sac": SACAgent}

# A representative LARGE per-transition paper reward (auction/terminal scale,
# grounded in metrics.csv: terminal_reward and auction step rewards reach
# O(1e3-1e4)). The scaled value the critic regresses on should land in a
# textbook-trainable band; reward_scale = 1.0 maps it to 1e4 and is rejected.
TYPICAL_PAPER_REWARD = 1.0e4
TRAINABLE_BAND = (0.1, 100.0)  # acceptable |scaled reward| for the typical case


def build_agent(algo: str, *overrides: str, master_seed: int = 1234):
    cfg = (load_dqn_cfg('algo.hyperparams.reward_scale=0.01', *overrides)
           if algo == "dqn" else load_algo_cfg(algo, *overrides))
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)
    return cfg, _AGENT_CLS[algo](cfg, seeds)


def _clob_action(algo: str, agent, rng: np.random.Generator | None = None):
    """An executable CLOB action of the right TYPE for the agent."""
    if algo == "dqn":
        if rng is None:
            return 0
        return int(rng.integers(len(agent.clob_grid)))
    dim = agent._act_dim["clob"]
    if rng is None:
        return np.zeros(dim, np.float32)
    return np.clip(rng.normal(size=dim), -1.0, 1.0).astype(np.float32)


def _clob_next_mask(algo: str, agent):
    """next_mask for an ordinary CLOB->CLOB transition (None for continuous,
    which derives admissibility in the adapter; DQN stores the next-grid mask)."""
    return np.ones(len(agent.clob_grid), bool) if algo == "dqn" else None


def _clob_targets(algo: str, agent, cfg, scaled_rewards: np.ndarray) -> np.ndarray:
    """Build a CLOB minibatch with the given (already replay-scaled) rewards
    (last row terminal) and return compute_targets as a numpy array."""
    clob_dim, auc_dim = len(cfg.features.clob), len(cfg.features.auction)
    n = len(scaled_rewards)
    done = np.zeros(n, bool)
    done[-1] = True  # one terminal row: zero bootstrap, y == scaled reward
    junction = np.zeros(n, bool)
    next_obs = np.zeros((n, max(clob_dim, auc_dim)), np.float32)
    reward = np.asarray(scaled_rewards, np.float32)
    if algo == "dqn":
        mask_dim = max(len(agent.clob_grid), len(agent.auction_grid))
        batch = ReplayBatch(
            obs=np.zeros((n, clob_dim), np.float32),
            action=np.zeros(n, np.int64),
            reward=reward,
            next_obs=next_obs,
            done=done,
            junction=junction,
            next_mask=np.ones((n, mask_dim), bool),
        )
    else:
        batch = ContinuousReplayBatch(
            obs=np.zeros((n, clob_dim), np.float32),
            action=np.zeros((n, agent._act_dim["clob"]), np.float32),
            reward=reward,
            next_obs=next_obs,
            done=done,
            junction=junction,
            next_cancel_admissible=np.zeros(n, bool),
        )
    return agent.compute_targets("clob", batch).detach().cpu().numpy()


@pytest.mark.parametrize("algo", ALGOS)
def test_reward_scale_in_trainable_band(algo):
    """Every method uses the manuscript's common reward scale exactly."""
    _, agent = build_agent(algo)
    s = agent.hp.reward_scale
    assert s == pytest.approx(0.01 if algo == "dqn" else 1.0e-3, rel=0, abs=1e-15)
    scaled = abs(s * TYPICAL_PAPER_REWARD)
    lo, hi = TRAINABLE_BAND
    assert lo <= scaled <= hi, (
        f"{algo}: reward_scale={s} maps a typical paper reward "
        f"{TYPICAL_PAPER_REWARD:g} to {scaled:g}, outside the trainable band {TRAINABLE_BAND}"
    )


@pytest.mark.parametrize("algo", ALGOS)
def test_observe_applies_reward_scale_on_the_replay_path(algo):
    """observe() stores reward_scale * r (paper units in, scaled units stored)."""
    cfg, agent = build_agent(algo)
    s = agent.hp.reward_scale
    clob_dim = len(cfg.features.clob)
    raw = 1.0e3
    agent.observe(
        Transition(
            obs=np.zeros(clob_dim, np.float32),
            action=_clob_action(algo, agent),
            reward=raw,
            next_obs=np.zeros(clob_dim, np.float32),
            done=False,
            phase="clob",
            next_phase="clob",
            next_mask=_clob_next_mask(algo, agent),
        )
    )
    stored = float(agent.replay["clob"].sample(1).reward[0])
    assert stored == pytest.approx(raw * s, rel=1e-6)
    assert abs(stored) <= TRAINABLE_BAND[1]


@pytest.mark.parametrize("algo", ALGOS)
def test_terminal_reward_unfolded_and_added_exactly_once(algo):
    """observe() does not fold the terminal clearing reward into the
    stored step reward. The env returns the combined reward r_step + r_tau_cl
    (with r_tau_cl in info['terminal_reward']); observe stores only the step
    reward and carries g = r_tau_cl in ``terminal_value``. The undiscounted
    terminal target is c_r*r_step + c_r*g exactly once."""
    cfg, agent = build_agent(algo)
    s = agent.hp.reward_scale
    clob_dim = len(cfg.features.clob)
    r_step, r_term = 3.0, 120.0
    agent.observe(
        Transition(
            obs=np.zeros(clob_dim, np.float32),
            action=_clob_action(algo, agent),
            reward=r_step + r_term,  # env folds the terminal reward into the step
            next_obs=np.zeros(clob_dim, np.float32),
            done=True,
            phase="clob",
            next_phase="clob",
            next_mask=None,
            info={"terminal_reward": r_term},
        )
    )
    batch = agent.replay["clob"].sample(1)
    assert float(batch.reward[0]) == pytest.approx(r_step * s, rel=1e-6)
    assert float(batch.terminal_value[0]) == pytest.approx(r_term * s, rel=1e-6)
    y = agent.compute_targets("clob", batch).detach().cpu().numpy()
    assert y[0] == pytest.approx((r_step + r_term) * s, rel=1e-6)


@pytest.mark.parametrize("algo", ALGOS)
def test_bellman_targets_finite_and_bounded_by_scaled_reward(algo):
    """compute_targets is finite and tracks the scaled-reward magnitude (no
    blow-up beyond reward + a small init-bootstrap slack) -- even when a single
    raw reward is at the episode-return extreme (~1e6)."""
    cfg, agent = build_agent(algo)
    s = agent.hp.reward_scale
    raw = np.array([1.0e6, -1.0e6, 5.0e5, -3.0e5, 1.0e4, -1.0e4, 0.0], dtype=np.float64)
    scaled = (raw * s).astype(np.float32)
    y = _clob_targets(algo, agent, cfg, scaled)
    assert np.all(np.isfinite(y)), f"{algo}: non-finite Bellman targets"
    # y = r_scaled + Q_target(x') (or just r_scaled on the terminal row);
    # at init Q is O(1), so |y| <= max|scaled| + a small slack.
    bound = float(np.abs(scaled).max()) * 1.5 + 100.0
    assert np.max(np.abs(y)) <= bound, f"{algo}: |target|={np.max(np.abs(y)):g} exceeds {bound:g}"


@pytest.mark.parametrize("algo", ALGOS)
def test_gradient_updates_stay_finite_under_paper_scale_rewards(algo):
    """A short SEEDED training segment fed paper-scale rewards keeps the
    per-update losses / TD errors / gradient norms finite and non-divergent."""
    cfg, agent = build_agent(
        algo,
        "rl.learning_starts_after_warmup=false",
        "algo.hyperparams.min_buffer=32",
        "algo.hyperparams.min_buffer_clob=32",
        "algo.hyperparams.min_buffer_auction=32",
        "algo.hyperparams.batch_size=16",
    )
    clob_dim = len(cfg.features.clob)
    rng = np.random.default_rng(0)
    diags = []
    for _ in range(200):
        raw = float(rng.uniform(-TYPICAL_PAPER_REWARD, TYPICAL_PAPER_REWARD))
        agent.observe(
            Transition(
                obs=rng.normal(size=clob_dim).astype(np.float32),
                action=_clob_action(algo, agent, rng),
                reward=raw,
                next_obs=rng.normal(size=clob_dim).astype(np.float32),
                done=False,
                phase="clob",
                next_phase="clob",
                next_mask=_clob_next_mask(algo, agent),
            )
        )
        d = agent.update()
        if d:
            diags.append(d)
    assert diags, f"{algo}: expected gradient updates after min_buffer was reached"
    for d in diags:
        for key in ("loss_clob", "td_abs_mean_clob", "td_abs_max_clob", "grad_norm_clob"):
            if key in d:
                assert math.isfinite(d[key]), f"{algo}: {key}={d[key]} not finite"
    # With scaled rewards the critic loss stays in a trainable range; at
    # reward_scale = 1.0 this segment drove it orders of magnitude past 1e4.
    assert diags[-1]["loss_clob"] < 1.0e4, (
        f"{algo}: final loss_clob={diags[-1]['loss_clob']:g} indicates divergence"
    )


# -- no method-specific reward clipping ---------------------------------------


@pytest.mark.parametrize("algo", ALGOS)
def test_large_rewards_are_scaled_but_never_clipped(algo):
    cfg, agent = build_agent(algo)
    s = agent.hp.reward_scale
    clob_dim = len(cfg.features.clob)

    def observe_raw(raw: float) -> None:
        agent.observe(
            Transition(
                obs=np.zeros(clob_dim, np.float32),
                action=_clob_action(algo, agent),
                reward=raw,
                next_obs=np.zeros(clob_dim, np.float32),
                done=False,
                phase="clob",
                next_phase="clob",
                next_mask=_clob_next_mask(algo, agent),
            )
        )

    observe_raw(1.0e8)
    observe_raw(-1.0e8)
    small = 1.0e3
    observe_raw(small)
    stored = agent.replay["clob"]._reward  # FIFO positions 0,1,2
    assert stored[0] == pytest.approx(1.0e8 * s, rel=1e-6)
    assert stored[1] == pytest.approx(-1.0e8 * s, rel=1e-6)
    assert stored[2] == pytest.approx(small * s, rel=1e-6)


# -- common frozen feature normalization -------------------------------------


@pytest.mark.parametrize("algo", ALGOS)
def test_all_methods_use_the_same_frozen_training_normalizer(algo):
    cfg = load_dqn_cfg() if algo == "dqn" else load_algo_cfg(algo)
    assert tuple(cfg.features.clob) == COMMON_FEATURES
    assert tuple(cfg.features.auction) == COMMON_FEATURES
    rows = np.vstack([np.arange(18, dtype=float), np.arange(18, dtype=float) + 2.0])
    normalizer = FeatureNormalizer(cfg.grid.tau_cl, relative_prices=cfg.rl.relative_price_features).fit(rows)
    transformed = normalizer.transform(rows[0])
    assert transformed.shape == (18,)
    assert transformed[0] == pytest.approx(rows[0, 0] / cfg.grid.tau_cl)
    assert transformed[4] == pytest.approx(rows[0, 4] / cfg.grid.tau_cl)
    assert transformed[12] == rows[0, 12]
    with pytest.raises(RuntimeError, match="frozen"):
        normalizer.update(rows)


def test_h_ablation_zeros_only_normalized_h_coordinate():
    rows = np.vstack([np.zeros(18), np.ones(18)])
    normal = FeatureNormalizer(150.0).fit(rows)
    ablated = FeatureNormalizer(150.0, zero_h_cl=True).fit(rows)
    x = np.arange(18, dtype=float)
    expected = normal.transform(x)
    actual = ablated.transform(x)
    assert actual.shape == expected.shape == (18,)
    assert actual[2] == 0.0
    np.testing.assert_array_equal(actual[np.arange(18) != 2], expected[np.arange(18) != 2])
