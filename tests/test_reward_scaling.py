"""Reward-scaling guard (maintenance fix; ruling D9, D10).

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
from lmm.env.features import FeatureExtractor
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
    cfg = load_dqn_cfg(*overrides) if algo == "dqn" else load_algo_cfg(algo, *overrides)
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)
    return cfg, _AGENT_CLS[algo](cfg, seeds)


def _clob_action(algo: str, agent, rng: np.random.Generator | None = None):
    """An executable CLOB action of the right TYPE for the agent."""
    if algo == "dqn":
        if rng is None:
            return 0
        return int(rng.integers(len(agent.clob_grid)))
    dim = agent._act_dim["clob"]
    return np.zeros(dim, np.float32) if rng is None else rng.normal(size=dim).astype(np.float32)


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
    """The configured reward_scale maps a typical large paper reward into the
    trainable band -- the regression guard against the divergent 1.0 default."""
    _, agent = build_agent(algo)
    s = agent.hp.reward_scale
    assert 0.0 < s < 1.0, f"{algo}: reward_scale={s} must be in (0, 1)"
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
    # Use a reward whose SCALED value stays below reward_clip, so this test
    # isolates the SCALE (the clip is exercised by
    # test_reward_clip_bounds_the_stored_replay_reward). 1e3 paper -> 1.0 scaled,
    # well under the continuous clip (8.0); DQN is unclipped anyway.
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
    # y = r_scaled + chi * Q_target(x') (or just r_scaled on the terminal row);
    # at init Q is O(1), so |y| <= max|scaled| + a small slack.
    bound = float(np.abs(scaled).max()) * 1.5 + 100.0
    assert np.max(np.abs(y)) <= bound, f"{algo}: |target|={np.max(np.abs(y)):g} exceeds {bound:g}"


@pytest.mark.parametrize("algo", ALGOS)
def test_gradient_updates_stay_finite_under_paper_scale_rewards(algo):
    """A short SEEDED training segment fed paper-scale rewards keeps the
    per-update losses / TD errors / gradient norms finite and non-divergent."""
    cfg, agent = build_agent(
        algo, "algo.hyperparams.min_buffer=32", "algo.hyperparams.batch_size=16"
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


# -- reward clipping (continuous agents; part (a)) ----------------------------


@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS)
def test_reward_clip_bounds_the_stored_replay_reward(algo):
    """The continuous configs clip the SCALED replay reward to +/- reward_clip,
    so the fictive-auction-reward exploit cannot inject an unbounded Bellman
    target; legitimate (sub-clip) rewards pass through with the scale only."""
    cfg, agent = build_agent(algo)
    s, clip = agent.hp.reward_scale, agent.hp.reward_clip
    assert clip is not None and clip > 0, f"{algo}: continuous configs must set reward_clip"
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
                next_mask=None,
            )
        )

    observe_raw(1.0e8)  # scaled = 1e5 >> clip  (exploit regime) -> +clip
    observe_raw(-1.0e8)  # -> -clip
    small = 1.0e3  # scaled = small * s; well below clip -> unclipped
    observe_raw(small)
    stored = agent.replay["clob"]._reward  # FIFO positions 0,1,2
    assert stored[0] == pytest.approx(clip, rel=1e-6)
    assert stored[1] == pytest.approx(-clip, rel=1e-6)
    assert stored[2] == pytest.approx(small * s, rel=1e-6)


def test_dqn_reward_is_unclipped_by_default():
    """The discrete DQN keeps reward_clip off (it uses robust Huber and is not
    part of the continuous exploit fix); only the scale applies."""
    _, agent = build_agent("dqn")
    assert getattr(agent.hp, "reward_clip", None) is None


# Calibration of reward_clip to the MEASURED reward envelope (seed-42
# reproduce_all analysis that motivated tightening clip 25.0 -> 8.0).
#
# The largest LEGITIMATE per-transition reward is the terminal reward, dominated
# by the inventory penalty lambda * I_max^2 = 0.5 * 100^2 = 5000 plus bounded
# execution PnL; the AS/TWAP benchmarks reach ~5500 paper units in the seed-42
# synthetic eval. The clip must sit ABOVE this so it never truncates the real
# objective signal (the terminal PnL / inventory term). The fictive per-step
# auction reward K^a*H_cl*(H_cl-S^a) is unbounded: the actor quotes S^a above the
# frozen mid -> cached H_cl ~115-136 -> single steps reach ~1.6e5 paper, and
# small-slope clearing-blowup steps reach ~1e9. So the clip must also be well
# BELOW the fictive exploit. These two bounds define the valid band.
HONEST_TERMINAL_PAPER = 5.5e3   # ceiling on a legitimate transition (terminal-dominated)
MILD_EXPLOIT_PAPER = 1.6e5      # representative fictive-reward exploit step
MIN_EXPLOIT_SUPPRESSION = 10.0  # min factor by which the clip must suppress it


@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS)
def test_reward_clip_calibrated_to_honest_envelope(algo):
    """reward_clip sits in the calibrated band: ABOVE the honest terminal
    envelope (so the real PnL/inventory objective is never clipped) yet well
    BELOW the unbounded fictive-auction exploit (so the critic target stays at
    the honest scale, killing the 1e9-1e11 divergence). The previous 2.0 was too
    tight (clipped legitimate terminals) and 25.0 too loose (left the target
    ~4.5x a legit terminal) -- this pins the regression on both sides."""
    _, agent = build_agent(algo)
    s, clip = agent.hp.reward_scale, agent.hp.reward_clip
    assert clip is not None and clip > 0, f"{algo}: continuous configs must set reward_clip"

    # (i) FLOOR: clip covers the honest terminal envelope in scaled units, else
    #     the learner clips real terminal-PnL / inventory-penalty rewards.
    honest_scaled = HONEST_TERMINAL_PAPER * s
    assert clip >= honest_scaled, (
        f"{algo}: reward_clip={clip} clips legitimate terminal rewards "
        f"(honest terminal envelope = {honest_scaled:g} scaled)"
    )
    # (ii) CEILING: tight enough to suppress a representative exploit step >= 10x.
    exploit_scaled = MILD_EXPLOIT_PAPER * s
    assert exploit_scaled / clip >= MIN_EXPLOIT_SUPPRESSION, (
        f"{algo}: reward_clip={clip} too loose -- suppresses the exploit only "
        f"{exploit_scaled / clip:.1f}x (need >= {MIN_EXPLOIT_SUPPRESSION:g}x). "
        f"The old reward_clip=25.0 failed this bound."
    )


# -- price-feature normalization (continuous agents; part (a)) -----------------


class _StubEnv:
    """Minimal env exposing the feature accessors (all O(1) except the price)."""

    inventory = 0.0
    depth_ask = depth_bid = top_ask = top_bid = 0.0
    n_mm = n_buy = n_sell = 0.0
    t = 0

    def __init__(self, h_cl: float, s_mid: float) -> None:
        self.h_cl = h_cl
        self.s_mid = s_mid


@pytest.mark.parametrize("algo", CONTINUOUS_ALGOS)
def test_continuous_price_features_are_centered_and_clipped(algo):
    """h_cl_norm / s_mid_norm are centered at S0 and clipped, so a spiking
    per-step H_cl cannot blow up the network input (paper-unit metrics
    unaffected -- this is observation-only)."""
    cfg = load_algo_cfg(algo)
    assert "h_cl_norm" in cfg.features.clob and "s_mid_norm" in cfg.features.clob
    assert "h_cl_norm" in cfg.features.auction and "s_mid_norm" in cfg.features.auction
    fe = FeatureExtractor(cfg.features, cfg.grid, cfg.clob_flow, cfg.auction_flow)
    S0 = cfg.grid.S0
    scale, clip = cfg.features.price_norm_scale, cfg.features.price_norm_clip
    h_idx = cfg.features.clob.index("h_cl_norm")

    # centered: H_cl == S0 maps to 0.
    assert fe.clob_features(_StubEnv(h_cl=S0, s_mid=S0))[h_idx] == pytest.approx(0.0)
    # in-range value: affine (x - S0) / scale.
    val = fe.clob_features(_StubEnv(h_cl=S0 + 2.0 * scale, s_mid=S0))[h_idx]
    assert val == pytest.approx(2.0)
    # spike: clipped to +/- clip (bounded network input).
    hi = fe.clob_features(_StubEnv(h_cl=S0 + 1e6, s_mid=S0))[h_idx]
    lo = fe.clob_features(_StubEnv(h_cl=S0 - 1e6, s_mid=S0))[h_idx]
    assert hi == pytest.approx(clip) and lo == pytest.approx(-clip)


def test_dqn_keeps_raw_price_features():
    cfg = load_dqn_cfg()
    assert "h_cl" in cfg.features.clob and "h_cl_norm" not in cfg.features.clob
    assert "s_mid" in cfg.features.auction and "s_mid_norm" not in cfg.features.auction
