"""Shared episode runner (Phase 4): ONE loop drives every policy.

Running the DQN, the benchmarks and the initial-DQN baseline through the
same function is the structural common-random-numbers guarantee (ruling D10,
AUDIT N10): the env's exogenous draws are policy-independent, so two policies
given the same env seed see the same market.

Discounting convention (paper objective J(pi) = E[sum_{t in T} chi^t r_t]):
the exponent is the DECISION TIME t on the paper's grid — CLOB decision
times hat_t_i are real-valued and their count varies per episode (Assumption
assump:presence), auction times are the integers tau_op..m, and the terminal
reward (folded into the final transition, ruling D9) is discounted at
chi^tau_cl. Reported evaluation "returns" are the UNDISCOUNTED episode sums
unless stated otherwise (CLAUDE.md); both are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lmm.agents.base import Agent, Transition
from lmm.env.mdp import MarketMakingEnv

__all__ = ["SEED_COMPONENTS", "EpisodeResult", "run_episode"]

# Canonical seed-bundle component list (ruling D10). Streams are a function
# of the SORTED full component set (utils/seeding.py), so every entry point
# (train / evaluate / tests) must use THIS list even when it consumes only a
# subset — otherwise the shared streams shift.
SEED_COMPONENTS = (
    "env_train",
    "env_eval",
    "env_final_eval",
    "exploration",
    "replay_clob",
    "replay_auction",
    "as_calibration",
    "as_sigma_paths",
)


@dataclass
class EpisodeResult:
    """Per-episode record (one metrics.csv row is built from this)."""

    env_seed: int
    return_undisc: float = 0.0
    return_disc: float = 0.0
    clob_reward_sum: float = 0.0
    auction_step_reward_sum: float = 0.0
    terminal_reward: float = 0.0
    s_cl: float = float("nan")
    z_tau_cl: float = float("nan")
    i_final: float = float("nan")
    h_at_tau_op: float = float("nan")
    cancel_count: int = 0
    n_steps: int = 0
    n_clob_steps: int = 0
    n_degenerate_fallbacks: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)


def run_episode(
    env: MarketMakingEnv,
    agent: Agent,
    env_seed: int,
    *,
    chi: float,
    train: bool,
) -> EpisodeResult:
    """Play one full episode; in train mode the agent observes every
    transition and update() runs after each env step (the per-env-step
    schedule and gating live inside the agent). Eval mode (train=False) is
    greedy, stores nothing, and consumes no exploration randomness."""
    agent.set_train(train)
    res = EpisodeResult(env_seed=env_seed)
    diag_lists: dict[str, list[float]] = {}

    obs, _ = env.reset(seed=env_seed)
    done = False
    while not done:
        phase = env.phase
        t_decision = env.t
        mask = env.action_mask()
        a = agent.act(obs, mask, phase, eval_mode=not train)
        next_obs, reward, done, _, info = env.step(a)
        next_mask = None if done else env.action_mask()

        agent.observe(
            Transition(
                obs=obs,
                action=a,
                reward=reward,
                next_obs=next_obs,
                done=done,
                phase=phase,
                next_phase=env.phase,
                next_mask=next_mask,
                info=info,
            )
        )
        if train:
            for k, v in agent.update().items():
                diag_lists.setdefault(k, []).append(v)

        res.n_steps += 1
        res.return_undisc += reward
        res.return_disc += chi**t_decision * reward
        if phase == "clob":
            res.n_clob_steps += 1
            res.clob_reward_sum += reward
        else:
            if res.n_steps == res.n_clob_steps + 1:
                res.h_at_tau_op = info["H_used"]  # first auction decision (t = tau_op)
            res.cancel_count += int(info["action"].cancel)
            res.auction_step_reward_sum += reward
        if done:
            # Terminal split: the folded reward is r_step(t_m) + r_terminal;
            # re-discount the terminal part at chi^tau_cl (module docstring).
            r_term = info["terminal_reward"]
            res.terminal_reward = r_term
            res.auction_step_reward_sum -= r_term
            res.return_disc += (chi ** float(env.grid.tau_cl) - chi**t_decision) * r_term
            res.s_cl = info["S_cl"]
            res.z_tau_cl = info["Z"]
            res.i_final = info["I_final"]
        obs = next_obs

    res.n_degenerate_fallbacks = env.eq2.n_degenerate_fallbacks
    for k, values in diag_lists.items():
        if k.startswith("n_grad_steps"):
            res.diagnostics[k] = float(sum(values))
        elif k.startswith("td_abs_max"):
            res.diagnostics[k] = float(max(values))
        else:
            res.diagnostics[k] = float(sum(values) / len(values))
    return res
