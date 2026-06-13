"""Train any agent on any setting from config (Phase 4).

Writes results/<experiment_name>/<run_name>/ with config_resolved.yaml,
seed.txt, git_sha.txt, metrics.csv (per-episode), checkpoints/, logs/run.log
(engineering conventions, CLAUDE.md). Figures/tables are produced separately
by make_figures.py / make_tables.py from these saved outputs.

Checkpoints: ``initial.pt`` BEFORE any training (the "initial-DQN" baseline
evaluated by evaluate.py), periodic resumable ``ckpt_ep{N}.pt`` (+ sidecar
``ckpt_ep{N}_trainstate.pt`` with the loop's own RNG/counters), ``best.pt``
on eval improvement, ``final.pt`` at the end. metrics.csv schema is
documented in docs/metrics_schema.md; floats are written with repr-exact
"%.17g" so byte-identical files certify determinism (ruling D10), except the
final wall_clock_s column which is excluded from comparisons.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from lmm.agents.base import Agent
from lmm.agents.ddpg import DDPGAgent
from lmm.agents.dqn import DQNAgent
from lmm.agents.sac import SACAgent
from lmm.agents.td3 import TD3Agent
from lmm.config import ExperimentConfig, add_config_cli, config_from_args
from lmm.env.action_spaces import ContinuousActionAdapter
from lmm.env.mdp import make_env
from lmm.rl.loops import SEED_COMPONENTS, EpisodeResult, run_episode
from lmm.utils.logging import create_run_dir, get_run_logger, write_run_metadata
from lmm.utils.seeding import SeedBundle, seed_everything

__all__ = [
    "build_parser",
    "main",
    "METRICS_COLUMNS",
    "make_agent",
    "draw_seed",
    "is_continuous",
    "wrap_env_for_agent",
    "CONTINUOUS_ALGOS",
]

METRICS_COLUMNS = [
    "episode",
    "env_seed",
    "epsilon",
    "return_undisc",
    "return_disc",
    "clob_reward_sum",
    "auction_step_reward_sum",
    "terminal_reward",
    "S_cl",
    "Z_tau_cl",
    "I_final",
    "H_at_tau_op",
    "cancel_count",
    "n_steps",
    "n_clob_steps",
    "n_degenerate_fallbacks",
    "loss_clob",
    "grad_norm_clob",
    "td_abs_mean_clob",
    "td_abs_max_clob",
    "n_grad_steps_clob",
    "buffer_clob",
    "loss_auction",
    "grad_norm_auction",
    "td_abs_mean_auction",
    "td_abs_max_auction",
    "n_grad_steps_auction",
    "buffer_auction",
    "eval_return_mean",
    "wall_clock_s",  # LAST column; excluded from determinism comparisons
]


# Continuous-action relaxation agents (Phase 5; ruling D9). DQN stays the
# discrete paper setting; these run on the ContinuousActionAdapter.
CONTINUOUS_ALGOS = {"ddpg": DDPGAgent, "td3": TD3Agent, "sac": SACAgent}
_AGENTS = {"dqn": DQNAgent, **CONTINUOUS_ALGOS}


def make_agent(cfg: ExperimentConfig, seeds: SeedBundle) -> Agent:
    """Algorithm dispatch: dqn (discrete, Phase 4) or ddpg/td3/sac (continuous
    relaxation, Phase 5)."""
    if cfg.algo is None:
        raise ValueError("config has no algo section; overlay configs/algo/*.yaml")
    try:
        agent_cls = _AGENTS[cfg.algo.name]
    except KeyError:
        raise ValueError(
            f"unknown algo {cfg.algo.name!r}; expected one of {sorted(_AGENTS)}"
        ) from None
    return agent_cls(cfg, seeds)


def is_continuous(cfg: ExperimentConfig) -> bool:
    """True iff the configured algorithm is a continuous-action variant."""
    return cfg.algo is not None and cfg.algo.name in CONTINUOUS_ALGOS


def wrap_env_for_agent(env, cfg: ExperimentConfig):
    """Wrap the env in the :class:`ContinuousActionAdapter` for DDPG/TD3/SAC;
    the discrete DQN uses the raw env unchanged."""
    if is_continuous(cfg):
        return ContinuousActionAdapter(
            env, continuous_cancel=cfg.algo.hyperparams.get("continuous_cancel", "threshold")
        )
    return env


def draw_seed(rng: np.random.Generator) -> int:
    """One env seed from a component stream (int32 range for gymnasium)."""
    return int(rng.integers(0, 2**31 - 1))


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if np.isnan(v) else format(v, ".17g")
    return str(v)


def _metrics_row(
    episode: int,
    res: EpisodeResult,
    epsilon: float,
    buffer_sizes: dict[str, int],
    eval_return_mean: Optional[float],
    wall_clock_s: float,
) -> list[str]:
    d = res.diagnostics
    values = {
        "episode": episode,
        "env_seed": res.env_seed,
        "epsilon": epsilon,
        "return_undisc": res.return_undisc,
        "return_disc": res.return_disc,
        "clob_reward_sum": res.clob_reward_sum,
        "auction_step_reward_sum": res.auction_step_reward_sum,
        "terminal_reward": res.terminal_reward,
        "S_cl": res.s_cl,
        "Z_tau_cl": res.z_tau_cl,
        "I_final": res.i_final,
        "H_at_tau_op": res.h_at_tau_op,
        "cancel_count": res.cancel_count,
        "n_steps": res.n_steps,
        "n_clob_steps": res.n_clob_steps,
        "n_degenerate_fallbacks": res.n_degenerate_fallbacks,
        "loss_clob": d.get("loss_clob"),
        "grad_norm_clob": d.get("grad_norm_clob"),
        "td_abs_mean_clob": d.get("td_abs_mean_clob"),
        "td_abs_max_clob": d.get("td_abs_max_clob"),
        "n_grad_steps_clob": int(d.get("n_grad_steps_clob", 0)),
        "buffer_clob": buffer_sizes["clob"],
        "loss_auction": d.get("loss_auction"),
        "grad_norm_auction": d.get("grad_norm_auction"),
        "td_abs_mean_auction": d.get("td_abs_mean_auction"),
        "td_abs_max_auction": d.get("td_abs_max_auction"),
        "n_grad_steps_auction": int(d.get("n_grad_steps_auction", 0)),
        "buffer_auction": buffer_sizes["auction"],
        "eval_return_mean": eval_return_mean,
        "wall_clock_s": wall_clock_s,
    }
    return [_fmt(values[c]) for c in METRICS_COLUMNS]


def _run_eval(env, agent, eval_seeds: list[int], chi: float) -> float:
    """Greedy evaluation on the fixed seed list; mean UNDISCOUNTED return."""
    returns = [run_episode(env, agent, s, chi=chi, train=False).return_undisc for s in eval_seeds]
    return float(np.mean(returns))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-train",
        description="Train an agent (configs select setting and algorithm).",
    )
    add_config_cli(parser)
    parser.add_argument("--run-name", required=True, help="results subdirectory name")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.master_seed")
    parser.add_argument("--symbol", default=None, help="historical setting: symbol to replay")
    parser.add_argument(
        "--resume", default=None, metavar="CKPT", help="resume from checkpoints/ckpt_ep{N}.pt"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    master_seed = args.seed if args.seed is not None else cfg.experiment.master_seed

    paths = create_run_dir(cfg.experiment.results_root, cfg.experiment.name, args.run_name)
    write_run_metadata(paths, cfg, master_seed)
    logger = get_run_logger(paths)

    # D10: one bundle from the master seed; torch determinism flags set.
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)
    env = wrap_env_for_agent(make_env(cfg, symbol=args.symbol), cfg)
    agent = make_agent(cfg, seeds)
    hp = agent.hp
    chi = cfg.rl.chi

    env_seed_rng = seeds.generators["env_train"]
    eval_seed_rng = seeds.generators["env_eval"]
    eval_seeds = [draw_seed(eval_seed_rng) for _ in range(hp.eval_n_seeds)]

    start_episode = 0
    best_eval = -np.inf
    if args.resume is not None:
        agent.load(args.resume)
        train_state = torch.load(
            Path(args.resume).with_name(Path(args.resume).stem + "_trainstate.pt"),
            weights_only=False,
        )
        env_seed_rng.bit_generator.state = train_state["env_seed_rng_state"]
        start_episode = int(train_state["episode"]) + 1
        best_eval = float(train_state["best_eval"])
        logger.info("resumed from %s at episode %d", args.resume, start_episode)
    else:
        # The "initial-DQN" baseline: the untrained networks, saved BEFORE
        # any training (evaluated by evaluate.py / regret.py --policy initial).
        agent.save(paths.checkpoints / "initial.pt")

    new_csv = start_episode == 0 or not paths.metrics_csv.exists()
    csv_file = paths.metrics_csv.open("w" if new_csv else "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(METRICS_COLUMNS)
        csv_file.flush()

    n_episodes = cfg.experiment.episodes
    logger.info(
        "training %s for %d episodes (master seed %d, run dir %s)",
        cfg.algo.name,
        n_episodes,
        master_seed,
        paths.run_dir,
    )

    for episode in range(start_episode, n_episodes):
        t0 = time.perf_counter()
        agent.start_episode(episode)
        env_seed = draw_seed(env_seed_rng)
        res = run_episode(env, agent, env_seed, chi=chi, train=True)

        eval_return_mean: Optional[float] = None
        if (episode + 1) % hp.eval_interval_episodes == 0:
            eval_return_mean = _run_eval(env, agent, eval_seeds, chi)
            if eval_return_mean > best_eval:
                best_eval = eval_return_mean
                agent.save(paths.checkpoints / "best.pt")
            logger.info(
                "episode %d: eval return %.4f (best %.4f)", episode, eval_return_mean, best_eval
            )

        if (episode + 1) % hp.checkpoint_interval_episodes == 0:
            ckpt = paths.checkpoints / f"ckpt_ep{episode + 1}.pt"
            agent.save(ckpt, include_replay=True)
            torch.save(
                {
                    "episode": episode,
                    "best_eval": best_eval,
                    "env_seed_rng_state": env_seed_rng.bit_generator.state,
                },
                ckpt.with_name(ckpt.stem + "_trainstate.pt"),
            )

        wall = time.perf_counter() - t0
        writer.writerow(
            _metrics_row(
                episode,
                res,
                agent.epsilon,
                {p: len(agent.replay[p]) for p in ("clob", "auction")},
                eval_return_mean,
                wall,
            )
        )
        csv_file.flush()

    agent.save(paths.checkpoints / "final.pt")
    csv_file.close()
    logger.info("done: %d episodes; final checkpoint at %s", n_episodes, paths.checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
