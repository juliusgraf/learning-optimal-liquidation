"""Frozen-policy evaluation with common random numbers (Phase 4).

Evaluates the checkpointed DQN, the untrained "initial" DQN, and the AS/TWAP
benchmarks on the SAME seed set (CRN: identical env seed for every policy
within an episode; fixes AUDIT N10), under ONE reward definition for all
policies (AUDIT C.4: the shared RewardParams from the run's resolved config;
any reward override applies to ALL policies identically). Reported returns
are UNDISCOUNTED episode sums (stated in metadata); the chi-discounted V_0
estimate is recorded alongside for regret.py.

The eval seed stream is the dedicated ``env_final_eval`` component of the
run's seed bundle — disjoint from the training and periodic-eval streams by
construction (ruling D10). Writes eval/records.csv (one row per (policy,
episode)) and eval/metadata.yaml.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.config import load_config, to_dict
from lmm.env.mdp import make_env
from lmm.experiments.tracing import EpisodeTraceRecorder
from lmm.experiments.train import draw_seed, make_agent, wrap_env_for_agent
from lmm.rl.loops import SEED_COMPONENTS, EpisodeResult, run_episode
from lmm.utils.seeding import seed_everything

__all__ = ["build_parser", "main", "RECORD_COLUMNS", "POLICIES"]

POLICIES = ("dqn", "initial", "as", "twap")

RECORD_COLUMNS = [
    "policy",
    "episode",
    "env_seed",
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
]


def _record_row(policy: str, episode: int, res: EpisodeResult) -> list[str]:
    def fmt(v):
        return format(v, ".17g") if isinstance(v, float) else str(v)

    return [
        policy,
        str(episode),
        str(res.env_seed),
        fmt(res.return_undisc),
        fmt(res.return_disc),
        fmt(res.clob_reward_sum),
        fmt(res.auction_step_reward_sum),
        fmt(res.terminal_reward),
        fmt(res.s_cl),
        fmt(res.z_tau_cl),
        fmt(res.i_final),
        fmt(res.h_at_tau_op),
        str(res.cancel_count),
        str(res.n_steps),
        str(res.n_clob_steps),
        str(res.n_degenerate_fallbacks),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-evaluate",
        description="Evaluate a frozen policy vs benchmarks with CRN.",
    )
    parser.add_argument("--run-dir", required=True, help="training run directory")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="checkpoint to evaluate (without .pt). Default 'best' = early "
        "stopping: the best-validation snapshot (selected on the env_eval "
        "stream, disjoint from the env_final_eval test seeds). Pass 'final' "
        "for the last-episode checkpoint.",
    )
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="evaluation episodes (default: algo.hyperparams.final_eval_n_seeds)")
    parser.add_argument("--symbol", default=None, help="historical setting: symbol to replay")
    parser.add_argument(
        "--trace-episodes", type=int, default=0,
        help="write per-step anatomy traces for the first N eval episodes "
        "(eval/traces/<policy>_ep<i>.csv) for make_figures; 0 = none",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    master_seed = int((run_dir / "seed.txt").read_text().strip())
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)
    n_episodes = (
        args.n_episodes
        if args.n_episodes is not None
        else int(cfg.algo.hyperparams["final_eval_n_seeds"])
    )

    # Disjoint final-eval seed stream (D10); the SAME list for every policy
    # is the CRN coupling (the env's exogenous draws are policy-independent).
    eval_rng = seeds.generators["env_final_eval"]
    eval_seeds = [draw_seed(eval_rng) for _ in range(n_episodes)]

    # One env per policy, all from the SAME resolved config => identical
    # RewardParams for all policies (AUDIT C.4; asserted in tests). The
    # learned-policy envs (dqn/initial) are wrapped in the continuous adapter
    # for DDPG/TD3/SAC runs; the benchmark envs stay raw (benchmarks submit
    # ClobAction/AuctionAction objects directly).
    learned = {"dqn", "initial"}
    envs = {
        p: (wrap_env_for_agent(make_env(cfg, symbol=args.symbol), cfg)
            if p in learned else make_env(cfg, symbol=args.symbol))
        for p in POLICIES
    }

    dqn = make_agent(cfg, seeds)
    dqn.load(run_dir / "checkpoints" / f"{args.checkpoint}.pt")
    initial = make_agent(cfg, seeds)
    initial.load(run_dir / "checkpoints" / "initial.pt")
    as_agent = ASBenchmarkAgent(cfg)
    calibration = as_agent.calibrate(
        envs["as"],
        rng_k=seeds.generators["as_calibration"],
        rng_sigma=seeds.generators["as_sigma_paths"],
    )
    twap = TWAPBenchmarkAgent(cfg)
    agents = {"dqn": dqn, "initial": initial, "as": as_agent, "twap": twap}
    for name, agent in agents.items():
        agent.bind(envs[name])

    records_path = run_dir / "eval" / "records.csv"
    with records_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RECORD_COLUMNS)
        for episode, env_seed in enumerate(eval_seeds):
            for name, agent in agents.items():
                agent.start_episode(episode)
                res = run_episode(envs[name], agent, env_seed, chi=cfg.rl.chi, train=False)
                writer.writerow(_record_row(name, episode, res))

    # Per-step anatomy traces (Phase 7): replay the FIRST n_trace eval seeds
    # for every policy with a recorder attached. The env resets by seed so this
    # reproduces the records.csv trajectories exactly (frozen/greedy policies
    # consume no exploration RNG; benchmarks reset per episode in start_episode).
    n_trace = min(args.trace_episodes, n_episodes)
    if n_trace > 0:
        traces_dir = run_dir / "eval" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        for episode in range(n_trace):
            env_seed = eval_seeds[episode]
            for name, agent in agents.items():
                agent.start_episode(episode)
                rec = EpisodeTraceRecorder()
                run_episode(
                    envs[name], agent, env_seed, chi=cfg.rl.chi, train=False, on_step=rec
                )
                rec.write(traces_dir / f"{name}_ep{episode}.csv")

    metadata = {
        "master_seed": master_seed,
        "checkpoint": str(run_dir / "checkpoints" / f"{args.checkpoint}.pt"),
        "early_stopping": args.checkpoint == "best",
        "n_episodes": n_episodes,
        "policies": list(POLICIES),
        "crn": "identical env seed per episode across all policies (env_final_eval stream)",
        "return_convention": "return_undisc = undiscounted episode sum (reported); "
        "return_disc = sum chi^t r_t with t the decision time, terminal at chi^tau_cl",
        "reward_params_shared_by_all_policies": to_dict(cfg.reward),  # AUDIT C.4
        "as_calibration": {k: float(v) for k, v in calibration.items()},
        "trace_episodes": n_trace,
        "setting": cfg.experiment.name,
        "symbol": args.symbol,  # historical ticker replayed (None for synthetic)
    }
    (run_dir / "eval" / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(f"wrote {records_path} ({n_episodes} episodes x {len(POLICIES)} policies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
