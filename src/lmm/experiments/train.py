"""Train any agent on any setting from config (Phase 4).

Writes the configured results root (currently results/revision_v11) with config_resolved.yaml,
seed.txt, git_sha.txt, metrics.csv (per-episode), checkpoints/, logs/run.log
(engineering conventions, CLAUDE.md). Figures/tables are produced separately
by make_figures.py / make_tables.py from these saved outputs.

Checkpoints: ``initial.pt`` BEFORE any training (the untrained initial-policy
diagnostic evaluated by evaluate.py), periodic resumable ``ckpt_ep{N}.pt`` (+ sidecar
``ckpt_ep{N}_trainstate.pt`` with the loop's own RNG/counters), ``best.pt``
on eligible economic-validation improvement, ``final.pt`` at the end. The
untrained and pre-maturity policies remain diagnostics and cannot become
``best.pt``. metrics.csv schema is
documented in docs/metrics_schema.md; floats are written with repr-exact
"%.17g" so byte-identical files certify determinism (ruling D10), except the
final wall_clock_s column which is excluded from comparisons.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from lmm.agents.base import Agent
from lmm.agents.ddpg import DDPGAgent
from lmm.agents.dqn import DQNAgent
from lmm.agents.sac import SACAgent
from lmm.agents.td3 import TD3Agent
from lmm.config import (
    ExperimentConfig,
    add_config_cli,
    config_from_args,
    economic_evaluation_config,
    load_config,
    to_dict,
)
from lmm.env.action_spaces import ContinuousActionAdapter
from lmm.env.features import FeatureNormalizer
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
    "fit_feature_normalizer",
]

METRICS_COLUMNS = [
    "episode",
    "env_seed",
    "epsilon",
    "return_undisc",
    "return_disc",
    "training_return",
    "clob_economic_cash",
    "auction_economic_cash",
    "cancellation_fees",
    "residual_mark",
    "terminal_penalty",
    "clob_shaping_adjustment",
    "auction_interim_shaping",
    "auction_shaping_clawback",
    "auction_terminal_shaping",
    "reward_baseline_adjustment",
    "clob_reward_sum",
    "auction_step_reward_sum",
    "terminal_reward",
    "initial_mid",
    "initial_inventory",
    "clob_exec_qty",
    "clob_cash",
    "auction_exec_qty",
    "auction_cash",
    "cancel_cost",
    "residual_liquidation_price",
    "residual_liquidation_cash",
    "liquidation_pnl_gross",
    "liquidation_pnl_net",
    "inventory_penalty",
    "economic_objective",
    "pnl",
    "risk_adjusted_pnl",
    "pnl_per_initial_notional",
    "risk_adjusted_pnl_per_initial_notional",
    "pnl_bps",
    "risk_adjusted_pnl_bps",
    "negative_terminal_inventory",
    "negative_terminal_inventory_magnitude",
    "S_cl",
    "Z_tau_cl",
    "I_final",
    "H_at_tau_op",
    "cancel_count",
    "n_steps",
    "n_clob_steps",
    "n_degenerate_fallbacks",
    "agent_price_displacement",
    "leave_agent_out_price",
    "continuous_clearing_price",
    "rounding_residual",
    "Q_supply",
    "Q_demand",
    "rho_supply",
    "rho_demand",
    "carryover_slope",
    "fallback_used",
    "self_trade_count",
    "H_forecast_bias",
    "H_forecast_MAE",
    "H_forecast_RMSE",
    "H_MAE_improvement_vs_mid",
    "H_MAE_improvement_vs_open_mid",
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
    "checkpoint_updates_clob",
    "checkpoint_updates_auction",
    "eval_return_mean",
    "eval_pnl_mean",
    "eval_risk_adjusted_pnl_mean",
    "eval_checkpoint_score",
    "eval_checkpoint_eligible",
    "eval_checkpoint_reportable",
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
        return ContinuousActionAdapter(env)
    return env


def draw_seed(rng: np.random.Generator) -> int:
    """One env seed from a component stream (int32 range for gymnasium)."""
    return int(rng.integers(0, 2**31 - 1))


def fit_feature_normalizer(
    cfg: ExperimentConfig,
    seeds: SeedBundle,
    *,
    symbol: str | None = None,
) -> tuple[FeatureNormalizer, list[int]]:
    """Fit once on dedicated training-only random-policy trajectories.

    This pass is disjoint from periodic validation and final evaluation.  The
    returned statistics are frozen before any policy network sees an
    observation, which guarantees identical representations in acting and
    replay from the first learning transition onward.
    """
    n_episodes = int(cfg.rl.normalizer_fit_episodes)
    if n_episodes <= 0:
        raise ValueError("rl.normalizer_fit_episodes must be positive")
    env = make_env(cfg, symbol=symbol, data_split="train")
    env_rng = seeds.generators["normalizer_env"]
    policy_rng = seeds.generators["normalizer_policy"]
    normalizer = FeatureNormalizer(
        cfg.grid.tau_cl,
        zero_h_cl=not cfg.rl.h_cl_feature_enabled,
    )
    used_seeds: list[int] = []
    for _ in range(n_episodes):
        env_seed = draw_seed(env_rng)
        used_seeds.append(env_seed)
        obs, _ = env.reset(seed=env_seed)
        done = False
        while not done:
            normalizer.update(obs)
            admissible = np.flatnonzero(env.action_mask())
            if not len(admissible):
                raise AssertionError("normalizer policy encountered an empty action set")
            action = int(admissible[int(policy_rng.integers(0, len(admissible)))])
            obs, _, done, _, _ = env.step(action)
    return normalizer.freeze(), used_seeds


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
    checkpoint_update_counts: dict[str, int],
    eval_return_mean: Optional[float],
    eval_pnl_mean: Optional[float],
    eval_risk_adjusted_pnl_mean: Optional[float],
    eval_checkpoint_score: Optional[float],
    eval_checkpoint_eligible: Optional[bool],
    eval_checkpoint_reportable: Optional[bool],
    wall_clock_s: float,
) -> list[str]:
    d = res.diagnostics
    values = {
        "episode": episode,
        "env_seed": res.env_seed,
        "epsilon": epsilon,
        "return_undisc": res.return_undisc,
        "return_disc": res.return_disc,
        "training_return": res.training_return,
        "clob_economic_cash": res.clob_economic_cash,
        "auction_economic_cash": res.auction_economic_cash,
        "cancellation_fees": res.cancellation_fees,
        "residual_mark": res.residual_mark,
        "terminal_penalty": res.terminal_penalty,
        "clob_shaping_adjustment": res.clob_shaping_adjustment,
        "auction_interim_shaping": res.auction_interim_shaping,
        "auction_shaping_clawback": res.auction_shaping_clawback,
        "auction_terminal_shaping": res.auction_terminal_shaping,
        "reward_baseline_adjustment": res.reward_baseline_adjustment,
        "clob_reward_sum": res.clob_reward_sum,
        "auction_step_reward_sum": res.auction_step_reward_sum,
        "terminal_reward": res.terminal_reward,
        "initial_mid": res.initial_mid,
        "initial_inventory": res.initial_inventory,
        "clob_exec_qty": res.clob_exec_qty,
        "clob_cash": res.clob_cash,
        "auction_exec_qty": res.auction_exec_qty,
        "auction_cash": res.auction_cash,
        "cancel_cost": res.cancel_cost,
        "residual_liquidation_price": res.residual_liquidation_price,
        "residual_liquidation_cash": res.residual_liquidation_cash,
        "liquidation_pnl_gross": res.liquidation_pnl_gross,
        "liquidation_pnl_net": res.liquidation_pnl_net,
        "inventory_penalty": res.inventory_penalty,
        "economic_objective": res.economic_objective,
        "pnl": res.pnl,
        "risk_adjusted_pnl": res.risk_adjusted_pnl,
        "pnl_per_initial_notional": res.pnl_per_initial_notional,
        "risk_adjusted_pnl_per_initial_notional": res.risk_adjusted_pnl_per_initial_notional,
        "pnl_bps": res.pnl_bps,
        "risk_adjusted_pnl_bps": res.risk_adjusted_pnl_bps,
        "negative_terminal_inventory": int(res.negative_terminal_inventory),
        "negative_terminal_inventory_magnitude": res.negative_terminal_inventory_magnitude,
        "S_cl": res.s_cl,
        "Z_tau_cl": res.z_tau_cl,
        "I_final": res.i_final,
        "H_at_tau_op": res.h_at_tau_op,
        "cancel_count": res.cancel_count,
        "n_steps": res.n_steps,
        "n_clob_steps": res.n_clob_steps,
        "n_degenerate_fallbacks": res.n_degenerate_fallbacks,
        "agent_price_displacement": res.agent_price_displacement,
        "leave_agent_out_price": res.leave_agent_out_price,
        "continuous_clearing_price": res.continuous_clearing_price,
        "rounding_residual": res.rounding_residual,
        "Q_supply": res.q_supply,
        "Q_demand": res.q_demand,
        "rho_supply": res.rho_supply,
        "rho_demand": res.rho_demand,
        "carryover_slope": res.carryover_slope,
        "fallback_used": int(res.fallback_used),
        "self_trade_count": res.self_trade_count,
        "H_forecast_bias": res.h_forecast_bias,
        "H_forecast_MAE": res.h_forecast_mae,
        "H_forecast_RMSE": res.h_forecast_rmse,
        "H_MAE_improvement_vs_mid": res.h_mae_improvement_vs_mid,
        "H_MAE_improvement_vs_open_mid": res.h_mae_improvement_vs_open_mid,
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
        "checkpoint_updates_clob": checkpoint_update_counts["clob"],
        "checkpoint_updates_auction": checkpoint_update_counts["auction"],
        "eval_return_mean": eval_return_mean,
        "eval_pnl_mean": eval_pnl_mean,
        "eval_risk_adjusted_pnl_mean": eval_risk_adjusted_pnl_mean,
        "eval_checkpoint_score": eval_checkpoint_score,
        "eval_checkpoint_eligible": (
            None if eval_checkpoint_eligible is None else int(eval_checkpoint_eligible)
        ),
        "eval_checkpoint_reportable": (
            None if eval_checkpoint_reportable is None else int(eval_checkpoint_reportable)
        ),
        "wall_clock_s": wall_clock_s,
    }
    return [_fmt(values[c]) for c in METRICS_COLUMNS]


def _run_eval(
    env, agent, eval_seeds: list[int], chi: float, checkpoint_metric: str
) -> dict[str, float]:
    """Greedy fixed-seed validation, retaining reward and economic metrics."""
    episodes = [run_episode(env, agent, s, chi=chi, train=False) for s in eval_seeds]
    if not episodes or not hasattr(episodes[0], checkpoint_metric):
        raise ValueError(f"unknown rl.checkpoint_metric {checkpoint_metric!r}")
    out = {
        "return_mean": float(np.mean([r.return_undisc for r in episodes])),
        "pnl_mean": float(np.mean([r.pnl for r in episodes])),
        "risk_adjusted_pnl_mean": float(
            np.mean([r.risk_adjusted_pnl for r in episodes])
        ),
        "checkpoint_score": float(
            np.mean([float(getattr(r, checkpoint_metric)) for r in episodes])
        ),
    }
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError(f"non-finite validation summary: {out}")
    return out


def _checkpoint_eligibility(
    cfg: ExperimentConfig, agent: Agent
) -> dict[str, object]:
    """Return the auditable joint maturity test for a reportable checkpoint.

    The learned policy is a coupled pair of phase networks, so eligibility is
    joint even though optimizer progress is phase-specific.  In the no-auction
    treatment only the CLOB threshold applies.
    """
    observed = agent.checkpoint_update_counts
    required = {
        "clob": int(cfg.rl.checkpoint_min_clob_updates),
        "auction": (
            int(cfg.rl.checkpoint_min_auction_updates)
            if cfg.experiment.auction_enabled
            else 0
        ),
    }
    deficits = {
        phase: max(0, required[phase] - observed.get(phase, 0))
        for phase in ("clob", "auction")
    }
    return {
        "eligible": not any(deficits.values()),
        "required_updates": required,
        "observed_updates": {
            phase: int(observed.get(phase, 0))
            for phase in ("clob", "auction")
        },
        "update_deficits": deficits,
        "auction_enabled": bool(cfg.experiment.auction_enabled),
    }


def _write_best_selection(
    path: Path,
    *,
    cfg: ExperimentConfig,
    metric: str,
    value: float,
    episode: int,
    eval_seeds: list[int],
    eligibility: dict[str, object],
    safety_reference_score: float,
    reportable: bool,
    candidate: str = "periodic_validation",
) -> None:
    """Write the complete mature-checkpoint selection provenance."""
    path.write_text(
        yaml.safe_dump(
            {
                "metric": metric,
                "mode": "max",
                "value": float(value),
                "episode": int(episode),
                "n_validation_seeds": len(eval_seeds),
                "validation_seeds": eval_seeds,
                "validation_frequency_episodes": cfg.rl.validation_frequency_episodes,
                "patience_evals": cfg.rl.validation_patience_evals,
                "patience_starts_after_first_eligible_validation": True,
                "seed_stream": "env_eval",
                "candidate": candidate,
                "eligibility": eligibility,
                "economic_safety": {
                    "require_improvement_over_initial": bool(
                        cfg.rl.checkpoint_require_initial_improvement
                    ),
                    "initial_validation_score": float(safety_reference_score),
                    "candidate_beats_initial": bool(
                        value > safety_reference_score
                    ),
                    "reportable": bool(reportable),
                },
            },
            sort_keys=False,
        )
    )


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


def _with_master_seed_override(
    cfg: ExperimentConfig, master_seed: int | None
) -> ExperimentConfig:
    """Reflect a single-run CLI seed override in resolved provenance."""
    if master_seed is None:
        return cfg
    return dataclasses.replace(
        cfg,
        experiment=dataclasses.replace(
            cfg.experiment,
            master_seed=master_seed,
            seeds=(master_seed,),
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # ``config_resolved.yaml`` is the canonical description of what was
    # executed.  A CLI seed override therefore has to be reflected in the
    # resolved dataclass before provenance is written (and before resume
    # compatibility is checked), rather than living only in ``seed.txt``.
    cfg = _with_master_seed_override(config_from_args(args), args.seed)
    master_seed = cfg.experiment.master_seed
    # Seed and apply any launcher-provided Torch thread budget before recording
    # runtime provenance or constructing networks.
    seeds = seed_everything(master_seed, SEED_COMPONENTS, seed_torch=True)

    paths = create_run_dir(cfg.experiment.results_root, cfg.experiment.name, args.run_name)
    material_outputs_exist = (
        paths.config_resolved.exists()
        or paths.metrics_csv.exists()
        or any(paths.checkpoints.iterdir())
    )
    if args.resume is None:
        if material_outputs_exist:
            raise FileExistsError(
                f"refusing to overwrite/reuse existing run artifacts in {paths.run_dir}; "
                "choose a new run name or pass --resume with a revised checkpoint"
            )
        write_run_metadata(paths, cfg, master_seed)
    else:
        if not paths.config_resolved.exists() or not paths.seed_txt.exists():
            raise FileNotFoundError(
                f"resume run {paths.run_dir} has no complete saved provenance"
            )
        saved_cfg = load_config(paths.config_resolved)
        saved_seed = int(paths.seed_txt.read_text().strip())
        if to_dict(saved_cfg) != to_dict(cfg) or saved_seed != master_seed:
            raise ValueError(
                "resume config/master seed does not match the existing revised run"
            )
    logger = get_run_logger(paths)

    # D10: ``seeds`` is the one bundle created above; Torch determinism flags
    # and the optional per-worker thread budget are already active.
    env = wrap_env_for_agent(
        make_env(cfg, symbol=args.symbol, data_split="train"), cfg
    )
    economic_cfg = economic_evaluation_config(cfg)
    validation_env = wrap_env_for_agent(
        make_env(economic_cfg, symbol=args.symbol, data_split="validation"), cfg
    )
    agent = make_agent(cfg, seeds)
    hp = agent.hp
    chi = cfg.rl.chi
    checkpoint_metric = cfg.rl.checkpoint_metric
    allowed_checkpoint_metrics = {"risk_adjusted_pnl"}
    if checkpoint_metric not in allowed_checkpoint_metrics:
        raise ValueError(
            f"rl.checkpoint_metric must be one of {sorted(allowed_checkpoint_metrics)}, "
            f"got {checkpoint_metric!r}"
        )

    env_seed_rng = seeds.generators["env_train"]
    eval_seed_rng = seeds.generators["env_eval"]
    eval_seeds = [draw_seed(eval_seed_rng) for _ in range(cfg.rl.validation_size)]

    start_episode = 0
    reference_validation_score: float | None = None
    best_mature_validation_score = -np.inf
    best_mature_episode: int | None = None
    best_validation_score = -np.inf
    best_episode: int | None = None
    validation_evals_without_improvement = 0
    if args.resume is not None:
        agent.load(args.resume)
        if getattr(agent, "_feature_normalizer", None) is None:
            raise ValueError("resume checkpoint has no fitted feature normalizer")
        train_state = torch.load(
            Path(args.resume).with_name(Path(args.resume).stem + "_trainstate.pt"),
            weights_only=False,
        )
        env_seed_rng.bit_generator.state = train_state["env_seed_rng_state"]
        start_episode = int(train_state["episode"]) + 1
        saved_metric = train_state.get("checkpoint_metric")
        if saved_metric != checkpoint_metric:
            raise ValueError(
                "resume checkpoint selection metric mismatch: "
                f"saved={saved_metric!r}, resolved={checkpoint_metric!r}"
            )
        best_validation_score = float(train_state["best_validation_score"])
        best_episode = train_state.get("best_episode")
        reference_validation_score = float(
            train_state["reference_validation_score"]
        )
        best_mature_validation_score = float(
            train_state["best_mature_validation_score"]
        )
        best_mature_episode = train_state.get("best_mature_episode")
        validation_evals_without_improvement = int(
            train_state.get("validation_evals_without_improvement", 0)
        )
        logger.info("resumed from %s at episode %d", args.resume, start_episode)
    else:
        normalizer, normalizer_seeds = fit_feature_normalizer(
            cfg, seeds, symbol=args.symbol
        )
        agent.set_feature_normalizer(normalizer)
        (paths.run_dir / "feature_normalizer.yaml").write_text(
            yaml.safe_dump(
                {
                    "fit_split": "training_calibration",
                    "n_episodes": len(normalizer_seeds),
                    "seed_stream": "normalizer_env",
                    "policy_seed_stream": "normalizer_policy",
                    "episode_seeds": normalizer_seeds,
                    "state": normalizer.state_dict(),
                },
                sort_keys=False,
            )
        )
        # The initial-policy diagnostic: the untrained networks, saved BEFORE
        # any training (evaluated by evaluate.py as the initial reference).
        agent.save(paths.checkpoints / "initial.pt")
        # Retain the untrained policy as a diagnostic reference, but never let
        # it become the reportable best checkpoint.  It has, by definition,
        # not passed either phase's learning-maturity gate.
        initial_validation = _run_eval(
            validation_env, agent, eval_seeds, chi, checkpoint_metric
        )
        reference_validation_score = initial_validation["checkpoint_score"]
        initial_eligibility = _checkpoint_eligibility(cfg, agent)
        (paths.checkpoints / "initial_validation.yaml").write_text(
            yaml.safe_dump(
                {
                    "metric": checkpoint_metric,
                    "value": initial_validation["checkpoint_score"],
                    "episode": -1,
                    "n_validation_seeds": len(eval_seeds),
                    "validation_seeds": eval_seeds,
                    "seed_stream": "env_eval",
                    "candidate": "initial_untrained_policy",
                    "reportable": False,
                    "economic_safety_floor": reference_validation_score,
                    "eligibility": initial_eligibility,
                },
                sort_keys=False,
            )
        )
        logger.info(
            "initial diagnostic validation %s %.6f (non-reportable safety floor)",
            checkpoint_metric,
            initial_validation["checkpoint_score"],
        )

    if reference_validation_score is None:
        raise AssertionError("initial economic safety reference was not initialized")

    new_csv = start_episode == 0 or not paths.metrics_csv.exists()
    csv_file = paths.metrics_csv.open("w" if new_csv else "a", newline="")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(METRICS_COLUMNS)
        csv_file.flush()

    grids_path = paths.run_dir / "realized_grids.jsonl"
    grids_file = grids_path.open("w" if new_csv else "a")
    forecast_path = paths.run_dir / "h_forecasts_train.csv"
    forecast_file = forecast_path.open("w" if new_csv else "a", newline="")
    forecast_fields = [
        "episode", "env_seed", "time", "decision_index", "phase", "time_to_close",
        "h_cl", "s_mid", "s_cl", "h_error", "mid_error",
    ]
    forecast_writer = csv.DictWriter(forecast_file, fieldnames=forecast_fields)
    if new_csv:
        forecast_writer.writeheader()
    training_episode_seeds: list[int] = []

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
        training_episode_seeds.append(env_seed)
        grids_file.write(
            json.dumps(
                {
                    "split": "train",
                    "episode": episode,
                    "env_seed": env_seed,
                    **res.realized_grid,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        for row in res.forecast_records:
            forecast_writer.writerow(
                {"episode": episode, "env_seed": env_seed, **row}
            )

        eval_return_mean: Optional[float] = None
        eval_pnl_mean: Optional[float] = None
        eval_risk_adjusted_pnl_mean: Optional[float] = None
        eval_checkpoint_score: Optional[float] = None
        eval_checkpoint_eligible: Optional[bool] = None
        eval_checkpoint_reportable: Optional[bool] = None
        stop_after_episode = False
        if (episode + 1) % cfg.rl.validation_frequency_episodes == 0:
            validation = _run_eval(
                validation_env, agent, eval_seeds, chi, checkpoint_metric
            )
            eval_return_mean = validation["return_mean"]
            eval_pnl_mean = validation["pnl_mean"]
            eval_risk_adjusted_pnl_mean = validation["risk_adjusted_pnl_mean"]
            eval_checkpoint_score = validation["checkpoint_score"]
            eligibility = _checkpoint_eligibility(cfg, agent)
            eval_checkpoint_eligible = bool(eligibility["eligible"])
            eval_checkpoint_reportable = bool(
                eval_checkpoint_eligible
                and (
                    not cfg.rl.checkpoint_require_initial_improvement
                    or eval_checkpoint_score > reference_validation_score
                )
            )
            mature_improved = bool(
                eval_checkpoint_eligible
                and eval_checkpoint_score > best_mature_validation_score
            )
            if mature_improved:
                best_mature_validation_score = eval_checkpoint_score
                best_mature_episode = episode
                validation_evals_without_improvement = 0
                agent.save(paths.checkpoints / "best_mature.pt")
                _write_best_selection(
                    paths.checkpoints / "best_mature_selection.yaml",
                    cfg=cfg,
                    metric=checkpoint_metric,
                    value=best_mature_validation_score,
                    episode=best_mature_episode,
                    eval_seeds=eval_seeds,
                    eligibility=eligibility,
                    safety_reference_score=reference_validation_score,
                    reportable=eval_checkpoint_reportable,
                    candidate="periodic_mature_validation",
                )
            elif eval_checkpoint_eligible:
                validation_evals_without_improvement += 1
                stop_after_episode = (
                    validation_evals_without_improvement
                    >= cfg.rl.validation_patience_evals
                )
            if (
                eval_checkpoint_reportable
                and eval_checkpoint_score > best_validation_score
            ):
                best_validation_score = eval_checkpoint_score
                best_episode = episode
                agent.save(paths.checkpoints / "best.pt")
                _write_best_selection(
                    paths.checkpoints / "best_selection.yaml",
                    cfg=cfg,
                    metric=checkpoint_metric,
                    value=best_validation_score,
                    episode=best_episode,
                    eval_seeds=eval_seeds,
                    eligibility=eligibility,
                    safety_reference_score=reference_validation_score,
                    reportable=True,
                )
            if eval_checkpoint_eligible:
                logger.info(
                    "episode %d: mature validation %s %.6f (best mature %.6f; "
                    "safety floor %.6f; reportable=%s); economic return %.4f; "
                    "maturity updates=%s",
                    episode,
                    checkpoint_metric,
                    eval_checkpoint_score,
                    best_mature_validation_score,
                    reference_validation_score,
                    eval_checkpoint_reportable,
                    eval_return_mean,
                    eligibility["observed_updates"],
                )
            else:
                logger.info(
                    "episode %d: diagnostic validation %s %.6f is not eligible; "
                    "update deficits=%s",
                    episode,
                    checkpoint_metric,
                    eval_checkpoint_score,
                    eligibility["update_deficits"],
                )

        if (episode + 1) % hp.checkpoint_interval_episodes == 0:
            ckpt = paths.checkpoints / f"ckpt_ep{episode + 1}.pt"
            agent.save(ckpt, include_replay=True)
            torch.save(
                {
                    "episode": episode,
                    "checkpoint_metric": checkpoint_metric,
                    "best_validation_score": best_validation_score,
                    "best_episode": best_episode,
                    "reference_validation_score": reference_validation_score,
                    "best_mature_validation_score": best_mature_validation_score,
                    "best_mature_episode": best_mature_episode,
                    "validation_evals_without_improvement": validation_evals_without_improvement,
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
                agent.checkpoint_update_counts,
                eval_return_mean,
                eval_pnl_mean,
                eval_risk_adjusted_pnl_mean,
                eval_checkpoint_score,
                eval_checkpoint_eligible,
                eval_checkpoint_reportable,
                wall,
            )
        )
        csv_file.flush()
        grids_file.flush()
        forecast_file.flush()
        if stop_after_episode:
            logger.info(
                "early stopping after episode %d: no validation improvement for %d evaluations",
                episode,
                validation_evals_without_improvement,
            )
            break

    agent.save(paths.checkpoints / "final.pt")
    checkpoint_failure: str | None = None
    if best_episode is None:
        final_eligibility = _checkpoint_eligibility(cfg, agent)
        if not bool(final_eligibility["eligible"]):
            checkpoint_failure = (
                "training ended before any checkpoint became reportable; "
                f"phase-update deficits={final_eligibility['update_deficits']}"
            )
        else:
            # This covers a budget whose final mature state falls between
            # periodic validation boundaries. It uses the same fixed split.
            validation = _run_eval(
                validation_env, agent, eval_seeds, chi, checkpoint_metric
            )
            final_score = validation["checkpoint_score"]
            final_episode = max(
                0, start_episode + len(training_episode_seeds) - 1
            )
            final_reportable = bool(
                not cfg.rl.checkpoint_require_initial_improvement
                or final_score > reference_validation_score
            )
            if final_score > best_mature_validation_score:
                best_mature_validation_score = final_score
                best_mature_episode = final_episode
                agent.save(paths.checkpoints / "best_mature.pt")
                _write_best_selection(
                    paths.checkpoints / "best_mature_selection.yaml",
                    cfg=cfg,
                    metric=checkpoint_metric,
                    value=best_mature_validation_score,
                    episode=best_mature_episode,
                    eval_seeds=eval_seeds,
                    eligibility=final_eligibility,
                    safety_reference_score=reference_validation_score,
                    reportable=final_reportable,
                    candidate="end_of_run_mature_validation",
                )
            if final_reportable:
                best_validation_score = final_score
                best_episode = final_episode
                agent.save(paths.checkpoints / "best.pt")
                _write_best_selection(
                    paths.checkpoints / "best_selection.yaml",
                    cfg=cfg,
                    metric=checkpoint_metric,
                    value=best_validation_score,
                    episode=best_episode,
                    eval_seeds=eval_seeds,
                    eligibility=final_eligibility,
                    safety_reference_score=reference_validation_score,
                    reportable=True,
                    candidate="end_of_run_validation",
                )
            else:
                checkpoint_failure = (
                    "no mature checkpoint beat the non-reportable initial "
                    f"economic safety floor: best_mature="
                    f"{best_mature_validation_score:.12g}, "
                    f"initial={reference_validation_score:.12g}"
                )
    if checkpoint_failure is not None:
        (paths.checkpoints / "selection_failure.yaml").write_text(
            yaml.safe_dump(
                {
                    "reason": checkpoint_failure,
                    "metric": checkpoint_metric,
                    "initial_validation_score": reference_validation_score,
                    "best_mature_validation_score": (
                        None
                        if not np.isfinite(best_mature_validation_score)
                        else best_mature_validation_score
                    ),
                    "best_mature_episode": best_mature_episode,
                    "maturity": _checkpoint_eligibility(cfg, agent),
                },
                sort_keys=False,
            )
        )
    csv_file.close()
    grids_file.close()
    forecast_file.close()
    (paths.run_dir / "split_seeds.yaml").write_text(
        yaml.safe_dump(
            {
                "master_seed": master_seed,
                "configured_seed_list": list(cfg.experiment.seeds),
                "training_seed_stream": "env_train",
                "training_episode_seeds_this_invocation": training_episode_seeds,
                "validation_seed_stream": "env_eval",
                "validation_episode_seeds": eval_seeds,
                "normalizer_seed_stream": "normalizer_env",
                "final_test_seed_stream": "env_final_eval",
                "final_test_size": cfg.rl.test_size,
            },
            sort_keys=False,
        )
    )
    completed_episodes = (
        start_episode + len(training_episode_seeds)
        if training_episode_seeds
        else start_episode
    )
    logger.info(
        "done: %d episodes completed; final checkpoint at %s",
        completed_episodes,
        paths.checkpoints,
    )
    if checkpoint_failure is not None:
        raise RuntimeError(checkpoint_failure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
