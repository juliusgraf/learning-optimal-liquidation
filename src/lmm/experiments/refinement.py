"""Coupled rough-Heston diagnostics; candidate meshes are not accuracy thresholds."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time
import tracemalloc

import numpy as np

from lmm.config import (
    economic_evaluation_config,
    load_config,
    price_generator_identity,
    to_dict,
)
from lmm.env.mdp import MarketMakingEnv
from lmm.market.generator import sample_episode_realization
from lmm.market.midprice import (
    RoughHestonMidPrice,
    aggregate_brownian_increments,
    internal_time_grid,
)


class CoupledMidPrice(RoughHestonMidPrice):
    """Diagnostic injection, kept out of policy observations and production config."""

    def __init__(self, params, grid, fine_times, fine_increments):
        super().__init__(params, grid)
        self._fine_times = fine_times
        self._fine_increments = fine_increments

    def prepare_grid(self, decision_times, opening, *, increments=None):
        coarse = internal_time_grid(
            decision_times, opening, self.params.rough_heston_max_step_minutes
        )
        coupled = aggregate_brownian_increments(
            self._fine_times, self._fine_increments, coarse
        )
        super().prepare_grid(decision_times, opening, increments=coupled)


class CoupledEnvironment(MarketMakingEnv):
    """Isolate each auction proposal's marks from state-dependent eligibility."""

    def reset(self, *, seed=None, options=None):
        self._coupling_seed = seed
        return super().reset(seed=seed, options=options)

    def _prepare_current_auction_proposals(self):
        streams = np.random.SeedSequence(
            [self._coupling_seed, self._auction_index, 190771]
        ).spawn(7)
        generators = [np.random.default_rng(stream) for stream in streams]
        self._pending_auction_events = self._generator.auction_flow.step(
            generators[0], proposal_rngs=generators[1:]
        )
        self._generator.auction_flow.assert_valid()


def moments(values):
    values = np.asarray(values, dtype=float)
    return dict(
        mean=float(values.mean()),
        std=float(values.std()),
        second_moment=float(np.mean(values**2)),
    )


def diagnose(
    cfg,
    *,
    episodes=32,
    seed=927401,
    steps=(1.0, 0.5, 0.25),
    run_dir=None,
    checkpoint="best",
):
    if episodes <= 0 or not steps:
        raise ValueError("positive episodes and at least one candidate step required")
    if cfg.midprice.model != "rough_heston":
        raise ValueError("refinement diagnostics require rough Heston")
    if (
        cfg.grid.time_unit != "minutes"
        or cfg.midprice.rough_heston.s_star != 252 * 6.5 * 60
    ):
        raise ValueError("diagnostics require minutes and s_star=98280")
    # Validate candidates even if constructed outside the config loader.
    for step in steps:
        internal_time_grid([0.0], 1.0, step)
    if any(step is None for step in steps):
        raise ValueError(
            "candidate steps must be positive; legacy is included automatically"
        )
    steps = [None, *sorted(set(steps), reverse=True)]
    labels = ["legacy" if step is None else f"{step:g}m" for step in steps]
    records = {label: [] for label in labels}
    price_arrays = {label: [] for label in labels}
    returns = {label: [] for label in labels}
    terminal_returns = {label: [] for label in labels}
    fixed = None
    artifact_notes = {
        "policy": None,
        "normalization": None,
        "calibration": "all supplied config quantities held fixed, including forecast weights",
        "reference_conditioning": "none without a trained policy",
    }
    if run_dir is not None:
        from lmm.experiments.train import make_agent
        from lmm.rl.loops import SEED_COMPONENTS
        from lmm.utils.seeding import seed_everything

        run_dir = Path(run_dir)
        fixed = make_agent(
            cfg, seed_everything(cfg.experiment.master_seed, SEED_COMPONENTS)
        )
        path = run_dir / "checkpoints" / f"{checkpoint}.pt"
        fixed.load(
            path
        )  # Validate against ORIGINAL config, before intentional sensitivity shifts.
        artifact_notes.update(
            policy=str(path.resolve()),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            normalization="frozen normalizer embedded in checkpoint",
            reference_conditioning="frozen inventory reference embedded in checkpoint",
        )
    for episode in range(episodes):
        env_seed = seed + episode
        tape = sample_episode_realization(
            np.random.default_rng(env_seed), cfg.clob_flow, cfg.grid
        )
        fine_times = internal_time_grid(
            tape.grid.clob_times, cfg.grid.tau_op, min(steps[1:])
        )
        # Two independent drivers, each summed separately for every coarse mesh.
        streams = np.random.SeedSequence([seed, episode, 80931]).spawn(2)
        dt_years = np.diff(fine_times) / (252 * 6.5 * 60)
        fine_dw = np.column_stack(
            [
                np.random.default_rng(stream).normal(size=len(dt_years))
                * np.sqrt(dt_years)
                for stream in streams
            ]
        )
        for label, step in zip(labels, steps):
            rough = replace(
                cfg.midprice.rough_heston, rough_heston_max_step_minutes=step
            )
            variant = replace(
                economic_evaluation_config(cfg),
                midprice=replace(cfg.midprice, rough_heston=rough),
            )
            # Config objects are intentionally replaced WITHOUT refitting or claiming
            # artifact compatibility: this command is exclusively fixed-policy sensitivity.
            tracemalloc.start()
            start = time.perf_counter()
            model = CoupledMidPrice(rough, cfg.grid, fine_times, fine_dw)
            env = CoupledEnvironment(variant, model)
            env.reset(seed=env_seed)
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            indices = np.searchsorted(
                model._internal_grid, np.r_[tape.grid.clob_times, cfg.grid.tau_op]
            )
            prices = model.raw_price_path[indices]
            rounded = np.r_[env._clob_mid_values, env._prepared_frozen_mid]
            diag = model.variance_diagnostics()
            if not diag["finite"]:
                raise FloatingPointError(
                    f"nonfinite path: episode={episode}, mesh={label}"
                )
            row = dict(
                episode=episode,
                env_seed=env_seed,
                internal_nodes=diag["nodes"],
                decision_count=len(tape.grid.decision_times),
                observation_prices=len(prices),
                negative_nodes=diag["negative_nodes"],
                negative_time_fraction=diag["negative_time_fraction"],
                minimum_variance=diag["minimum_variance"],
                reset_seconds=elapsed,
                reset_peak_traced_bytes=peak,
                generator_identity=price_generator_identity(variant),
            )
            if fixed is not None:
                from lmm.experiments.train import wrap_env_for_agent
                from lmm.rl.loops import run_episode

                start = time.perf_counter()
                result = run_episode(
                    wrap_env_for_agent(env, cfg),
                    fixed,
                    env_seed,
                    chi=cfg.rl.chi,
                    train=False,
                )
                row.update(
                    policy_episode_seconds=time.perf_counter() - start,
                    shortfall_bps=-result.pnl_bps,
                    penalized_shortfall_bps=-result.risk_adjusted_pnl_bps,
                    forecast_mae=result.h_forecast_mae,
                    forecast_rmse=result.h_forecast_rmse,
                    transitions=result.n_steps,
                )
            records[label].append(row)
            price_arrays[label].append((prices, rounded))
            returns[label].extend(np.diff(np.log(prices)))
            terminal_returns[label].append(np.log(prices[-1] / prices[0]))
    summary = {}
    for label in labels:
        rows = records[label]
        raw = np.concatenate([pair[0] for pair in price_arrays[label]])
        diff = np.concatenate(
            [
                pair[0] - fine[0]
                for pair, fine in zip(price_arrays[label], price_arrays[labels[-1]])
            ]
        )
        rounded_diff = np.concatenate(
            [
                pair[1] - fine[1]
                for pair, fine in zip(price_arrays[label], price_arrays[labels[-1]])
            ]
        )
        summary[label] = dict(
            revealed_raw_price=moments(raw),
            interval_log_return=moments(returns[label]),
            opening_log_return=moments(terminal_returns[label]),
            price_rmse_vs_finest=float(np.sqrt(np.mean(diff**2))),
            price_max_abs_vs_finest=float(np.max(np.abs(diff))),
            rounded_price_rmse_vs_finest=float(np.sqrt(np.mean(rounded_diff**2))),
            negative_node_fraction=sum(r["negative_nodes"] for r in rows)
            / sum(r["internal_nodes"] for r in rows),
            negative_time_fraction=float(
                np.mean([r["negative_time_fraction"] for r in rows])
            ),
            minimum_variance=min(r["minimum_variance"] for r in rows),
            mean_internal_nodes=float(np.mean([r["internal_nodes"] for r in rows])),
            mean_reset_seconds=float(np.mean([r["reset_seconds"] for r in rows])),
            max_reset_peak_traced_bytes=max(r["reset_peak_traced_bytes"] for r in rows),
        )
        if fixed is not None:
            for metric in (
                "shortfall_bps",
                "penalized_shortfall_bps",
                "forecast_mae",
                "forecast_rmse",
            ):
                values = [r[metric] for r in rows]
                paired = np.asarray(values) - np.asarray(
                    [r[metric] for r in records["legacy"]]
                )
                summary[label][metric] = moments(values)
                summary[label][metric + "_paired_delta_vs_legacy"] = dict(
                    **moments(paired),
                    standard_error=float(np.std(paired, ddof=1) / np.sqrt(episodes))
                    if episodes > 1
                    else None,
                )
    return dict(
        schema="rough-heston-refinement-diagnostic-v1",
        seed=seed,
        episodes=episodes,
        steps_minutes=steps,
        finest_reference_minutes=min(steps[1:]),
        interpretation="candidate meshes only; no established accuracy threshold or convergence claim",
        comparison="fixed-policy numerical sensitivity"
        if fixed
        else "price-generator numerical sensitivity",
        retrained_policy_comparison="not performed",
        artifacts_held_fixed=artifact_notes,
        coupling="same exogenous CLOB tape; two independent finest-grid Brownian drivers summed separately; auction indicators and marks use common independent streams keyed by episode, auction decision, and proposal",
        auction_acceptance="state-dependent eligibility, accepted updates and cancellation targets may differ with changed prices; proposal streams prevent conditional draws shifting other proposals",
        weighting=dict(
            price_moments="pooled revealed nodes including initial and opening",
            interval_returns="pooled decision-interval log returns, not annualized",
            opening_returns="one per episode",
            negative_node_fraction="pooled internal nodes including initial and opening, before positive-part truncation",
            negative_time_fraction="left-endpoint indicator weighted by interval duration, then equal episodes",
            forecasts="episode metric over original decision observations, then equal episodes",
        ),
        costs="wall time and tracemalloc peak for complete env construction/reset with coupled paths; excludes common fine-driver generation and policy loading; traced allocations, not process RSS",
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            platform=platform.platform(),
        ),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "config.py",
                Path(__file__).parents[1] / "market/midprice.py",
                Path(__file__).parents[1] / "market/auction.py",
                Path(__file__).parents[1] / "env/mdp.py",
            )
        },
        config=to_dict(cfg),
        summary=summary,
        per_episode=records,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument(
        "--run-dir", type=Path, help="load and hold this trained run fixed"
    )
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=927401)
    parser.add_argument("--steps", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.run_dir and args.config:
        parser.error("use either --run-dir or --config")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic {args.output}")
    cfg = (
        load_config(args.run_dir / "config_resolved.yaml")
        if args.run_dir
        else load_config(
            *(
                args.config
                or ["configs/base.yaml", "configs/synthetic_rough_heston.yaml"]
            )
        )
    )
    result = diagnose(
        cfg,
        episodes=args.episodes,
        seed=args.seed,
        steps=args.steps,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
