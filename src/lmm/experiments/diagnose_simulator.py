"""Fast, policy-free diagnostics for the shared simulator calibration.

This command deliberately submits no strategic CLOB liquidity and one auction
no-op.  It therefore probes the exogenous book/carry-over contract without any
RL optimization.  The no-op path is conservative: an active strategic ask can
absorb buy flow that would otherwise consume exogenous depth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from lmm.config import add_config_cli, config_from_args
from lmm.env.action_spaces import AuctionAction, ClobAction
from lmm.env.mdp import make_env

__all__ = ["diagnose", "build_parser", "main"]


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def diagnose(
    cfg,
    *,
    repo_root: str | Path,
    symbols: Sequence[str] | None,
    episodes: int,
    seed: int,
    data_split: str = "train",
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    historical = cfg.midprice.historical if cfg.midprice.model == "historical" else None
    if historical is not None:
        targets = list(symbols or historical.symbols)
        unknown = sorted(set(targets) - set(historical.symbols))
        if unknown:
            raise ValueError(f"unknown historical symbol(s): {unknown}")
    elif cfg.midprice.model == "rough_heston":
        if symbols:
            raise ValueError("--symbols applies only to historical mid-price paths")
        targets = ["rough_heston"]
    else:
        raise ValueError(f"unsupported midprice model {cfg.midprice.model!r}")

    per_symbol: dict[str, Any] = {}
    pooled_slopes: list[float] = []
    pooled_ranges: list[float] = []
    pooled_residuals: list[float] = []
    pooled_arrivals: list[float] = []
    for symbol_index, symbol in enumerate(targets):
        env = make_env(
            cfg,
            symbol=symbol if historical is not None else None,
            repo_root=repo_root,
            data_split=data_split,
        )
        slopes: list[float] = []
        ranges: list[float] = []
        residuals: list[float] = []
        arrivals: list[float] = []
        steps: list[float] = []
        for episode in range(episodes):
            env_seed = int(seed + symbol_index * 1_000_003 + episode)
            env.reset(seed=env_seed)
            n_arrivals = 0
            n_clob_decisions = 0
            while env.phase == "clob":
                _, _, _, _, info = env.step(ClobAction(0.0, 0))
                n_arrivals += int(info["n_buy_step"]) + int(info["n_sell_step"])
                n_clob_decisions += 1
            residual = float(
                np.sum(env._generator.book.ask_volumes)  # noqa: SLF001
                + np.sum(env._generator.book.bid_volumes)  # noqa: SLF001
            )
            mid_range_ticks = float(
                (np.max(env._clob_mid_values) - np.min(env._clob_mid_values))  # noqa: SLF001
                / cfg.grid.alpha
            )
            _, _, _, _, auction_info = env.step(AuctionAction(0.0, 0, 0))
            slopes.append(float(auction_info["carryover_slope"]))
            ranges.append(mid_range_ticks)
            residuals.append(residual)
            arrivals.append(float(n_arrivals))
            steps.append(float(n_clob_decisions))

        slope_array = np.asarray(slopes)
        per_symbol[symbol] = {
            "episodes": episodes,
            "fallback_rate": float(np.mean(slope_array < cfg.auction_flow.D_mu)),
            "positive_carryover_rate": float(np.mean(slope_array > 0.0)),
            "carryover_slope": _summary(slopes),
            "mid_range_ticks": _summary(ranges),
            "final_residual_exogenous_volume": _summary(residuals),
            "clob_arrivals_both_sides": _summary(arrivals),
            "clob_decisions": _summary(steps),
        }
        pooled_slopes.extend(slopes)
        pooled_ranges.extend(ranges)
        pooled_residuals.extend(residuals)
        pooled_arrivals.extend(arrivals)

    pooled_array = np.asarray(pooled_slopes)
    return {
        "contract": "no strategic CLOB order; first auction action is no-op",
        "clearing_mechanism": cfg.auction_flow.clearing_mechanism,
        "artifact_schema_version": cfg.experiment.artifact_schema_version,
        "midprice_model": cfg.midprice.model,
        "data_split": data_split,
        "episodes_per_symbol": episodes,
        "symbols": targets,
        "parameters": {
            "lambda0": cfg.clob_flow.lambda0,
            "V_inf": cfg.clob_flow.V_inf,
            "rho_lob": cfg.clob_flow.rho_lob,
            "L_max": cfg.clob_flow.L_max,
            "D_mu": cfg.auction_flow.D_mu,
            "auction_B_inf": cfg.auction_flow.B_inf,
            "agent_slope_choices": list(cfg.actions.auction_K_multipliers),
            "agent_B_inf": cfg.actions.B_inf,
            "agent_B_max": cfg.actions.B_max,
            "agent_local_offset_max": cfg.actions.B_max,
        },
        "pooled": {
            "fallback_rate": float(np.mean(pooled_array < cfg.auction_flow.D_mu)),
            "positive_carryover_rate": float(np.mean(pooled_array > 0.0)),
            "carryover_slope": _summary(pooled_slopes),
            "mid_range_ticks": _summary(pooled_ranges),
            "final_residual_exogenous_volume": _summary(pooled_residuals),
            "clob_arrivals_both_sides": _summary(pooled_arrivals),
        },
        "per_symbol": per_symbol,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-diagnose-simulator",
        description="Run short shared-simulator carry-over/liquidity diagnostics without training.",
    )
    add_config_cli(parser)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=70_001)
    parser.add_argument("--data-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--assert-ready", action="store_true")
    parser.add_argument("--max-fallback-rate", type=float, default=0.10)
    parser.add_argument("--min-positive-carryover-rate", type=float, default=0.90)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    historical = cfg.midprice.historical if cfg.midprice.model == "historical" else None
    if historical is None and args.symbols:
        parser.error("--symbols applies only to historical mid-price paths")
    symbols = (args.symbols or list(historical.symbols)) if historical is not None else None
    report = diagnose(
        cfg,
        repo_root=args.repo_root,
        symbols=symbols,
        episodes=args.episodes,
        seed=args.seed,
        data_split=args.data_split,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    if args.assert_ready:
        fallback_offenders = {
            symbol: values["fallback_rate"]
            for symbol, values in report["per_symbol"].items()
            if values["fallback_rate"] > args.max_fallback_rate
        }
        carryover_offenders = {
            symbol: values["positive_carryover_rate"]
            for symbol, values in report["per_symbol"].items()
            if values["positive_carryover_rate"]
            < args.min_positive_carryover_rate
        }
        if fallback_offenders or carryover_offenders:
            print(
                "simulator calibration gate failed: "
                f"fallback>{args.max_fallback_rate:g}={fallback_offenders}; "
                "positive carry-over"
                f"<{args.min_positive_carryover_rate:g}={carryover_offenders}"
            )
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
