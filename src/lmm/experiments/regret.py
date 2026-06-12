"""PRegret(T) from saved per-episode records (Phase 4).

PRegret(T) = sum_{e=1}^{E} (V_0^{pi_benchmark}(x_{0,e}) - V_0^{pi_e}(x_{0,e})),
T = (m+2)E, estimated per episode with common random numbers (same env seed
for the learned policy and the benchmark within an episode) — paper
`sec:learning` regret definition (the only part of Sec. 4 that is kept, D9).
V_0 is the paper's chi-DISCOUNTED value, so ``--returns discounted`` is the
default; ``--returns undiscounted`` matches the reported evaluation returns.
Reads eval/records.csv written by evaluate.py and writes
eval/regret_<benchmark>.csv (per-episode and cumulative).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from lmm.config import load_config

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-regret",
        description="Compute PRegret(T) curves from saved per-episode records.",
    )
    parser.add_argument("--run-dir", required=True, help="training run directory")
    parser.add_argument("--benchmark", default="as", choices=["as", "twap"], help="pi_benchmark")
    parser.add_argument(
        "--policy", default="dqn", choices=["dqn", "initial"], help="the evaluated policy pi"
    )
    parser.add_argument(
        "--returns",
        default="discounted",
        choices=["discounted", "undiscounted"],
        help="V_0 estimator: chi-discounted (paper definition; default) or undiscounted",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config_resolved.yaml")
    col = "return_disc" if args.returns == "discounted" else "return_undisc"

    by_policy: dict[str, dict[int, dict[str, str]]] = {}
    with (run_dir / "eval" / "records.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            by_policy.setdefault(row["policy"], {})[int(row["episode"])] = row

    bench = by_policy.get(args.benchmark)
    pol = by_policy.get(args.policy)
    if not bench or not pol:
        raise SystemExit(
            f"records.csv lacks policy rows for {args.benchmark!r}/{args.policy!r}; "
            "run lmm-evaluate first"
        )
    episodes = sorted(bench)
    if episodes != sorted(pol):
        raise SystemExit("episode sets differ between benchmark and policy records")

    out_path = run_dir / "eval" / f"regret_{args.benchmark}.csv"
    cum = 0.0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "env_seed", "v_benchmark", "v_policy", "regret", "cum_regret"])
        for e in episodes:
            rb, rp = bench[e], pol[e]
            if rb["env_seed"] != rp["env_seed"]:
                raise SystemExit(
                    f"CRN violation at episode {e}: env seeds differ "
                    f"({rb['env_seed']} vs {rp['env_seed']})"
                )
            v_b, v_p = float(rb[col]), float(rp[col])
            regret = v_b - v_p
            cum += regret
            writer.writerow(
                [e, rb["env_seed"]] + [format(v, ".17g") for v in (v_b, v_p, regret, cum)]
            )

    # Episode horizon m + 2 = tau_cl + 1 nominal steps (decisions at 0..m
    # plus the terminal clearing) — paper grid convention.
    E = len(episodes)
    T = (cfg.grid.tau_cl + 1) * E
    print(
        f"PRegret(T) = {cum:.6f}  [benchmark={args.benchmark}, policy={args.policy}, "
        f"{args.returns} returns, E={E}, T=(m+2)E={T}]"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
