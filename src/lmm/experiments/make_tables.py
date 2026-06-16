"""Regenerate all tables from saved run outputs (Phase 7).

Emits BOTH ``.tex`` (booktabs) and ``.csv`` into the output dir. Reported
returns are UNDISCOUNTED episode sums (stated in each caption; CLAUDE.md
objective conventions). All inputs are saved run-dir artifacts — no env
stepping.

``--run-dir`` accepts one or more run dirs (the cross-algorithm tables need the
sibling DDPG/TD3/SAC runs). Tables:
  - eval_summary_final          (synthetic group)   -> replaces tab:eval_summary_final
  - dqn_results_full (+ improvements)  (historical) -> replaces tab:dqn_results_full
  - params_generative / params_midprice (primary config) -> replaces tab:params_generative
  - hyperparams_<algo>          (per resolved algo config)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence

from lmm.experiments import plotting as P
from lmm.experiments import tables as T

__all__ = ["build_parser", "main"]


def _group_by_setting(runs):
    groups: dict[str, list] = {}
    for r in runs:
        groups.setdefault(r.setting, []).append(r)
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-make-tables",
        description="Regenerate tables from saved run outputs.",
    )
    parser.add_argument("--run-dir", required=True, nargs="+", help="run directory/ies")
    parser.add_argument("--out", default=None, help="output dir (default: <run>/tables)")
    parser.add_argument(
        "--multiseed", action="store_true",
        help="build ONLY the cross-seed IQM/CI aggregate tables (the run dirs "
        "should span >= 2 seeds of the same setting). Skips the single-seed and "
        "parameter/hyperparameter tables.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = [Path(d) for d in args.run_dir]
    out = Path(args.out) if args.out else run_dirs[0] / "tables"
    out.mkdir(parents=True, exist_ok=True)

    runs = P.collect_runs(run_dirs)
    written: list[Path] = []

    def safe(fn, *, what: str):
        try:
            fn()
        except Exception as exc:  # one bad table must not abort the batch
            warnings.warn(f"table {what!r} failed: {exc}", stacklevel=2)

    # (a)/(b) per-setting result tables. With --multiseed, build ONLY the
    # cross-seed aggregates (the single-seed builders would silently keep just
    # one seed's run per algo if handed pooled seeds).
    for setting, group in _group_by_setting(runs).items():
        if not any(r.records is not None for r in group):
            continue
        if args.multiseed:
            n_seeds = len({r.seed for r in group if r.seed is not None})
            if n_seeds < 2:
                warnings.warn(f"--multiseed: {setting} has <2 seeds; skipping", stacklevel=2)
                continue
            if setting == "historical_sp500":
                def _hist_ms(group=group):
                    table = T.build_historical_results_multiseed(group)
                    written.extend(T.write_table(table, out, "dqn_results_multiseed"))
                safe(_hist_ms, what=f"historical_multiseed[{setting}]")
            else:
                def _eval_ms(group=group):
                    table = T.build_eval_summary_multiseed(group)
                    written.extend(T.write_table(table, out, "eval_summary_multiseed"))
                safe(_eval_ms, what=f"eval_summary_multiseed[{setting}]")
            continue
        if setting == "historical_sp500":
            def _hist(group=group):
                ret_t, imp_t = T.build_historical_results(group)
                written.extend(T.write_table(ret_t, out, "dqn_results_full"))
                written.extend(T.write_table(imp_t, out, "dqn_results_improvements"))
            safe(_hist, what=f"historical[{setting}]")
        else:
            def _eval(group=group):
                table = T.build_eval_summary(group)
                written.extend(T.write_table(table, out, "eval_summary_final"))
            safe(_eval, what=f"eval_summary[{setting}]")

    if args.multiseed:
        print(f"wrote {len(written)} table files to {out}")
        return 0

    # (c) parameter tables from the PRIMARY run's resolved config.
    def _params():
        cfg = P.read_config(run_dirs[0])
        for name, table in T.build_param_tables(cfg).items():
            written.extend(T.write_table(table, out, name))
    safe(_params, what="params")

    # (c) per-algorithm hyperparameter tables (one per distinct algo present).
    seen: dict[str, object] = {}
    for r in runs:
        if r.cfg.algo is not None and r.algo not in seen:
            seen[r.algo] = r.cfg
    for algo, cfg in seen.items():
        def _hp(cfg=cfg, algo=algo):
            table = T.build_hyperparam_table(cfg)
            if table is not None:
                written.extend(T.write_table(table, out, f"hyperparams_{algo}"))
        safe(_hp, what=f"hyperparams_{algo}")

    print(f"wrote {len(written)} table files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
