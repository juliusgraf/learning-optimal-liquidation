"""Regenerate all tables from saved run outputs (Phase 7).

Emits BOTH ``.tex`` (booktabs) and ``.csv`` into the output dir. Reported
returns are UNDISCOUNTED episode sums (stated in each caption). All inputs are saved run-dir artifacts — no env
stepping.

``--run-dir`` accepts one or more run dirs (the cross-algorithm tables need the
sibling DDPG/TD3/SAC runs). Tables:
  - eval_summary_final          (synthetic group)   -> replaces tab:eval_summary_final
  - historical_results_full (+ improvements)  (historical, currency)
  - historical_results_full_bps (+ improvements_bps) (historical, normalized)
  - params_generative / params_midprice (primary config) -> replaces tab:params_generative
  - hyperparams_<algo>          (per resolved algo config)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence

from lmm.experiments import plotting as P
from lmm.experiments import publication
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--multiseed", action="store_true",
        help="build ONLY the cross-seed IQM/CI aggregate tables (the run dirs "
        "should span >= 2 seeds of the same setting). Skips the single-seed and "
        "parameter/hyperparameter tables.",
    )
    mode.add_argument(
        "--cross-treatment",
        action="store_true",
        help="build ONLY the paired multiseed synthetic-treatment contrast table "
        "and its seed-level provenance CSV",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="reject smoke/incomplete artifacts (requires the resolved 800-episode "
        "training budget, 100 held-out episodes, and a mature best checkpoint)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dirs = [Path(d) for d in args.run_dir]
    out = Path(args.out) if args.out else run_dirs[0] / "tables"
    out.mkdir(parents=True, exist_ok=True)

    runs = P.collect_runs(run_dirs)
    written: list[Path] = []
    failures: list[str] = []

    if len(runs) != len(run_dirs):
        failures.append(
            f"resolved only {len(runs)} of {len(run_dirs)} requested run directories"
        )
    settings = {run.setting for run in runs}
    if len(settings) > 1 and not args.cross_treatment:
        failures.append(
            "one table-generation invocation must contain exactly one setting; "
            f"got {sorted(settings)}"
        )

    def safe(fn, *, what: str):
        try:
            fn()
        except Exception as exc:
            message = f"table {what!r} failed: {exc}"
            failures.append(message)

    # Different settings write the same canonical filenames and would silently
    # overwrite one another.  Fail before producing a mixed publication set.
    if len(settings) > 1 and not args.cross_treatment:
        for message in failures:
            warnings.warn(message, stacklevel=2)
        return 1

    if args.publication:
        safe(
            lambda: publication.validate_publication_runs(runs),
            what="publication artifact validation",
        )
        if failures and not args.cross_treatment:
            for message in failures:
                warnings.warn(message, stacklevel=2)
            return 1

    if args.cross_treatment:
        def _treatments():
            publication.validate_treatment_run_configs(runs)
            table, paired = T.build_synthetic_treatment_contrasts(runs)
            written.extend(
                T.write_table(
                    table, out, "synthetic_treatment_contrasts_multiseed"
                )
            )
            provenance = out / "synthetic_treatment_contrasts_by_seed.csv"
            paired.to_csv(provenance, index=False)
            written.append(provenance)

        if not failures:
            safe(_treatments, what="synthetic treatment contrasts")
        if not written:
            failures.append("no cross-treatment table artifacts were generated")
        for path in written:
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty generated table artifact: {path}")
        for message in failures:
            warnings.warn(message, stacklevel=2)
        print(f"wrote {len(written)} table files to {out}")
        return 1 if failures else 0

    # (a)/(b) per-setting result tables. With --multiseed, build ONLY the
    # cross-seed aggregates (the single-seed builders would silently keep just
    # one seed's run per algo if handed pooled seeds).
    for setting, group in _group_by_setting(runs).items():
        if not any(r.records is not None for r in group):
            failures.append(f"{setting}: no eval/records.csv artifacts were supplied")
            continue
        if args.multiseed:
            n_seeds = len({r.seed for r in group if r.seed is not None})
            if n_seeds < 2:
                message = f"--multiseed: {setting} has <2 seeds"
                failures.append(message)
                warnings.warn(message, stacklevel=2)
                continue
            if P.is_historical_setting(setting):
                def _hist_ms(group=group):
                    table = T.build_historical_results_multiseed(group)
                    written.extend(
                        T.write_table(table, out, "historical_results_multiseed")
                    )
                    table_bps = T.build_historical_results_multiseed(
                        group, metric=T.NORMALIZED_PRIMARY_COL
                    )
                    written.extend(
                        T.write_table(
                            table_bps, out, "historical_results_multiseed_bps"
                        )
                    )
                safe(_hist_ms, what=f"historical_multiseed[{setting}]")
            else:
                def _eval_ms(group=group):
                    table = T.build_eval_summary_multiseed(group)
                    written.extend(T.write_table(table, out, "eval_summary_multiseed"))
                safe(_eval_ms, what=f"eval_summary_multiseed[{setting}]")
            continue
        if P.is_historical_setting(setting):
            def _hist(group=group):
                ret_t, imp_t = T.build_historical_results(group)
                written.extend(T.write_table(ret_t, out, "historical_results_full"))
                written.extend(
                    T.write_table(imp_t, out, "historical_results_improvements")
                )
                ret_bps, imp_bps = T.build_historical_results(
                    group, metric=T.NORMALIZED_PRIMARY_COL
                )
                written.extend(
                    T.write_table(ret_bps, out, "historical_results_full_bps")
                )
                written.extend(
                    T.write_table(
                        imp_bps, out, "historical_results_improvements_bps"
                    )
                )
            safe(_hist, what=f"historical[{setting}]")
        else:
            def _eval(group=group):
                table = T.build_eval_summary(group)
                written.extend(T.write_table(table, out, "eval_summary_final"))
            safe(_eval, what=f"eval_summary[{setting}]")

    if args.multiseed:
        if not written:
            failures.append("no multiseed table artifacts were generated")
        for path in written:
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty generated table artifact: {path}")
        for message in failures:
            warnings.warn(message, stacklevel=2)
        print(f"wrote {len(written)} table files to {out}")
        return 1 if failures else 0

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

    if not written:
        failures.append("no table artifacts were generated")
    for path in written:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty generated table artifact: {path}")
    for message in failures:
        warnings.warn(message, stacklevel=2)
    print(f"wrote {len(written)} table files to {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
