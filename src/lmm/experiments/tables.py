"""Table builders + emitters for Phase 7 (booktabs ``.tex`` and ``.csv``).

Builders return a phase-agnostic :class:`Table` (raw cell values + a per-row
format name); the emitters render LaTeX (booktabs, paper-faithful) and a clean
CSV (plain numbers, no thousands separators). All inputs are saved run-dir
artifacts (``eval/records.csv``, ``eval/metadata.yaml``, ``config_resolved.yaml``)
— no env stepping.

Aggregation across runs (figure f / multi-algo tables): the learned policy is
labelled with ``config_resolved.yaml:algo.name``; the canonical
AS/TWAP/initial columns come from the DQN run of the group. Revised
runs use risk-adjusted marked-to-market PnL as the primary comparison outcome;
artifacts from older environment contracts are rejected before these builders
are called.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from lmm.config import ExperimentConfig
from lmm.experiments import stats
from lmm.experiments.plotting import POLICY_LABELS, POLICY_ORDER, RunInfo

__all__ = [
    "Table",
    "Row",
    "to_booktabs",
    "to_csv",
    "build_eval_summary",
    "build_historical_results",
    "build_eval_summary_multiseed",
    "build_historical_results_multiseed",
    "build_param_tables",
    "build_hyperparam_table",
]

PRIMARY_COL = "risk_adjusted_pnl"
PNL_COL = "pnl"
NORMALIZED_PRIMARY_COL = "risk_adjusted_pnl_bps"
OUTCOME_PRIORITY = (PRIMARY_COL,)

OUTCOME_LABELS = {
    PRIMARY_COL: "Risk-adjusted PnL",
    NORMALIZED_PRIMARY_COL: "Risk-adjusted PnL (bps)",
}

OUTCOME_CAPTIONS = {
    PRIMARY_COL: (
        "risk-adjusted marked-to-market P\\&L (PnL less terminal inventory "
        "penalty; currency units)"
    ),
    NORMALIZED_PRIMARY_COL: (
        "risk-adjusted marked-to-market P\\&L normalized by initial notional "
        "and reported in basis points"
    ),
}


# ---------------------------------------------------------------------------
# Table representation + emitters
# ---------------------------------------------------------------------------


@dataclass
class Row:
    label: str
    cells: list[Any]  # aligned to Table.columns; None -> blank
    # one format for the whole row, or a per-cell list aligned to ``cells``.
    fmt: "str | list[str]" = "money"  # money|num2|int|pct|pct_ci|pval|sci|raw|header


@dataclass
class Table:
    columns: list[str]
    rows: list[Row]
    caption: str
    label: str
    row_label_header: str = ""
    note: str = ""
    section_breaks: set[int] = field(default_factory=set)  # \midrule before these rows


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _sci_tex(v: float) -> str:
    if v == 0.0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(v))))
    man = v / 10.0**exp
    return f"${man:.2f} \\times 10^{{{exp}}}$"


def _fmt_tex(value: Any, fmt: str) -> str:
    if _is_blank(value):
        return ""
    if fmt == "header":
        return ""
    if fmt == "money":
        return f"{value:,.1f}"
    if fmt == "num2":
        return f"{value:,.2f}"
    if fmt == "int":
        return f"{int(round(value))}"
    if fmt == "pct":
        return f"{value:+.1f}\\%"
    if fmt == "pct_ci":
        p, lo, hi = value
        return f"{p:+.1f}\\% [{lo:+.1f}, {hi:+.1f}]"
    if fmt == "money_ci":
        p, lo, hi = value
        return f"{p:,.0f} [{lo:,.0f}, {hi:,.0f}]"
    if fmt == "num2_ci":
        p, lo, hi = value
        return f"{p:,.2f} [{lo:,.2f}, {hi:,.2f}]"
    if fmt == "pval":
        return "$<0.001$" if value < 1e-3 else f"{value:.3f}"
    if fmt == "sci":
        return _sci_tex(value)
    return str(value)


def _fmt_csv(value: Any, fmt: str) -> str:
    if _is_blank(value):
        return ""
    if fmt == "header":
        return ""
    if fmt in ("money", "num2"):
        return f"{value:.2f}" if fmt == "num2" else f"{value:.1f}"
    if fmt == "int":
        return f"{int(round(value))}"
    if fmt == "pct":
        return f"{value:.1f}"
    if fmt == "pct_ci":
        p, lo, hi = value
        return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"
    if fmt == "money_ci":
        p, lo, hi = value
        return f"{p:.1f} [{lo:.1f}, {hi:.1f}]"
    if fmt == "num2_ci":
        p, lo, hi = value
        return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"
    if fmt == "pval":
        return f"{value:.3g}"
    if fmt == "sci":
        return f"{value:.3e}"
    return str(value)


def _cell_fmt(row: Row, j: int) -> str:
    return row.fmt[j] if isinstance(row.fmt, list) else row.fmt


def to_booktabs(table: Table) -> str:
    col_fmt = "l" + "c" * len(table.columns)
    out = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join([table.row_label_header, *table.columns]) + " \\\\",
        "\\midrule",
    ]
    for i, row in enumerate(table.rows):
        if i in table.section_breaks:
            out.append("\\midrule")
        if row.fmt == "header":
            span = len(table.columns) + 1
            out.append(f"\\multicolumn{{{span}}}{{l}}{{{row.label}}} \\\\")
            continue
        cells = [_fmt_tex(c, _cell_fmt(row, j)) for j, c in enumerate(row.cells)]
        out.append(" & ".join([row.label, *cells]) + " \\\\")
    out += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{table.caption}}}",
        f"\\label{{{table.label}}}",
        "\\end{table}",
    ]
    return "\n".join(out) + "\n"


def to_csv(table: Table, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([table.row_label_header, *table.columns])
        for row in table.rows:
            if row.fmt == "header":
                w.writerow([row.label, *([""] * len(table.columns))])
                continue
            w.writerow([row.label, *[_fmt_csv(c, _cell_fmt(row, j)) for j, c in enumerate(row.cells)]])


def write_table(table: Table, out_dir: str | Path, name: str) -> list[Path]:
    """Write ``out_dir/name.tex`` and ``out_dir/name.csv``; returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{name}.tex"
    csv_path = out_dir / f"{name}.csv"
    tex_path.write_text(to_booktabs(table))
    to_csv(table, csv_path)
    return [tex_path, csv_path]


# ---------------------------------------------------------------------------
# Per-policy aggregation
# ---------------------------------------------------------------------------


def _policy_frames(runs: list[RunInfo]) -> dict[str, pd.DataFrame]:
    """Map policy key -> its eval records DataFrame for ONE group of runs.

    Learned policies are keyed by ``run.algo`` (and records use that label);
    AS/TWAP/initial come from the DQN run if present, else the first run that
    has them.
    """
    identities: set[tuple[str, str | None, str, int]] = set()
    contexts: set[tuple[str, str | None, int]] = set()
    for run in runs:
        identity = (run.setting, run.symbol, run.algo, run.seed)
        if identity in identities:
            raise ValueError(
                "duplicate run identity supplied to table aggregation: "
                f"setting={run.setting!r}, symbol={run.symbol!r}, "
                f"algo={run.algo!r}, seed={run.seed}"
            )
        identities.add(identity)
        contexts.add((run.setting, run.symbol, run.seed))
    if len(contexts) > 1:
        raise ValueError(
            "single-seed policy aggregation requires one setting/symbol/seed "
            f"context; got {sorted(contexts, key=repr)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    dqn_run = next((r for r in runs if r.algo == "dqn" and r.records is not None), None)
    shared_src = dqn_run or next((r for r in runs if r.records is not None), None)
    if shared_src is not None:
        for pol in ("initial", "as", "twap"):
            sub = shared_src.records[shared_src.records["policy"] == pol]
            if not sub.empty:
                frames[pol] = sub.reset_index(drop=True)
    for r in runs:
        if r.records is None:
            continue
        learned = r.records[r.records["policy"] == r.algo]
        if not learned.empty:
            frames[r.algo] = learned.reset_index(drop=True)
    return frames


def _primary_col(df: pd.DataFrame) -> str:
    """Require the revised primary outcome in one artifact."""
    if PRIMARY_COL not in df.columns:
        raise KeyError(f"revised artifact is missing required column {PRIMARY_COL!r}")
    return PRIMARY_COL


def _common_primary_col(frames) -> str:
    """Require the revised primary outcome in every supplied artifact."""
    frames = [df for df in frames if df is not None]
    if not frames:
        return PRIMARY_COL
    if not all(PRIMARY_COL in df.columns for df in frames):
        raise KeyError(f"all revised artifacts must contain {PRIMARY_COL!r}")
    return PRIMARY_COL


def _require_common_metric(frames, metric: str) -> str:
    frames = [df for df in frames if df is not None]
    if not frames or not all(metric in df.columns for df in frames):
        raise KeyError(f"all revised artifacts must contain {metric!r}")
    return metric


def _uses_absolute_differences(metric: str) -> bool:
    return metric in (PRIMARY_COL, NORMALIZED_PRIMARY_COL)


def _difference_units(metric: str) -> str:
    return "Basis Points" if metric == NORMALIZED_PRIMARY_COL else "Currency Units"


def _returns(df: pd.DataFrame, col: str | None = None) -> np.ndarray:
    return df.sort_values("episode")[col or _primary_col(df)].to_numpy(dtype=float)


def _episode_index(df: pd.DataFrame) -> tuple[int, ...]:
    """Return and validate the evaluation episode index of one policy frame."""
    if "episode" not in df.columns:
        raise KeyError("evaluation artifact is missing required column 'episode'")
    if df["episode"].isna().any():
        raise ValueError("evaluation artifact contains a missing episode index")
    episodes = tuple(sorted(int(v) for v in df["episode"].tolist()))
    if len(episodes) != len(set(episodes)):
        raise ValueError("evaluation artifact contains duplicate policy/episode rows")
    if not episodes:
        raise ValueError("evaluation artifact contains no episodes")
    return episodes


def _common_evaluation_episode_count(frames) -> int:
    """Require a complete common episode set and return its actual size.

    Publication tables make paired comparisons.  Allowing a merge to silently
    discard unmatched episodes would both mislabel the sample size and break
    the common-random-numbers contract, so mismatched episode sets are errors.
    """
    frames = [df for df in frames if df is not None]
    if not frames:
        raise ValueError("no evaluation policy records were supplied")
    expected = _episode_index(frames[0])
    for frame in frames[1:]:
        observed = _episode_index(frame)
        if observed != expected:
            raise ValueError(
                "evaluation policy episode sets differ; refusing to report a "
                "partially aligned comparison"
            )
    return len(expected)


def _selected_frames_across_seeds(runs: list[RunInfo]) -> list[pd.DataFrame]:
    """Frames selected by aggregate builders, grouped by full run context."""
    by_context: dict[tuple[str, str | None, int], list[RunInfo]] = {}
    for run in runs:
        if run.records is not None:
            by_context.setdefault(
                (run.setting, run.symbol, run.seed), []
            ).append(run)
    return [
        frame
        for context in sorted(by_context, key=repr)
        for frame in _policy_frames(by_context[context]).values()
    ]


def _aligned(
    a: pd.DataFrame, b: pd.DataFrame, col: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """CRN-aligned (paired) value arrays on the shared episode index."""
    if _episode_index(a) != _episode_index(b):
        raise ValueError("paired policy episode sets differ")
    merged = a.merge(b, on="episode", suffixes=("_a", "_b"))
    if "env_seed_a" in merged and not (merged["env_seed_a"] == merged["env_seed_b"]).all():
        raise ValueError("CRN violation: env seeds differ between policies at matched episodes")
    metric = col or _common_primary_col((a, b))
    return merged[f"{metric}_a"].to_numpy(float), merged[f"{metric}_b"].to_numpy(float)


def _seed_policy_means_indexed(
    runs: list[RunInfo], col: str | None = None
) -> dict[str, dict[int, float]]:
    """Return ``policy -> seed -> mean outcome`` for a homogeneous group.

    Keeping the seed identity is essential for paired policy-minus-benchmark
    confidence intervals. Duplicate representations of a policy/seed pair are
    accepted only when their means agree numerically.
    """
    metric = col or _common_primary_col(r.records for r in runs if r.records is not None)
    by_seed: dict[int, list[RunInfo]] = {}
    for r in runs:
        if r.seed is None or r.records is None:
            continue
        by_seed.setdefault(r.seed, []).append(r)
    out: dict[str, dict[int, float]] = {}
    for seed in sorted(by_seed):
        frames = _policy_frames(by_seed[seed])
        for pol, df in frames.items():
            value = float(np.mean(_returns(df, metric)))
            existing = out.setdefault(pol, {}).get(seed)
            if existing is not None and not np.isclose(existing, value):
                raise ValueError(
                    f"conflicting means for policy={pol!r}, seed={seed}: "
                    f"{existing} vs {value}"
                )
            out[pol][seed] = value
    return out


def _seed_policy_means(runs: list[RunInfo], col: str | None = None) -> dict[str, list[float]]:
    """Policy -> seed-ordered per-seed means (one number per seed)."""
    indexed = _seed_policy_means_indexed(runs, col)
    return {
        pol: [by_seed[seed] for seed in sorted(by_seed)]
        for pol, by_seed in indexed.items()
    }


# ---------------------------------------------------------------------------
# (a) eval-summary  (replaces tab:eval_summary_final)
# ---------------------------------------------------------------------------


def build_eval_summary(runs: list[RunInfo], *, rng: int = 0) -> Table:
    """Replacement for ``tab:eval_summary_final`` (one setting group).

    Rows: mean/SE/std/median risk-adjusted PnL, its economic decomposition,
    final inventory, shaped-reward diagnostics, paired currency differences,
    and paired Wilcoxon/t-test p-values under CRN.
    """
    frames = _policy_frames(runs)
    n_eval = _common_evaluation_episode_count(frames.values())
    cols = [p for p in POLICY_ORDER if p in frames]
    learned = [p for p in cols if p not in ("initial", "as", "twap")]
    headers = [POLICY_LABELS.get(p, p) for p in cols]

    def stat_row(label: str, fn: Callable[[pd.DataFrame], float], fmt: str) -> Row:
        return Row(label, [fn(frames[p]) for p in cols], fmt)

    metric = _common_primary_col(frames.values())
    metric_label = OUTCOME_LABELS[metric]
    rows: list[Row] = [
        stat_row(f"Mean {metric_label}", lambda d: float(np.mean(_returns(d, metric))), "money"),
        stat_row(f"SE {metric_label}", lambda d: stats.std_error(_returns(d, metric)), "money"),
        stat_row(
            f"Std {metric_label}", lambda d: float(np.std(_returns(d, metric), ddof=1)), "money"
        ),
        stat_row(f"Median {metric_label}", lambda d: float(np.median(_returns(d, metric))), "money"),
        stat_row("Mean Final Inventory", lambda d: float(np.mean(_returns(d, "I_final"))), "num2"),
        stat_row("Mean CLOB Reward", lambda d: float(np.mean(_returns(d, "clob_reward_sum"))), "money"),
        stat_row(
            "Mean Auction Reward",
            lambda d: float(np.mean(_returns(d, "auction_step_reward_sum"))),
            "money",
        ),
    ]
    decomposition = (
        ("Mean PnL", PNL_COL),
        ("Mean Cancellation Cost", "cancel_cost"),
        ("Mean Terminal Inventory Penalty", "inventory_penalty"),
    )
    has_decomposition = all(
        all(col in d for col in (PNL_COL, "cancel_cost", "inventory_penalty"))
        for d in frames.values()
    )
    if has_decomposition:
        for offset, (label, col) in enumerate(decomposition, start=4):
            rows.insert(
                offset,
                stat_row(label, lambda d, c=col: float(np.mean(_returns(d, c))), "money"),
            )
    if all("return_undisc" in d for d in frames.values()):
        rows.insert(
            7 if has_decomposition else 4,
            stat_row(
                "Mean Shaped Return (Diagnostic)",
                lambda d: float(np.mean(_returns(d, "return_undisc"))),
                "money",
            ),
        )

    means = {p: float(np.mean(_returns(frames[p], metric))) for p in cols}

    def comparison_row(label: str, base: str, skip: set[str]) -> Row:
        cells = [
            None
            if (p in skip or base not in means)
            else (
                means[p] - means[base]
            )
            for p in cols
        ]
        return Row(label, cells, "money")

    section_breaks: set[str] = set()
    comparison_header = f"Mean {metric_label} Differences ({_difference_units(metric)})"
    rows.append(Row(comparison_header, [None] * len(cols), "header"))
    section_breaks.add(len(rows) - 1)
    if "initial" in frames:
        rows.append(comparison_row("vs Initial", "initial", {"initial"}))
    if "as" in frames:
        rows.append(comparison_row("vs AS", "as", {"initial", "as"}))
    if "twap" in frames:
        rows.append(comparison_row("vs TWAP", "twap", {"initial", "as", "twap"}))

    def pvalue_row(label: str, bench: str, test: Callable) -> Row:
        cells: list[Any] = []
        for p in cols:
            if p in learned and bench in frames:
                a, b = _aligned(frames[p], frames[bench], metric)
                cells.append(test(a, b)[1])
            else:
                cells.append(None)
        return Row(label, cells, "pval")

    rows.append(Row("Paired tests (p-value)", [None] * len(cols), "header"))
    section_breaks.add(len(rows) - 1)
    if "as" in frames:
        rows.append(pvalue_row("Wilcoxon vs AS", "as", stats.wilcoxon))
        rows.append(pvalue_row("Paired-t vs AS", "as", stats.paired_t))
    if "twap" in frames:
        rows.append(pvalue_row("Wilcoxon vs TWAP", "twap", stats.wilcoxon))
        rows.append(pvalue_row("Paired-t vs TWAP", "twap", stats.paired_t))

    return Table(
        columns=headers,
        rows=rows,
        caption=(
            f"Evaluation results ({n_eval} episodes; {OUTCOME_CAPTIONS[metric]}; "
            "common random numbers across policies)."
        ),
        label="tab:eval_summary_final",
        row_label_header="Metric",
        section_breaks={i for i, _ in enumerate(rows) if i in section_breaks},
    )


# ---------------------------------------------------------------------------
# (b) historical per-ticker  (replaces tab:dqn_results_full)
# ---------------------------------------------------------------------------


def _group_by_symbol(runs: list[RunInfo]) -> dict[str, list[RunInfo]]:
    groups: dict[str, list[RunInfo]] = {}
    for r in runs:
        groups.setdefault(r.symbol or "?", []).append(r)
    return groups


def build_historical_results(
    runs: list[RunInfo], *, rng: int = 0, metric: str = PRIMARY_COL
) -> tuple[Table, Table]:
    """Replacement for ``tab:dqn_results_full``: per-ticker mean outcomes and a
    companion comparison table (each algo vs AS and TWAP, with bootstrap CIs).

    Returns ``(returns_table, improvements_table)``.
    """
    groups = _group_by_symbol(runs)
    symbols = sorted(groups)
    metric = _require_common_metric(
        (r.records for r in runs if r.records is not None), metric
    )
    if metric not in OUTCOME_LABELS:
        raise ValueError(f"unsupported historical output metric {metric!r}")
    normalized = metric == NORMALIZED_PRIMARY_COL
    value_fmt = "num2" if normalized else "money"
    ci_fmt = "num2_ci" if normalized else "money_ci"
    # Algorithm columns present anywhere, in canonical order.
    algos_present: list[str] = []
    for sym in symbols:
        for r in groups[sym]:
            if r.algo in ("dqn", "ddpg", "td3", "sac") and r.algo not in algos_present:
                algos_present.append(r.algo)
    algos_present = [a for a in ["dqn", "ddpg", "td3", "sac"] if a in algos_present]

    ret_cols = ["$\\hat\\sigma$", POLICY_LABELS["initial"], "AS", "TWAP"] + [POLICY_LABELS[a] for a in algos_present]
    ret_rows: list[Row] = []
    # accumulate per-symbol means for the Mean row
    acc: dict[str, list[float]] = {c: [] for c in ret_cols}
    imp_rows: list[Row] = []
    evaluation_frames: list[pd.DataFrame] = []
    imp_cols: list[str] = []
    for a in algos_present:
        imp_cols += [f"{POLICY_LABELS[a]} vs AS", f"{POLICY_LABELS[a]} vs TWAP"]
    imp_acc: dict[str, list[float]] = {c: [] for c in imp_cols}

    for sym in symbols:
        frames = _policy_frames(groups[sym])
        evaluation_frames.extend(frames.values())
        meta = next((r.metadata for r in groups[sym] if r.metadata), {})
        sigma = meta.get("as_calibration", {}).get("sigma", float("nan"))

        def m(pol: str) -> float:
            return float(np.mean(_returns(frames[pol], metric))) if pol in frames else float("nan")

        cells: list[Any] = [sigma, m("initial"), m("as"), m("twap")]
        for a in algos_present:
            cells.append(m(a))
        # sigma in scientific notation, returns in money.
        row_fmt = ["sci"] + [value_fmt] * (len(ret_cols) - 1)
        ret_rows.append(Row(sym, cells, row_fmt))
        for c, v in zip(ret_cols, cells):
            acc[c].append(v if not _is_blank(v) else np.nan)

        imp_cells: list[Any] = []
        for a in algos_present:
            for bench in ("as", "twap"):
                if a in frames and bench in frames:
                    av, bv = _aligned(frames[a], frames[bench], metric)
                    imp_cells.append(stats.bootstrap_ci(av - bv, rng=rng))
                else:
                    imp_cells.append(None)
        imp_rows.append(
            Row(sym, imp_cells, ci_fmt)
        )
        for c, v in zip(imp_cols, imp_cells):
            imp_acc[c].append(v[0] if (v is not None) else np.nan)

    # Mean rows (average across tickers). The Mean sigma cell is left blank
    # (matches the paper, which has no sigma for the Mean row).
    mean_cells = [float(np.nanmean(acc[c])) if len(acc[c]) else float("nan") for c in ret_cols]
    mean_cells[0] = None
    ret_rows.append(Row("Mean", mean_cells, ["sci"] + [value_fmt] * (len(ret_cols) - 1)))
    imp_mean = [float(np.nanmean(imp_acc[c])) if len(imp_acc[c]) else float("nan") for c in imp_cols]
    imp_rows.append(Row("Mean", imp_mean, value_fmt))

    n_eval = _common_evaluation_episode_count(evaluation_frames)
    returns_table = Table(
        columns=ret_cols,
        rows=ret_rows,
        caption=(
            "Per-ticker mean outcomes on the historical S\\&P 500 setting "
            f"({n_eval} episodes; {OUTCOME_CAPTIONS[metric]}; $\\hat\\sigma$ = estimated "
            "continuous-session volatility)."
        ),
        label="tab:dqn_results_full_bps" if normalized else "tab:dqn_results_full",
        row_label_header="Symbol",
        section_breaks={len(ret_rows) - 1},
    )
    improvements_table = Table(
        columns=imp_cols,
        rows=imp_rows,
        caption=(
            f"Per-ticker paired {OUTCOME_LABELS[metric].lower()} differences vs AS and TWAP "
            f"({_difference_units(metric).lower()}, with bootstrap 95\\% CIs over eval episodes)."
        ),
        label=(
            "tab:dqn_results_improvements_bps"
            if normalized
            else "tab:dqn_results_improvements"
        ),
        row_label_header="Symbol",
        section_breaks={len(imp_rows) - 1},
    )
    return returns_table, improvements_table


# ---------------------------------------------------------------------------
# (b2) cross-seed aggregates (IQM + bootstrap CI over seeds; rliable-style)
# ---------------------------------------------------------------------------


def build_eval_summary_multiseed(runs: list[RunInfo], *, rng: int = 0) -> Table:
    """Cross-seed aggregate for the synthetic setting. Each seed contributes one
    number per policy (its evaluation-sample mean outcome); these are aggregated across
    seeds with the IQM (interquartile mean) and a percentile-bootstrap 95\\% CI
    (Agarwal et al. 2021). Few seeds => wide CIs (the honest multi-seed signal)."""
    metric = _common_primary_col(r.records for r in runs if r.records is not None)
    n_eval = _common_evaluation_episode_count(_selected_frames_across_seeds(runs))
    indexed = _seed_policy_means_indexed(runs, metric)
    sm = {
        pol: [by_seed[seed] for seed in sorted(by_seed)]
        for pol, by_seed in indexed.items()
    }
    n_seeds = max((len(v) for v in sm.values()), default=0)
    cols = [p for p in POLICY_ORDER if p in sm]
    headers = [POLICY_LABELS.get(p, p) for p in cols]
    iqm_ci = {p: stats.iqm_ci(np.array(sm[p]), rng=rng) for p in cols}
    outcome_label = OUTCOME_LABELS[metric]
    outcome_caption = OUTCOME_CAPTIONS[metric]

    rows: list[Row] = [
        Row(f"IQM {outcome_label} [95\\% CI]", [iqm_ci[p] for p in cols], "money_ci"),
        Row("Mean of seed-means", [float(np.mean(sm[p])) for p in cols], "money"),
        Row("Seeds (n)", [len(sm[p]) for p in cols], "int"),
    ]
    section_breaks = {len(rows)}
    rows.append(
        Row(
            f"Paired Seed-level {outcome_label} Difference vs Benchmark "
            f"({_difference_units(metric)}; IQM [95\\% CI])",
            [None] * len(cols),
            "header",
        )
    )

    def imp(label: str, base: str, skip: set[str]) -> Row:
        def paired_ci(policy: str):
            shared = sorted(set(indexed[policy]) & set(indexed[base]))
            differences = np.asarray(
                [indexed[policy][seed] - indexed[base][seed] for seed in shared],
                dtype=float,
            )
            return stats.iqm_ci(differences, rng=rng)

        return Row(
            label,
            [
                None
                if (p in skip or base not in indexed)
                else paired_ci(p)
                for p in cols
            ],
            "money_ci",
        )

    if "as" in indexed:
        rows.append(imp("vs AS", "as", {"initial", "as"}))
    if "twap" in indexed:
        rows.append(imp("vs TWAP", "twap", {"initial", "as", "twap"}))

    return Table(
        columns=headers,
        rows=rows,
        caption=f"Cross-seed aggregate (synthetic; {n_seeds} seeds; {n_eval} "
        f"evaluation episodes per policy and seed). IQM of the per-seed mean "
        f"{outcome_caption} with percentile-bootstrap 95\\% CIs over seeds; "
        "policy-minus-benchmark intervals use paired seed-level differences. "
        "Reported policy = best mature validation checkpoint above the initial economic safety floor.",
        label="tab:eval_summary_multiseed",
        row_label_header="Metric",
        section_breaks=section_breaks,
    )


def build_historical_results_multiseed(
    runs: list[RunInfo], *, rng: int = 0, metric: str = PRIMARY_COL
) -> Table:
    """Cross-seed aggregate for the historical setting: per-ticker IQM of the
    per-seed mean outcomes, with a final row pooling all ticker$\\times$seed runs
    into an IQM with a bootstrap 95\\% CI."""
    groups = _group_by_symbol(runs)
    n_eval = _common_evaluation_episode_count(_selected_frames_across_seeds(runs))
    metric = _require_common_metric(
        (r.records for r in runs if r.records is not None), metric
    )
    if metric not in OUTCOME_LABELS:
        raise ValueError(f"unsupported historical output metric {metric!r}")
    normalized = metric == NORMALIZED_PRIMARY_COL
    value_fmt = "num2" if normalized else "money"
    ci_fmt = "num2_ci" if normalized else "money_ci"
    outcome_caption = OUTCOME_CAPTIONS[metric]
    symbols = sorted(groups)
    algos = [a for a in ["dqn", "ddpg", "td3", "sac"]
             if any(r.algo == a for sym in symbols for r in groups[sym])]
    col_keys = ["as", "twap"] + algos
    headers = ["AS", "TWAP"] + [POLICY_LABELS[a] for a in algos]

    pooled: dict[str, list[float]] = {k: [] for k in col_keys}
    pooled_differences: dict[str, dict[str, list[float]]] = {
        benchmark: {algo: [] for algo in algos}
        for benchmark in ("as", "twap")
    }
    outcome_rows: list[Row] = []
    paired_rows: list[Row] = []
    for sym in symbols:
        indexed = _seed_policy_means_indexed(groups[sym], metric)
        sm = {
            policy: [by_seed[seed] for seed in sorted(by_seed)]
            for policy, by_seed in indexed.items()
        }
        cells: list[Any] = []
        for k in col_keys:
            vals = sm.get(k, [])
            pooled[k] += list(vals)
            cells.append(stats.iqm(np.array(vals)) if vals else float("nan"))
        outcome_rows.append(Row(sym, cells, value_fmt))

        for benchmark in ("as", "twap"):
            comparison_cells: list[Any] = [None, None]
            for algo in algos:
                shared = sorted(
                    set(indexed.get(algo, {})) & set(indexed.get(benchmark, {}))
                )
                differences = [
                    indexed[algo][seed] - indexed[benchmark][seed]
                    for seed in shared
                ]
                pooled_differences[benchmark][algo].extend(differences)
                comparison_cells.append(
                    stats.iqm_ci(np.asarray(differences), rng=rng)
                    if differences
                    else None
                )
            paired_rows.append(
                Row(f"{sym} vs {POLICY_LABELS[benchmark]}", comparison_cells, ci_fmt)
            )

    all_outcomes = Row(
        "All tickers (IQM [95\\% CI])",
        [stats.iqm_ci(np.array(pooled[k]), rng=rng) if pooled[k] else None for k in col_keys],
        ci_fmt,
    )
    rows: list[Row] = [*outcome_rows]
    paired_start = len(rows)
    rows.append(
        Row(
            f"Paired Seed-level Differences ({_difference_units(metric)}; IQM [95\\% CI])",
            [None] * len(col_keys),
            "header",
        )
    )
    rows.extend(paired_rows)
    all_start = len(rows)
    rows.append(all_outcomes)
    for benchmark in ("as", "twap"):
        rows.append(
            Row(
                f"All tickers vs {POLICY_LABELS[benchmark]}",
                [None, None]
                + [
                    stats.iqm_ci(
                        np.asarray(pooled_differences[benchmark][algo]), rng=rng
                    )
                    if pooled_differences[benchmark][algo]
                    else None
                    for algo in algos
                ],
                ci_fmt,
            )
        )
    return Table(
        columns=headers,
        rows=rows,
        caption=f"Cross-seed aggregate (historical S\\&P 500; {n_eval} evaluation "
        "episodes per policy, ticker, and seed). Per-ticker IQM of "
        f"the per-seed mean {outcome_caption}; aggregate rows pool all ticker$\\times$seed "
        "runs into an IQM with a bootstrap 95\\% CI. Policy-minus-benchmark "
        "intervals use paired seed-level differences, per ticker and pooled. "
        "Best mature validation checkpoint above the initial economic safety floor.",
        label=(
            "tab:dqn_results_multiseed_bps"
            if normalized
            else "tab:dqn_results_multiseed"
        ),
        row_label_header="Symbol",
        section_breaks={paired_start, all_start},
    )


# ---------------------------------------------------------------------------
# (c) model-parameter + per-algorithm hyperparameter tables
# ---------------------------------------------------------------------------


def _get(cfg: ExperimentConfig, spec: Any) -> Any:
    """Resolve a dotted attribute path (str) or a callable(cfg)."""
    if callable(spec):
        return spec(cfg)
    obj: Any = cfg
    for part in spec.split("."):
        obj = getattr(obj, part)
    return obj


# (symbol, dotted-path-or-callable, comment) — matches tab:params_generative.
PARAM_SYMBOLS: list[tuple[str, Any, str]] = [
    ("$[t]$", "grid.time_unit", "Physical unit of the simulator clock"),
    ("$\\tau^{\\mathrm{op}}$", "grid.tau_op", "Auction opening time from session start"),
    ("$\\tau^{\\mathrm{cl}}$", "grid.tau_cl", "Clearing time from session start"),
    ("$I_0$", "grid.I0", "Initial inventory"),
    ("$\\lambda_0$", "clob_flow.lambda0", "Per-side Poisson intensity per clock unit"),
    ("$v_m$", "clob_flow.v_m", "Pareto distribution scale parameter"),
    ("$\\gamma_m$", "clob_flow.gamma_m", "Pareto distribution shape parameter"),
    ("$V_\\infty$", "clob_flow.V_inf", "Beta distribution scaling parameter"),
    ("$\\beta_a$", "clob_flow.beta_a", "First Beta distribution shape parameter"),
    ("$\\beta_b$", "clob_flow.beta_b", "Second Beta distribution shape parameter"),
    ("$\\rho$", "clob_flow.depth_decay", "Limit order book volume decay parameter"),
    ("$V$", "clob_flow.V", "Exogenous market/taker-order volume cap"),
    ("$V_{\\max}$", "actions.V_max", "Strategic CLOB submitted-volume cap"),
    ("$L_{\\mathrm{book}}$", "clob_flow.L_max", "Maximum exogenous CLOB depth"),
    ("$L_{\\mathrm{agent}}$", "actions.L_max", "Maximum strategic CLOB quote offset"),
    (
        "$B_\\infty$",
        "actions.B_inf",
        "Common absolute strategic/exogenous auction price-deviation bound",
    ),
    (
        "$B_{\\mathrm{max}}$",
        "actions.B_max",
        "Indicative-centred local policy-coordinate half-width",
    ),
    ("$D_\\mu$", "auction_flow.D_mu", "Minimum active exogenous auction slope"),
    ("$U_1$", "auction_flow.K_min", "Exogenous supply slope lower bound"),
    ("$U_2$", "auction_flow.K_max", "Exogenous supply slope upper bound"),
    ("$M_1$", "auction_flow.M1", "Exogenous supply spread lower bound"),
    ("$M_2$", "auction_flow.M2", "Exogenous supply spread upper bound"),
    ("$p_1$", "auction_flow.p1", "New-MM probability (one arrival each active minute)"),
    ("$p_2$", "auction_flow.p2", "MM cancellation probability (zero: schedules persist)"),
    ("$p_3$", "auction_flow.p3", "New market taker arrival probability"),
    ("$p_4$", "auction_flow.p4", "Market taker cancellation probability (ruling D7)"),
    ("$\\lambda$", "reward.lambda_inv", "Inventory penalty"),
    ("$q$", "reward.q", "Wrong-side dealing penalty"),
    ("$k^\\star$", "reward.k_star", "Tolerance"),
    ("$d$", "reward.d", "Cancellation cost per unit"),
    ("$H_0$", "algo1.H0", "Initial projected clearing signal"),
    ("$\\eta_H$", "algo1.eta_H", "Projected clearing-signal smoothing coefficient"),
    ("$\\alpha$", "grid.alpha", "Tick size"),
    ("$\\beta$", lambda c: c.actions.auction_K_grid_max / c.actions.auction_K_grid_n, "Tick size of grid on $K^a$"),
    ("$\\mathcal{K}$", "actions.auction_K_grid_n", "Upper bound on $K^a/\\beta$"),
    ("Slope indices", "actions.auction_K_multipliers", "Strategic auction slope grid"),
]

ROUGH_HESTON_SYMBOLS: list[tuple[str, Any, str]] = [
    ("$H$", "midprice.rough_heston.H", "Hurst exponent"),
    ("$\\rho$", "midprice.rough_heston.rho", "Price-volatility correlation"),
    ("$V_0$", "midprice.rough_heston.v0", "Initial variance"),
    ("$\\theta$", "midprice.rough_heston.theta", "Variance-drift level (long-run variance is theta/varsigma)"),
    ("$\\varsigma$", "midprice.rough_heston.kappa", "Variance mean-reversion rate"),
    ("$\\nu$", "midprice.rough_heston.xi", "Volatility of volatility"),
    ("$s^\\star$", "midprice.rough_heston.s_star", "Trading clock units per year"),
]

# Paper symbols for the RL hyperparameters that have one; others render blank.
HP_SYMBOLS: dict[str, str] = {
    "lr": "$\\eta$",
    "actor_lr": "$\\eta_\\mu$",
    "critic_lr": "$\\eta_Q$",
    "batch_size": "$B$",
    "buffer_size": "$\\bar N$",
    "min_buffer": "$\\underline N$",
    "target_soft_tau": "$\\tau_{\\mathrm{soft}}$",
    "epsilon_start": "$\\varepsilon_0$",
    "epsilon_end": "$\\varepsilon_\\infty$",
}


def _val_str(v: Any) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _param_table(cfg, spec_rows, caption, label) -> Table:
    rows = [
        Row(sym, [_val_str(_get(cfg, getter)), comment], "raw") for sym, getter, comment in spec_rows
    ]
    return Table(
        columns=["Value", "Comment"],
        rows=rows,
        caption=caption,
        label=label,
        row_label_header="Symbol",
    )


def build_param_tables(cfg: ExperimentConfig) -> dict[str, Table]:
    """Model-parameter tables from the resolved config: the generative-market
    table (replaces ``tab:params_generative``) and the mid-price model table."""
    tables = {
        "params_generative": _param_table(
            cfg,
            PARAM_SYMBOLS,
            "Generative stochastic market model parameters (from the resolved run config).",
            "tab:params_generative",
        )
    }
    if cfg.midprice.model == "rough_heston" and cfg.midprice.rough_heston is not None:
        tables["params_midprice"] = _param_table(
            cfg,
            ROUGH_HESTON_SYMBOLS,
            "Rough Heston mid-price parameters (Richard et al. scheme; ruling D14).",
            "tab:params_midprice",
        )
    elif cfg.midprice.model == "historical" and cfg.midprice.historical is not None:
        hist_rows = [
            ("CSV", "midprice.historical.csv_path", "Frozen mid-price input (ruling D13)"),
            ("Symbols", "midprice.historical.symbols", "S\\&P 500 tickers"),
            (
                "Model rebase",
                "midprice.historical.normalize_first",
                "In-environment session-start level; source artifact remains raw",
            ),
            (
                "Rows",
                "midprice.historical.n_rows",
                "Rows consumed through auction open ($= \\tau^{\\mathrm{op}}+1$)",
            ),
        ]
        tables["params_midprice"] = _param_table(
            cfg,
            hist_rows,
            "Historical mid-price configuration (ruling D13).",
            "tab:params_midprice",
        )
    return tables


def build_hyperparam_table(cfg: ExperimentConfig) -> Optional[Table]:
    """Per-algorithm RL hyperparameter table from the resolved config."""
    if cfg.algo is None:
        return None
    rows: list[Row] = [Row("$\\chi$ (discount)", [HP_SYMBOLS.get("chi", "$\\chi$"), _val_str(cfg.rl.chi)], "raw")]
    for key in (
        "normalizer_fit_episodes",
        "validation_size",
        "validation_frequency_episodes",
        "validation_patience_evals",
        "checkpoint_min_clob_updates",
        "checkpoint_min_auction_updates",
        "checkpoint_require_initial_improvement",
        "test_size",
    ):
        rows.append(
            Row(
                f"rl.{key}".replace("_", "\\_"),
                ["", _val_str(getattr(cfg.rl, key))],
                "raw",
            )
        )
    for key, value in cfg.algo.hyperparams.items():
        rows.append(Row(key.replace("_", "\\_"), [HP_SYMBOLS.get(key, ""), _val_str(value)], "raw"))
    return Table(
        columns=["Symbol", "Value"],
        rows=rows,
        caption=f"{cfg.algo.name.upper()} hyperparameters (from the resolved run config).",
        label=f"tab:hyperparams_{cfg.algo.name}",
        row_label_header="Hyperparameter",
    )
