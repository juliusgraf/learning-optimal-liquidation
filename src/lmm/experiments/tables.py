"""Table builders + emitters for Phase 7 (booktabs ``.tex`` and ``.csv``).

Builders return a phase-agnostic :class:`Table` (raw cell values + a per-row
format name); the emitters render LaTeX (booktabs, paper-faithful) and a clean
CSV (plain numbers, no thousands separators). All inputs are saved run-dir
artifacts (``eval/records.csv``, ``eval/metadata.yaml``, ``config_resolved.yaml``)
— no env stepping.

Aggregation across runs (figure f / multi-algo tables): the learned policy is
labelled ``"dqn"`` in every run's records, so its ALGORITHM identity comes from
``config_resolved.yaml:algo.name`` (via :func:`plotting.collect_runs`); the
canonical AS/TWAP/initial columns come from the DQN run of the group. Returns
are the UNDISCOUNTED episode sums (CLAUDE.md); stated in each caption.
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
    "build_param_tables",
    "build_hyperparam_table",
]

RETURN_COL = "return_undisc"


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

    Learned policies are keyed by ``run.algo`` (records label them ``"dqn"``);
    AS/TWAP/initial come from the DQN run if present, else the first run that
    has them.
    """
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
        learned = r.records[r.records["policy"] == "dqn"]
        if not learned.empty:
            frames[r.algo] = learned.reset_index(drop=True)
    return frames


def _returns(df: pd.DataFrame, col: str = RETURN_COL) -> np.ndarray:
    return df.sort_values("episode")[col].to_numpy(dtype=float)


def _aligned(a: pd.DataFrame, b: pd.DataFrame, col: str = RETURN_COL) -> tuple[np.ndarray, np.ndarray]:
    """CRN-aligned (paired) value arrays on the shared episode index."""
    merged = a.merge(b, on="episode", suffixes=("_a", "_b"))
    if "env_seed_a" in merged and not (merged["env_seed_a"] == merged["env_seed_b"]).all():
        raise ValueError("CRN violation: env seeds differ between policies at matched episodes")
    return merged[f"{col}_a"].to_numpy(float), merged[f"{col}_b"].to_numpy(float)


# ---------------------------------------------------------------------------
# (a) eval-summary  (replaces tab:eval_summary_final)
# ---------------------------------------------------------------------------


def build_eval_summary(runs: list[RunInfo], *, rng: int = 0) -> Table:
    """Replacement for ``tab:eval_summary_final`` (one setting group).

    Rows: mean/SE/std/median return, mean final inventory, mean CLOB & auction
    reward; relative improvements (vs Initial, AS, TWAP); paired Wilcoxon and
    t-test p-values (learned algos vs AS and vs TWAP) on CRN returns.
    """
    frames = _policy_frames(runs)
    cols = [p for p in POLICY_ORDER if p in frames]
    learned = [p for p in cols if p not in ("initial", "as", "twap")]
    headers = [POLICY_LABELS.get(p, p) for p in cols]

    def stat_row(label: str, fn: Callable[[pd.DataFrame], float], fmt: str) -> Row:
        return Row(label, [fn(frames[p]) for p in cols], fmt)

    rows: list[Row] = [
        stat_row("Mean Return", lambda d: float(np.mean(_returns(d))), "money"),
        stat_row("SE Return", lambda d: stats.std_error(_returns(d)), "money"),
        stat_row("Std Return", lambda d: float(np.std(_returns(d), ddof=1)), "money"),
        stat_row("Median Return", lambda d: float(np.median(_returns(d))), "money"),
        stat_row("Mean Final Inventory", lambda d: float(np.mean(_returns(d, "I_final"))), "num2"),
        stat_row("Mean CLOB Reward", lambda d: float(np.mean(_returns(d, "clob_reward_sum"))), "money"),
        stat_row(
            "Mean Auction Reward",
            lambda d: float(np.mean(_returns(d, "auction_step_reward_sum"))),
            "money",
        ),
    ]

    means = {p: float(np.mean(_returns(frames[p]))) for p in cols}

    def improvement_row(label: str, base: str, skip: set[str]) -> Row:
        cells = [
            None if (p in skip or base not in means) else stats.rel_improvement(means[p], means[base])
            for p in cols
        ]
        return Row(label, cells, "pct")

    section_breaks: set[str] = set()
    rows.append(Row("Relative Improvements (Mean Return)", [None] * len(cols), "header"))
    section_breaks.add(len(rows) - 1)
    if "initial" in frames:
        rows.append(improvement_row("vs Initial", "initial", {"initial"}))
    if "as" in frames:
        rows.append(improvement_row("vs AS", "as", {"initial", "as"}))
    if "twap" in frames:
        rows.append(improvement_row("vs TWAP", "twap", {"initial", "as", "twap"}))

    def pvalue_row(label: str, bench: str, test: Callable) -> Row:
        cells: list[Any] = []
        for p in cols:
            if p in learned and bench in frames:
                a, b = _aligned(frames[p], frames[bench])
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
        caption="Evaluation results (100 episodes; undiscounted returns, "
        "common random numbers across policies).",
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


def build_historical_results(runs: list[RunInfo], *, rng: int = 0) -> tuple[Table, Table]:
    """Replacement for ``tab:dqn_results_full``: per-ticker mean returns and a
    companion improvements table (each algo vs AS and TWAP, with bootstrap CIs).

    Returns ``(returns_table, improvements_table)``.
    """
    groups = _group_by_symbol(runs)
    symbols = sorted(groups)
    # Algorithm columns present anywhere, in canonical order.
    algos_present: list[str] = []
    for sym in symbols:
        for r in groups[sym]:
            if r.algo in ("dqn", "ddpg", "td3", "sac") and r.algo not in algos_present:
                algos_present.append(r.algo)
    algos_present = [a for a in ["dqn", "ddpg", "td3", "sac"] if a in algos_present]

    ret_cols = ["$\\hat\\sigma$", "Initial NFQ", "AS", "TWAP"] + [POLICY_LABELS[a] for a in algos_present]
    ret_rows: list[Row] = []
    # accumulate per-symbol means for the Mean row
    acc: dict[str, list[float]] = {c: [] for c in ret_cols}
    imp_rows: list[Row] = []
    imp_cols: list[str] = []
    for a in algos_present:
        imp_cols += [f"{POLICY_LABELS[a]} vs AS", f"{POLICY_LABELS[a]} vs TWAP"]
    imp_acc: dict[str, list[float]] = {c: [] for c in imp_cols}

    for sym in symbols:
        frames = _policy_frames(groups[sym])
        meta = next((r.metadata for r in groups[sym] if r.metadata), {})
        sigma = meta.get("as_calibration", {}).get("sigma", float("nan"))

        def m(pol: str) -> float:
            return float(np.mean(_returns(frames[pol]))) if pol in frames else float("nan")

        cells: list[Any] = [sigma, m("initial"), m("as"), m("twap")]
        for a in algos_present:
            cells.append(m(a))
        # sigma in scientific notation, returns in money.
        row_fmt = ["sci"] + ["money"] * (len(ret_cols) - 1)
        ret_rows.append(Row(sym, cells, row_fmt))
        for c, v in zip(ret_cols, cells):
            acc[c].append(v if not _is_blank(v) else np.nan)

        imp_cells: list[Any] = []
        for a in algos_present:
            for bench in ("as", "twap"):
                if a in frames and bench in frames:
                    av, bv = _aligned(frames[a], frames[bench])
                    imp_cells.append(stats.bootstrap_improvement_ci(av, bv, rng=rng))
                else:
                    imp_cells.append(None)
        imp_rows.append(Row(sym, imp_cells, "pct_ci"))
        for c, v in zip(imp_cols, imp_cells):
            imp_acc[c].append(v[0] if (v is not None) else np.nan)

    # Mean rows (average across tickers). The Mean sigma cell is left blank
    # (matches the paper, which has no sigma for the Mean row).
    mean_cells = [float(np.nanmean(acc[c])) if len(acc[c]) else float("nan") for c in ret_cols]
    mean_cells[0] = None
    ret_rows.append(Row("Mean", mean_cells, ["sci"] + ["money"] * (len(ret_cols) - 1)))
    imp_mean = [float(np.nanmean(imp_acc[c])) if len(imp_acc[c]) else float("nan") for c in imp_cols]
    imp_rows.append(Row("Mean", imp_mean, "pct"))

    returns_table = Table(
        columns=ret_cols,
        rows=ret_rows,
        caption="Per-ticker mean returns on the historical S\\&P 500 setting "
        "(100 episodes; undiscounted; $\\hat\\sigma$ = estimated continuous-session "
        "volatility).",
        label="tab:dqn_results_full",
        row_label_header="Symbol",
        section_breaks={len(ret_rows) - 1},
    )
    improvements_table = Table(
        columns=imp_cols,
        rows=imp_rows,
        caption="Per-ticker relative improvements vs AS and TWAP (\\%, with "
        "bootstrap 95\\% CIs over eval episodes).",
        label="tab:dqn_results_improvements",
        row_label_header="Symbol",
        section_breaks={len(imp_rows) - 1},
    )
    return returns_table, improvements_table


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
    ("$\\tau^{\\mathrm{op}}$", "grid.tau_op", "Auction opening time"),
    ("$\\tau^{\\mathrm{cl}}$", "grid.tau_cl", "Clearing time"),
    ("$I_0$", "grid.I0", "Initial inventory"),
    ("$\\lambda_0$", "clob_flow.lambda0", "Continuous phase Poisson intensity"),
    ("$v_m$", "clob_flow.v_m", "Pareto distribution scale parameter"),
    ("$\\gamma_m$", "clob_flow.gamma_m", "Pareto distribution shape parameter"),
    ("$V_\\infty$", "clob_flow.V_inf", "Beta distribution scaling parameter"),
    ("$\\beta_a$", "clob_flow.beta_a", "First Beta distribution shape parameter"),
    ("$\\beta_b$", "clob_flow.beta_b", "Second Beta distribution shape parameter"),
    ("$\\rho$", "clob_flow.depth_decay", "Limit order book volume decay parameter"),
    ("$V$", "clob_flow.V_max", "Maximum volume admitted by the market"),
    ("$U_1$", "auction_flow.K_min", "Exogenous supply slope lower bound"),
    ("$U_2$", "auction_flow.K_max", "Exogenous supply slope upper bound"),
    ("$M_1$", lambda c: -c.auction_flow.price_band_ticks, "Exogenous supply spread lower bound"),
    ("$M_2$", "auction_flow.price_band_ticks", "Exogenous supply spread upper bound"),
    ("$p_1$", "auction_flow.p1", "New market maker arrival probability"),
    ("$p_2$", "auction_flow.p2", "Market maker cancellation probability"),
    ("$p_3$", "auction_flow.p3", "New market taker arrival probability"),
    ("$p_4$", "auction_flow.p4", "Market taker cancellation probability (ruling D7)"),
    ("$\\lambda$", "reward.lambda_inv", "Inventory penalty"),
    ("$q$", "reward.q", "Wrong-side dealing penalty"),
    ("$k^\\star$", "reward.k_star", "Tolerance"),
    ("$d$", "reward.d", "Cancellation cost per unit"),
    ("$\\alpha$", "grid.alpha", "Tick size"),
    ("$\\beta$", lambda c: c.actions.auction_K_grid_max / c.actions.auction_K_grid_n, "Tick size of grid on $K^a$"),
    ("$\\mathcal{K}$", "actions.auction_K_grid_n", "Upper bound on $K^a/\\beta$"),
]

ROUGH_HESTON_SYMBOLS: list[tuple[str, Any, str]] = [
    ("$H$", "midprice.rough_heston.H", "Hurst exponent"),
    ("$\\rho$", "midprice.rough_heston.rho", "Price-volatility correlation"),
    ("$V_0$", "midprice.rough_heston.v0", "Initial variance"),
    ("$\\theta$", "midprice.rough_heston.theta", "Long-run variance"),
    ("$\\lambda$", "midprice.rough_heston.kappa", "Variance mean-reversion rate"),
    ("$\\nu$", "midprice.rough_heston.xi", "Volatility of volatility"),
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
            ("Norm.", "midprice.historical.normalize_first", "Session-start normalization"),
            ("Rows", "midprice.historical.n_rows", "Rows consumed ($= \\tau^{\\mathrm{op}}$)"),
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
    for key, value in cfg.algo.hyperparams.items():
        rows.append(Row(key.replace("_", "\\_"), [HP_SYMBOLS.get(key, ""), _val_str(value)], "raw"))
    return Table(
        columns=["Symbol", "Value"],
        rows=rows,
        caption=f"{cfg.algo.name.upper()} hyperparameters (from the resolved run config).",
        label=f"tab:hyperparams_{cfg.algo.name}",
        row_label_header="Hyperparameter",
    )
