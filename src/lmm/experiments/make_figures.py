"""Regenerate all figures from saved run outputs (Phase 7).

Figures are ALWAYS regenerated from results/revision_v2/<...>/metrics.csv and eval/ records
(incl. eval/traces/ for the anatomy figures) — never produced inside training
code (engineering conventions). No env stepping happens here.

``--run-dir`` accepts one or more run directories: the FIRST is the primary run
(figures a-e read from it); ALL of them feed the cross-algorithm comparison
(figure f). Each figure that lacks its inputs is skipped with a warning, so a
single DQN-only run still yields a-e. Every figure is written as both PDF (for
LaTeX) and PNG.

Figures:
  a training_diagnostics  (metrics.csv)            -> replaces dqn_training_loss / *_returns
  b policy_difference_curve (eval/policy_difference_*.csv)
  c episode_anatomy       (eval/traces/dqn_ep0)    -> replaces episode_*
  d benchmark_anatomy     (eval/traces/as,twap)    -> replaces benchmark_behavior_*
  j cancellation_strategy (eval/traces/dqn_ep0)    -> c_t (cancel-all) over the auction
  e eval_distributions    (eval/records.csv)       -> replaces final_evaluation_*
  f algorithm_comparison  (all run dirs' records)  -> new
  g convergence_curves    (all run dirs' metrics, multi-seed)  -> new
  h reward_decomposition  (run-level IQM/CI across all runs, per setting; multi-seed) -> new
  i policy_difference_multiseed (DQN economic differences, IQM/CI; multi-seed)
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from lmm.experiments import plotting as P
from lmm.experiments import stats

__all__ = ["build_parser", "main"]

# Human-readable setting names for figure titles (raw config names are ugly).
_PRETTY_SETTING = {"synthetic_rough_heston": "synthetic", "historical_sp500": "historical"}

_PRIMARY_COL = "risk_adjusted_pnl"
_OUTCOME_PRIORITY = (_PRIMARY_COL,)
_OUTCOME_LABELS = {
    _PRIMARY_COL: "Risk-adjusted PnL (currency units)",
}
_EVAL_OUTCOME_COLUMNS = (
    (_PRIMARY_COL, "eval_risk_adjusted_pnl_mean"),
)


def _primary_col(df: pd.DataFrame) -> str:
    """Require the revised primary outcome."""
    if _PRIMARY_COL not in df.columns:
        raise KeyError(f"revised artifact is missing required column {_PRIMARY_COL!r}")
    return _PRIMARY_COL


def _common_primary_col(frames) -> str:
    """Highest-priority outcome shared by every supplied artifact."""
    frames = [df for df in frames if df is not None]
    if not frames:
        return _PRIMARY_COL
    if not all(_PRIMARY_COL in df.columns for df in frames):
        raise KeyError(f"all revised artifacts must contain {_PRIMARY_COL!r}")
    return _PRIMARY_COL


def _outcome_label(metric: str) -> str:
    return _OUTCOME_LABELS[metric]


def _common_eval_col(frames) -> tuple[str, str]:
    """Return (records metric, validation column) shared by all metrics files."""
    frames = [df for df in frames if df is not None]
    for metric, col in _EVAL_OUTCOME_COLUMNS:
        if frames and all(col in df.columns for df in frames):
            return metric, col
    raise KeyError(
        "metrics artifacts share none of the supported validation columns "
        f"{tuple(col for _, col in _EVAL_OUTCOME_COLUMNS)}"
    )


# ---------------------------------------------------------------------------
# (a) training diagnostics
# ---------------------------------------------------------------------------


def fig_training_diagnostics(run_dir: Path, out: Path) -> None:
    df = P.read_metrics(run_dir)
    if df is None:
        return
    fig, axes = P.plt.subplots(1, 3, figsize=(12.0, 3.6))

    ax = axes[0]
    for col, name in (("loss_clob", "CLOB"), ("loss_auction", "Auction")):
        if col in df:
            sub = df[["episode", col]].dropna()
            if not sub.empty:
                ax.plot(sub["episode"], sub[col], label=name)
    ax.set(xlabel="Episode", ylabel="Huber loss", title="Training loss")
    ax.legend()

    ax = axes[1]
    metric = _primary_col(df)
    metric_label = _outcome_label(metric)
    ep, outcome = df["episode"], df[metric]
    ax.plot(ep, outcome, color="0.7", linewidth=0.8, label=metric_label)
    window = max(1, min(50, len(df) // 5))
    ax.plot(ep, outcome.rolling(window, min_periods=1).mean(),
            color=P.POLICY_COLORS["dqn"], label=f"Trailing mean ({window})")
    ax.set(xlabel="Episode", ylabel=metric_label, title="Training episode outcome")
    ax.legend()

    ax = axes[2]
    if "epsilon" in df:
        ax.plot(ep, df["epsilon"], color=P.POLICY_COLORS["as"])
    ax.set(xlabel="Episode", ylabel=r"$\varepsilon$", title="Exploration schedule")

    fig.tight_layout()
    P.save_fig(fig, out, "training_diagnostics")


# ---------------------------------------------------------------------------
# (b) paired fixed-policy economic differences
# ---------------------------------------------------------------------------


def fig_policy_difference_curve(run_dir: Path, out: Path) -> None:
    """Plot cumulative paired risk-adjusted-PnL differences."""
    fig, ax = P.plt.subplots()
    drawn = False
    for bench in ("as", "twap"):
        df = P.read_policy_difference(run_dir, bench)
        if df is None:
            continue
        x = df["episode"].to_numpy(float) + 1.0
        per_ep = df["policy_minus_benchmark"].to_numpy(float)
        cum = df["cumulative_policy_minus_benchmark"].to_numpy(float)
        lo, hi = stats.bootstrap_cumulative_band(per_ep)
        color = P.POLICY_COLORS[bench]
        ax.plot(x, cum, color=color, label=f"vs {P.POLICY_LABELS[bench]}")
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
        drawn = True
    if not drawn:
        P.plt.close(fig)
        warnings.warn(
            f"no policy_difference_*.csv under {run_dir}/eval; skipping difference figure",
            stacklevel=2,
        )
        return
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="Eval episode",
        ylabel="Cumulative policy minus benchmark $\\Pi_\\lambda$",
        title="Paired fixed-policy differences (bootstrap 95% CI)",
    )
    ax.legend()
    fig.tight_layout()
    P.save_fig(fig, out, "policy_difference_curve")


# ---------------------------------------------------------------------------
# (c) episode anatomy
# ---------------------------------------------------------------------------


def _phase_split(tr):
    return tr[tr["phase"] == "clob"], tr[tr["phase"] == "auction"]


def fig_episode_anatomy(run_dir: Path, out: Path, *, policy: str = "dqn", episode: int = 0) -> None:
    tr = P.read_trace(run_dir, policy, episode)
    if tr is None:
        return
    cfg = P.read_config(run_dir)
    tau_op, tau_cl = float(cfg.grid.tau_op), float(cfg.grid.tau_cl)
    clob, auc = _phase_split(tr)

    fig, axes = P.plt.subplots(3, 3, figsize=(13.0, 9.0))
    blue, orange, green = P.POLICY_COLORS["dqn"], P.POLICY_COLORS["as"], P.POLICY_COLORS["twap"]

    def mark(ax):
        P.mark_auction(ax, tau_op, tau_cl)

    ax = axes[0, 0]
    ax.plot(tr["t"], tr["s_mid"], color=blue); mark(ax)
    ax.set(title=r"Mid price $S_t^{\mathrm{mid}}$", xlabel="t")

    ax = axes[0, 1]
    ax.plot(tr["t"], tr["inventory"], color=blue); mark(ax)
    ax.set(title=r"Inventory $I_t$", xlabel="t")

    ax = axes[0, 2]
    if not clob.empty:
        ax.stem(clob["t"], clob["E_t"], basefmt=" ", linefmt=blue, markerfmt="o")
    mark(ax)
    ax.set(title=r"Executed volume $E_t$", xlabel="t")

    ax = axes[1, 0]
    ax.plot(tr["t"], tr["h_cl"], color=blue, label=r"$H_t^{\mathrm{cl}}$")
    term = tr[tr["is_terminal"] == 1]
    if not term.empty and term["S_cl"].notna().any():
        ax.scatter([tau_cl], [term["S_cl"].iloc[0]], color=orange, zorder=5, label=r"$S^{\mathrm{cl}}$")
    mark(ax)
    ax.set(title=r"Clearing price $H_t^{\mathrm{cl}}$", xlabel="t"); ax.legend()

    ax = axes[1, 1]
    if not clob.empty:
        ax.plot(clob["t"], clob["top_ask"], color=orange, label=r"$V_t^{+,1}$")
        ax.plot(clob["t"], clob["top_bid"], color=green, label=r"$V_t^{-,1}$")
    mark(ax)
    ax.set(title="Top-of-book volumes", xlabel="t"); ax.legend()

    ax = axes[1, 2]
    if not auc.empty:
        ax.step(auc["t"], auc["n_mm"], where="post", label=r"$M_t$", color=blue)
        ax.step(auc["t"], auc["n_buy_auc"], where="post", label=r"$N_t^{+}$", color=orange)
        ax.step(auc["t"], auc["n_sell_auc"], where="post", label=r"$N_t^{-}$", color=green)
    mark(ax)
    ax.set(title="Auction arrivals", xlabel="t"); ax.legend()

    ax = axes[2, 0]
    ax.plot(tr["t"], tr["reward"], color=blue); mark(ax)
    ax.set(title=r"One-step reward $R_t$", xlabel="t")

    ax = axes[2, 1]
    ax.plot(tr["t"], tr["cum_reward"], color=blue); mark(ax)
    ax.set(title="Cumulative reward", xlabel="t")

    ax = axes[2, 2]
    if not clob.empty:
        ax.bar(clob["t"], clob["act_volume"], width=0.6, color=blue, label=r"$v_t$ (CLOB)")
    if not auc.empty:
        ax.bar(auc["t"], auc["act_Ka"], width=0.6, color=orange, label=r"$K_t^a$ (auction)")
    twin = ax.twinx()
    if not clob.empty:
        twin.plot(clob["t"], clob["act_delta"], color=green, linewidth=1.0, label=r"$\delta_t$")
    if not auc.empty:
        twin.plot(auc["t"], auc["act_offset"], color="0.4", linewidth=1.0, label="offset")
    twin.set_ylabel("offset / $\\delta$")
    mark(ax)
    ax.set(title="Actions", xlabel="t"); ax.legend(loc="upper left")

    fig.suptitle(f"Episode anatomy ({P.POLICY_LABELS.get(policy, policy)}, episode {episode})")
    fig.tight_layout()
    P.save_fig(fig, out, "episode_anatomy")


# ---------------------------------------------------------------------------
# (d) benchmark anatomy
# ---------------------------------------------------------------------------


def fig_benchmark_anatomy(run_dir: Path, out: Path, *, episode: int = 0) -> None:
    as_tr = P.read_trace(run_dir, "as", episode)
    twap_tr = P.read_trace(run_dir, "twap", episode)
    if as_tr is None or twap_tr is None:
        return
    cfg = P.read_config(run_dir)
    tau_op, tau_cl = float(cfg.grid.tau_op), float(cfg.grid.tau_cl)

    fig, axes = P.plt.subplots(2, 2, figsize=(11.0, 7.0))
    series = (("as", as_tr), ("twap", twap_tr))

    def mark(ax):
        P.mark_auction(ax, tau_op, tau_cl)

    ax = axes[0, 0]
    for key, tr in series:
        ax.plot(tr["t"], tr["inventory"], color=P.POLICY_COLORS[key], label=P.POLICY_LABELS[key])
    mark(ax); ax.set(title=r"Inventory $I_t$", xlabel="t"); ax.legend()

    ax = axes[0, 1]
    for key, tr in series:
        clob = tr[tr["phase"] == "clob"]
        ax.plot(clob["t"], clob["E_t"], color=P.POLICY_COLORS[key], label=P.POLICY_LABELS[key])
    mark(ax); ax.set(title=r"Executed volume $E_t$ (CLOB)", xlabel="t"); ax.legend()

    ax = axes[1, 0]
    for key, tr in series:
        clob = tr[tr["phase"] == "clob"]
        ax.plot(clob["t"], clob["S_bullet"], color=P.POLICY_COLORS[key], label=f"{P.POLICY_LABELS[key]} $S^\\bullet$")
    mark(ax); ax.set(title="Submitted CLOB price", xlabel="t"); ax.legend()

    ax = axes[1, 1]
    for key, tr in series:
        ax.plot(tr["t"], tr["cum_reward"], color=P.POLICY_COLORS[key], label=P.POLICY_LABELS[key])
    mark(ax); ax.set(title="Cumulative reward", xlabel="t"); ax.legend()

    fig.suptitle(f"Benchmark anatomy (episode {episode})")
    fig.tight_layout()
    P.save_fig(fig, out, "benchmark_anatomy")


# ---------------------------------------------------------------------------
# (j) cancellation strategy c_t (same eval episode as the anatomy figures)
# ---------------------------------------------------------------------------


def fig_cancellation_strategy(
    run_dir: Path, out: Path, *, policy: str = "dqn", episode: int = 0
) -> None:
    """Cancel-all action c_t over the auction window for a single eval episode.

    Companion to ``episode_anatomy`` (called with the SAME ``policy``/``episode``,
    hence the same eval seed under CRN): the anatomy figure has no panel for the
    scalar cancel-all action A^5_t = c_t in {0, 1} (auction phase only,
    t in {n+1, ..., m}; CLAUDE.md ruling D4). This isolates the cancellation
    strategy, reading ``act_cancel`` (c_t) straight from the saved trace
    (``eval/traces/<policy>_ep<episode>.csv``); no env stepping happens here.
    """
    tr = P.read_trace(run_dir, policy, episode)
    if tr is None:
        return
    auc = tr[tr["phase"] == "auction"]
    if auc.empty:
        warnings.warn(
            f"no auction rows in {policy}_ep{episode}; skipping cancellation figure",
            stacklevel=2,
        )
        return

    t = auc["t"].to_numpy(float)
    c = auc["act_cancel"].fillna(0.0).to_numpy(float)
    blue = P.POLICY_COLORS[policy]

    fig, ax = P.plt.subplots(figsize=(9.0, 3.0))
    # A single, one-colour impulse plot: one marker per auction decision at its
    # value c_t in {0, 1}, with a stem to the c_t = 0 baseline. No second series,
    # no overlaid bars/crosses -- the y position alone reads off c_t.
    markerline, stemline, _ = ax.stem(t, c, basefmt=" ")
    P.plt.setp(stemline, color=blue, linewidth=1.2)
    P.plt.setp(markerline, color=blue, markersize=5)
    ax.axhline(0.0, color="0.85", linewidth=0.8, zorder=0)

    ax.set_ylim(-0.15, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    ax.set_ylabel(r"$c_t$")
    ax.set_xlabel("t")

    ax.set_title(
        rf"Cancellation strategy $c_t$ "
        f"({P.POLICY_LABELS.get(policy, policy)}, episode {episode})"
    )
    fig.tight_layout()
    P.save_fig(fig, out, "cancellation_strategy")


# ---------------------------------------------------------------------------
# (e) evaluation distributions
# ---------------------------------------------------------------------------


def fig_eval_distributions(run_dir: Path, out: Path, *, legacy_style: bool = False) -> None:
    df = P.read_records(run_dir)
    if df is None:
        return
    policies = [p for p in P.POLICY_ORDER if p in set(df["policy"])]
    by = {p: df[df["policy"] == p].sort_values("episode") for p in policies}
    labels = [P.POLICY_LABELS[p] for p in policies]
    colors = [P.POLICY_COLORS[p] for p in policies]
    metric = _primary_col(df)
    metric_label = _outcome_label(metric)

    if legacy_style:
        fig, ax = P.plt.subplots()
        means = [float(by[p][metric].mean()) for p in policies]
        errs = [float(by[p][metric].std(ddof=1)) for p in policies]
        ax.bar(labels, means, yerr=errs, color=colors, capsize=4)
        ax.set(ylabel=f"Mean {metric_label}", title="Evaluation outcome (mean ± std)")
        fig.tight_layout()
        P.save_fig(fig, out, "eval_distributions_bars")
        return

    fig, axes = P.plt.subplots(1, 3, figsize=(13.0, 4.0))

    ax = axes[0]
    data = [by[p][metric].to_numpy(float) for p in policies]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20)
    ax.set(ylabel=metric_label, title="Primary-outcome distribution")

    ax = axes[1]
    learned = [p for p in policies if p not in ("initial", "as", "twap")]
    diff_data, diff_labels = [], []
    for p in learned:
        for bench in ("as", "twap"):
            if bench in by:
                m = by[p].merge(by[bench], on="episode", suffixes=("", "_b"))
                diff_data.append((m[metric] - m[f"{metric}_b"]).to_numpy(float))
                diff_labels.append(f"{P.POLICY_LABELS[p]}\n- {P.POLICY_LABELS[bench]}")
    if diff_data:
        ax.boxplot(diff_data, tick_labels=diff_labels, showmeans=True)
        ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(ylabel=f"Paired {metric_label} difference (CRN)", title="Paired differences")

    ax = axes[2]
    x = np.arange(len(policies))
    clob_m = [float(by[p]["clob_reward_sum"].mean()) for p in policies]
    auc_m = [float(by[p]["auction_step_reward_sum"].mean()) for p in policies]
    ax.bar(x - 0.2, clob_m, width=0.4, label="CLOB", color=P.POLICY_COLORS["as"])
    ax.bar(x + 0.2, auc_m, width=0.4, label="Auction", color=P.POLICY_COLORS["twap"])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set(ylabel="Mean reward", title="CLOB / auction decomposition"); ax.legend()

    fig.tight_layout()
    P.save_fig(fig, out, "eval_distributions")


# ---------------------------------------------------------------------------
# (f) algorithm comparison (multi-run)
# ---------------------------------------------------------------------------


def fig_algorithm_comparison(run_dirs, out: Path) -> None:
    runs = P.collect_runs(run_dirs)
    runs = [r for r in runs if r.records is not None]
    if not runs:
        return
    settings = sorted({r.setting for r in runs})
    fig, axes = P.plt.subplots(1, len(settings), figsize=(5.5 * len(settings), 4.0), squeeze=False)
    for col, setting in enumerate(settings):
        ax = axes[0][col]
        sruns = [r for r in runs if r.setting == setting]
        metric = _common_primary_col(r.records for r in sruns)
        # mean over tickers per algo (synthetic has a single group).
        algo_vals: dict[str, list[float]] = {}
        for r in sruns:
            rows = r.records[r.records["policy"] == r.algo]
            learned = rows[metric].to_numpy(float)
            if learned.size:
                algo_vals.setdefault(r.algo, []).extend(learned.tolist())
        algos = [a for a in P.ALGO_ORDER if a in algo_vals]
        means, los, his, colors, labels = [], [], [], [], []
        for a in algos:
            pt, lo, hi = stats.bootstrap_ci(np.asarray(algo_vals[a]))
            means.append(pt); los.append(pt - lo); his.append(hi - pt)
            colors.append(P.POLICY_COLORS[a]); labels.append(P.POLICY_LABELS[a])
        ax.bar(labels, means, yerr=[los, his], color=colors, capsize=4)
        ax.set(ylabel=f"Mean {_outcome_label(metric)}", title=setting)
    fig.suptitle("Algorithm comparison (primary outcome, bootstrap 95% CI)")
    fig.tight_layout()
    P.save_fig(fig, out, "algorithm_comparison")


# ---------------------------------------------------------------------------
# (h) shaped-return decomposition: CLOB / auction-fictive / terminal-shaped
# ---------------------------------------------------------------------------

# Setting labels for the paper-grade title parenthetical.
_SETTING_TITLE = {
    "synthetic_rough_heston": "synthetic setting",
    "historical_sp500": "historical setting",
}

# The three additive reward components: (records.csv column, panel title).
# Their sum equals ``return_undisc`` (verified). CLAUDE.md reward forms:
# CLOB clamped f_c, per-step fictive auction f_a, and terminal clearing reward.
_REWARD_COMPONENTS = [
    ("clob_reward_sum", "Limit-order (CLOB) phase"),
    ("auction_step_reward_sum", "Auction phase: fictive shaping reward"),
    ("terminal_reward", "Terminal shaped reward (cash + utility penalties)"),
]


def fig_reward_decomposition(run_dirs, out: Path) -> None:
    """Cross-seed reward decomposition, one figure per setting (multiseed).

    Splits each method's undiscounted shaped return into its three additive parts —
    the CLOB (limit-order) reward, the per-step *fictive* auction shaping
    reward, and the terminal shaped reward.  The latter is not P&L: it mixes
    signed auction cash, wrong-side utility, and the inventory penalty.  The
    risk-adjusted PnL is reported separately as the primary outcome.

    The sample for each (method, component) is the set of RUN-level means — one
    number per run, its mean component reward over the 100 eval episodes —
    aggregated across ALL runs of the setting (seeds, and tickers in the
    historical setting: the same 5x5 configurations as the convergence and
    policy-difference figures). Bars are the IQM across runs with a bootstrap 95% CI across
    runs (the rliable convention, matching eval_summary/dqn_results_multiseed),
    so component bars roughly add up to the multiseed eval-table totals (exactly
    only up to IQM's non-additivity). Run-level aggregation also tames the
    heavy-tailed per-episode auction reward. A companion CSV records the plotted
    numbers. Emitted only with >= 2 runs. Reads eval/records.csv only.

    The learned policy is labelled with each run's ``algo.name``; benchmark rows
    (``as``/``twap``) are taken once per (ticker, seed) (identical across algos
    under CRN, so pooling per-algo would replicate them and shrink the CI).
    """
    runs = [r for r in P.collect_runs(run_dirs) if r.records is not None and r.seed is not None]
    if not runs:
        return
    settings = sorted({r.setting for r in runs})
    single_setting = len(settings) == 1
    for setting in settings:
        sruns = [r for r in runs if r.setting == setting]

        def _add_run(pooled, method, rows):
            d = pooled.setdefault(method, {c: [] for c, _ in _REWARD_COMPONENTS})
            for col, _ in _REWARD_COMPONENTS:
                d[col].append(float(rows[col].mean()))

        # method -> {component column -> list of per-RUN means}
        pooled: dict[str, dict[str, list[float]]] = {}
        bench_seen: set = set()
        for r in sruns:
            learned = r.records[r.records["policy"] == r.algo]
            if not learned.empty:
                _add_run(pooled, r.algo, learned)
            key = (r.symbol, r.seed)
            if key not in bench_seen:
                bench_seen.add(key)
                for b in ("as", "twap"):
                    rows = r.records[r.records["policy"] == b]
                    if not rows.empty:
                        _add_run(pooled, b, rows)
        methods = [a for a in P.ALGO_ORDER if a in pooled]
        methods += [b for b in ("as", "twap") if b in pooled]
        n_runs = max((len(pooled[m]["clob_reward_sum"]) for m in methods), default=0)
        if not methods or n_runs < 2:
            continue
        labels = [P.POLICY_LABELS[m] for m in methods]
        colors = [P.POLICY_COLORS[m] for m in methods]
        x = np.arange(len(methods))
        across = "seeds" if setting != "historical_sp500" else "runs"

        csv_rows: list[tuple] = []  # (component, method, iqm, lo, hi, n_runs)
        fig, axes = P.plt.subplots(1, 3, figsize=(13.5, 4.2), squeeze=False)
        for ax, (col, title) in zip(axes[0], _REWARD_COMPONENTS):
            pts, los, his = [], [], []
            for m in methods:
                vals = np.asarray(pooled[m][col], float)
                pt, lo, hi = stats.iqm_ci(vals, n_boot=2000)
                pts.append(pt); los.append(pt - lo); his.append(hi - pt)
                csv_rows.append((title, P.POLICY_LABELS[m], pt, lo, hi, int(vals.size)))
            ax.bar(x, pts, yerr=[los, his], color=colors, capsize=4)
            ax.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_title(title)
        axes[0][0].set_ylabel(f"Undiscounted reward\n(IQM, 95% CI across {across})")
        fig.suptitle(
            "Reward decomposition over evaluation phases "
            f"({_SETTING_TITLE.get(setting, setting)})"
        )
        fig.tight_layout()
        name = "reward_decomposition" if single_setting else f"reward_decomposition_{setting}"
        P.save_fig(fig, out, name)
        with (Path(out) / f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "method", "iqm", "ci_lo", "ci_hi", "n_runs"])
            for comp, method, pt, lo, hi, n in csv_rows:
                writer.writerow([comp, method, *(format(v, ".17g") for v in (pt, lo, hi)), n])


def fig_algorithm_comparison_multiseed(run_dirs, out: Path) -> None:
    """Cross-seed comparison (rliable-style): per algo, the IQM of the per-seed
    mean primary outcome with a bootstrap 95% CI over seeds; AS/TWAP IQM reference lines.
    One number per run (its 100-episode mean) is the seed-level sample. Emitted
    only when >= 2 seeds are present."""
    runs = [r for r in P.collect_runs(run_dirs) if r.records is not None and r.seed is not None]
    if len({r.seed for r in runs}) < 2:
        return
    settings = sorted({r.setting for r in runs})
    fig, axes = P.plt.subplots(1, len(settings), figsize=(5.5 * len(settings), 4.0), squeeze=False)
    for col, setting in enumerate(settings):
        ax = axes[0][col]
        sruns = [r for r in runs if r.setting == setting]
        metric = _common_primary_col(r.records for r in sruns)
        algo_vals: dict[str, list[float]] = {}
        bench_vals: dict[str, list[float]] = {"as": [], "twap": []}
        bench_seen: set = set()
        for r in sruns:
            learned_rows = r.records[r.records["policy"] == r.algo]
            learned = learned_rows[metric].to_numpy(float)
            if learned.size:
                algo_vals.setdefault(r.algo, []).append(float(np.mean(learned)))
            # Benchmark per-run means are identical across the algo runs of a
            # given (ticker, seed) under CRN; take each once so the reference
            # lines match the multiseed tables' deduped benchmark IQM.
            key = (r.symbol, r.seed)
            if key not in bench_seen:
                bench_seen.add(key)
                for b in ("as", "twap"):
                    brows = r.records[r.records["policy"] == b]
                    bv = brows[metric].to_numpy(float)
                    if bv.size:
                        bench_vals[b].append(float(np.mean(bv)))
        algos = [a for a in P.ALGO_ORDER if a in algo_vals]
        pts, los, his, colors, labels = [], [], [], [], []
        for a in algos:
            pt, lo, hi = stats.iqm_ci(np.asarray(algo_vals[a]))
            pts.append(pt); los.append(pt - lo); his.append(hi - pt)
            colors.append(P.POLICY_COLORS[a]); labels.append(P.POLICY_LABELS[a])
        ax.bar(labels, pts, yerr=[los, his], color=colors, capsize=4)
        for b, ls in (("as", "--"), ("twap", ":")):
            if bench_vals[b]:
                ax.axhline(stats.iqm(np.asarray(bench_vals[b])), ls=ls, color="0.3", lw=1.2,
                           label=P.POLICY_LABELS.get(b, b))
        # When several settings share the figure, name them on the panels;
        # otherwise the setting goes in the suptitle (avoids a redundant title).
        title = _PRETTY_SETTING.get(setting, setting) if len(settings) > 1 else ""
        ax.set(ylabel=_outcome_label(metric), title=title)
        ax.legend(fontsize=7)
    if len(settings) == 1:
        fig.suptitle(f"Final performance ({_PRETTY_SETTING.get(settings[0], settings[0])})")
    else:
        fig.suptitle("Final performance")
    fig.tight_layout()
    P.save_fig(fig, out, "algorithm_comparison_multiseed")


def fig_convergence_curves(run_dirs, out: Path) -> None:
    """Multi-seed training-convergence figure (three panels, one curve per
    learned algorithm, aggregated across all runs of a single setting — seeds,
    and tickers in the historical setting):

      (1) greedy validation outcome vs episode -> policy convergence/plateau. A
          star marks the across-run median ``best.pt`` (early-stopping) episode;
          faint AS/TWAP lines give the benchmark level.
      (2) auction critic loss vs episode (log-y) -> numerical stability
          (bounded, stationary; no divergence). The CLOB loss is uniformly
          small and is omitted for clarity.
      (3) auction mean |TD error| vs episode -> Bellman-residual stabilization.

    The central line is the interquartile mean (IQM) across runs at each
    episode; the eval panel shows a bootstrap 95% CI band (few eval points), the
    dense loss/TD panels a 25-75% interquartile band. Honest by construction:
    DQN's late-training degradation stays visible and the best.pt marker shows
    which checkpoint is reported. Emitted only with >= 2 runs of one setting.
    """
    import matplotlib.lines as mlines

    runs = [r for r in P.collect_runs(run_dirs) if r.seed is not None]
    if not runs:
        return
    by_setting: dict[str, list] = {}
    for r in runs:
        by_setting.setdefault(r.setting, []).append(r)
    setting = max(by_setting, key=lambda s: len(by_setting[s]))
    runs = by_setting[setting]
    if len(runs) < 2:
        return

    metrics_by_algo: dict[str, list] = {}
    for r in runs:
        df = P.read_metrics(r.run_dir)
        supported = {col for _, col in _EVAL_OUTCOME_COLUMNS}
        if df is None or not (supported & set(df.columns)):
            continue
        metrics_by_algo.setdefault(r.algo, []).append(df)
    algos = [a for a in P.ALGO_ORDER if a in metrics_by_algo]
    if not algos:
        return
    metric, eval_col = _common_eval_col(
        df for algo_dfs in metrics_by_algo.values() for df in algo_dfs
    )

    def _aligned(dfs, col, *, window=1):
        """Per-episode matrix [n_episodes x n_runs] aligned on the episode
        index (outer join), optionally rolling-mean-smoothed per run."""
        cols = []
        for k, df in enumerate(dfs):
            if col not in df.columns or "episode" not in df.columns:
                continue
            sub = df[["episode", col]].dropna()
            if sub.empty:
                continue
            s = pd.Series(sub[col].to_numpy(float),
                          index=sub["episode"].to_numpy(int), name=k)
            if window > 1:
                s = s.rolling(window, min_periods=1).mean()
            cols.append(s)
        if not cols:
            return np.empty(0, int), np.empty((0, 0))
        mat = pd.concat(cols, axis=1).sort_index()
        return mat.index.to_numpy(int), mat.to_numpy(float)

    fig, axes = P.plt.subplots(1, 3, figsize=(13.0, 3.8))

    # -- panel 1: validation outcome (policy convergence) -------------------
    ax = axes[0]
    for a in algos:
        dfs = metrics_by_algo[a]
        eps, mat = _aligned(dfs, eval_col)
        if eps.size == 0:
            continue
        pts, los, his = [], [], []
        for row in mat:
            p, lo, hi = stats.iqm_ci(row[~np.isnan(row)])
            pts.append(p); los.append(lo); his.append(hi)
        pts, los, his = map(np.asarray, (pts, los, his))
        c = P.POLICY_COLORS[a]
        ax.plot(eps, pts, color=c, marker="o", ms=3, label=P.POLICY_LABELS[a])
        ax.fill_between(eps, los, his, color=c, alpha=0.15, linewidth=0)
        # best.pt = across-run median argmax episode, snapped to the eval grid
        best = [int(df.loc[df[eval_col].idxmax(), "episode"])
                for df in dfs if df[eval_col].notna().any()]
        if best:
            j = int(np.argmin(np.abs(eps - int(np.median(best)))))
            ax.scatter([eps[j]], [pts[j]], marker="*", s=160, color=c,
                       edgecolor="k", linewidth=0.5, zorder=6)
    bench: dict[str, list] = {"as": [], "twap": []}
    for r in runs:
        if r.records is None:
            continue
        for b in bench:
            rows = r.records[r.records["policy"] == b]
            if metric not in rows:
                continue
            bv = rows[metric].to_numpy(float)
            if bv.size:
                bench[b].append(float(np.mean(bv)))
    for b, ls in (("as", "--"), ("twap", ":")):
        if bench[b]:
            ax.axhline(stats.iqm(np.asarray(bench[b])), ls=ls, color="0.4",
                       lw=1.0, label=P.POLICY_LABELS[b])
    ax.set(xlabel="Episode", ylabel=f"Validation {_outcome_label(metric)}",
           title="Policy convergence")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], marker="*", linestyle="none",
                                 markerfacecolor="0.3", markeredgecolor="k",
                                 markersize=10, label="best.pt"))
    ax.legend(handles=handles, fontsize=6, ncol=2)

    # -- panels 2-3: stability diagnostics (dense, smoothed) ----------------
    for ax, col, ylab, title, logy in (
        (axes[1], "loss_auction", "Critic loss", "Critic-loss stability (auction phase)", True),
        (axes[2], "td_abs_mean_auction", r"Mean $|$TD error$|$", "Bellman residual (auction phase)", False),
    ):
        for a in algos:
            eps, mat = _aligned(metrics_by_algo[a], col, window=25)
            if eps.size == 0:
                continue
            line = np.full(len(eps), np.nan)
            lo = np.full(len(eps), np.nan)
            hi = np.full(len(eps), np.nan)
            for i, row in enumerate(mat):
                v = row[~np.isnan(row)]
                if v.size:
                    line[i] = stats.iqm(v)
                    lo[i] = np.percentile(v, 25)
                    hi[i] = np.percentile(v, 75)
            c = P.POLICY_COLORS[a]
            ax.plot(eps, line, color=c, lw=1.2, label=P.POLICY_LABELS[a])
            ax.fill_between(eps, lo, hi, color=c, alpha=0.12, linewidth=0)
        if logy:
            ax.set_yscale("log")
        ax.set(xlabel="Episode", ylabel=ylab, title=title)

    fig.suptitle(f"Training convergence ({_PRETTY_SETTING.get(setting, setting)})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    P.save_fig(fig, out, "convergence_curves")


# ---------------------------------------------------------------------------
# (i) cross-seed paired economic differences (DQN; IQM + bootstrap 95% CI)
# ---------------------------------------------------------------------------


def fig_policy_difference_multiseed(run_dirs, out: Path, *, symbol: str | None = None) -> None:
    """Cross-config cumulative risk-adjusted-PnL difference for DQN.

    Each run accumulates ``DQN - benchmark`` under CRN. Curves
    are aligned by eval-episode index; the central line is the IQM across runs at
    each episode and the band is a percentile bootstrap 95% CI across runs (same
    construction as the convergence figure's eval panel). Two curves per figure:
    vs AS and vs TWAP; positive means DQN outperforms the benchmark.

    Aggregates over ALL runs of a setting — seeds, and tickers in the historical
    setting (the same 5x5 configurations as the convergence figure). Pass
    ``symbol`` to restrict the historical figure to a single ticker. Emitted only
    with >= 2 runs. Reads eval/records.csv only.
    """
    runs = [r for r in P.collect_runs(run_dirs)
            if r.records is not None and r.algo == "dqn" and r.seed is not None]
    if not runs:
        return
    for setting in sorted({r.setting for r in runs}):
        is_hist = setting == "historical_sp500"
        sruns = [r for r in runs if r.setting == setting]
        if is_hist and symbol:
            sruns = [r for r in sruns if r.symbol == symbol]
        by_cfg = {}
        for r in sruns:
            by_cfg.setdefault((r.symbol, r.seed), r)   # one run per (ticker, seed)
        cfgs = sorted(by_cfg, key=lambda k: (str(k[0]), k[1]))
        if len(cfgs) < 2:
            continue
        n_seeds = len({s for _, s in cfgs})
        n_tickers = len({t for t, _ in cfgs})
        fig, ax = P.plt.subplots()
        drawn = False
        for bench in ("as", "twap"):
            curves = []
            for k in cfgs:
                df = by_cfg[k].records
                p = df[df["policy"] == "dqn"][["episode", "env_seed", _PRIMARY_COL]]
                b = df[df["policy"] == bench][["episode", "env_seed", _PRIMARY_COL]]
                m = p.merge(b, on="episode", suffixes=("_p", "_b")).sort_values("episode")
                if m.empty or (m["env_seed_p"].to_numpy() != m["env_seed_b"].to_numpy()).any():
                    continue  # CRN guard: paired outcome gaps need the shared env seed
                differences = (
                    m[f"{_PRIMARY_COL}_p"].to_numpy(float)
                    - m[f"{_PRIMARY_COL}_b"].to_numpy(float)
                )
                curves.append(np.cumsum(differences))
            if len(curves) < 2:
                continue
            length = min(c.size for c in curves)
            mat = np.vstack([c[:length] for c in curves])      # [n_runs x length]
            x = np.arange(1, length + 1)
            pts, los, his = [], [], []
            for i in range(length):
                pt, lo, hi = stats.iqm_ci(mat[:, i], n_boot=2000)
                pts.append(pt); los.append(lo); his.append(hi)
            color = P.POLICY_COLORS[bench]
            ax.plot(x, pts, color=color, label=f"vs {P.POLICY_LABELS[bench]}")
            ax.fill_between(x, los, his, color=color, alpha=0.2, linewidth=0)
            drawn = True
        if not drawn:
            P.plt.close(fig)
            continue
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        base = _SETTING_TITLE.get(setting, setting)
        if not is_hist:
            title = f"DQN paired differences across {n_seeds} seeds ({base})"
        elif symbol:
            title = f"DQN paired differences across {n_seeds} seeds ({base}, {symbol})"
        else:
            title = (f"DQN paired differences across {len(cfgs)} runs "
                     f"({base}, {n_tickers} tickers × {n_seeds} seeds)")
        ax.set(
            xlabel="Eval episode",
            ylabel="Cumulative DQN minus benchmark $\\Pi_\\lambda$\n(IQM, 95% CI)",
            title=title,
        )
        ax.legend()
        fig.tight_layout()
        name = (
            f"policy_difference_multiseed_{symbol}"
            if (is_hist and symbol)
            else "policy_difference_multiseed"
        )
        P.save_fig(fig, out, name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-make-figures",
        description="Regenerate figures from saved run outputs.",
    )
    parser.add_argument("--run-dir", required=True, nargs="+", help="run directory/ies")
    parser.add_argument("--out", default=None, help="output dir (default: <run>/figures)")
    parser.add_argument("--legacy-style", action="store_true",
                        help="bar-chart variant of the evaluation-distribution figure")
    parser.add_argument("--policy", default=None, help="policy for the episode-anatomy figure")
    parser.add_argument("--episode", type=int, default=0, help="traced episode index to plot")
    parser.add_argument(
        "--multiseed", action="store_true",
        help="build ONLY the cross-seed IQM/CI figures (run dirs spanning >= 2 seeds)",
    )
    parser.add_argument(
        "--difference-symbol", default=None,
        help="restrict the historical cross-config DQN difference figure to one ticker "
             "(default: all tickers × seeds, like the convergence figure)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    P.apply_style()
    run_dirs = [Path(d) for d in args.run_dir]
    primary = run_dirs[0]
    primary_cfg = P.read_config(primary)
    learned_policy = (
        args.policy
        if args.policy is not None
        else (primary_cfg.algo.name if primary_cfg.algo is not None else "dqn")
    )
    out = Path(args.out) if args.out else primary / "figures"
    out.mkdir(parents=True, exist_ok=True)

    if args.multiseed:
        funcs = [
            ("algorithm_comparison_multiseed",
             lambda: fig_algorithm_comparison_multiseed(run_dirs, out)),
            ("convergence_curves",
             lambda: fig_convergence_curves(run_dirs, out)),
            ("policy_difference_multiseed",
             lambda: fig_policy_difference_multiseed(
                 run_dirs, out, symbol=args.difference_symbol
             )),
            ("reward_decomposition",
             lambda: fig_reward_decomposition(run_dirs, out)),
        ]
    else:
        funcs = [
            ("training_diagnostics", lambda: fig_training_diagnostics(primary, out)),
            ("policy_difference_curve", lambda: fig_policy_difference_curve(primary, out)),
            ("episode_anatomy",
             lambda: fig_episode_anatomy(
                 primary, out, policy=learned_policy, episode=args.episode
             )),
            ("benchmark_anatomy", lambda: fig_benchmark_anatomy(primary, out, episode=args.episode)),
            ("cancellation_strategy",
             lambda: fig_cancellation_strategy(
                 primary, out, policy=learned_policy, episode=args.episode
             )),
            ("eval_distributions",
             lambda: fig_eval_distributions(primary, out, legacy_style=args.legacy_style)),
            ("algorithm_comparison", lambda: fig_algorithm_comparison(run_dirs, out)),
        ]
    for name, fn in funcs:
        try:
            fn()
        except Exception as exc:  # one bad figure must not abort the batch
            warnings.warn(f"figure {name!r} failed: {exc}", stacklevel=2)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
