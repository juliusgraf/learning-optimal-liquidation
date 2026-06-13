"""Regenerate all figures from saved run outputs (Phase 7).

Figures are ALWAYS regenerated from results/<...>/metrics.csv and eval/ records
(incl. eval/traces/ for the anatomy figures) — never produced inside training
code (engineering conventions). No env stepping happens here.

``--run-dir`` accepts one or more run directories: the FIRST is the primary run
(figures a-e read from it); ALL of them feed the cross-algorithm comparison
(figure f). Each figure that lacks its inputs is skipped with a warning, so a
single DQN-only run still yields a-e. Every figure is written as both PDF (for
LaTeX) and PNG.

Figures:
  a training_diagnostics  (metrics.csv)            -> replaces dqn_training_loss / *_returns
  b regret_curve          (eval/regret_*.csv)      -> replaces dqn_vs_glft_regret
  c episode_anatomy       (eval/traces/dqn_ep0)    -> replaces episode_*
  d benchmark_anatomy     (eval/traces/as,twap)    -> replaces benchmark_behavior_*
  e eval_distributions    (eval/records.csv)       -> replaces final_evaluation_*
  f algorithm_comparison  (all run dirs' records)  -> new
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from lmm.experiments import plotting as P
from lmm.experiments import stats

__all__ = ["build_parser", "main"]


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
    ep, ret = df["episode"], df["return_undisc"]
    ax.plot(ep, ret, color="0.7", linewidth=0.8, label="Return")
    window = max(1, min(50, len(df) // 5))
    ax.plot(ep, ret.rolling(window, min_periods=1).mean(),
            color=P.POLICY_COLORS["dqn"], label=f"Trailing mean ({window})")
    ax.set(xlabel="Episode", ylabel="Undiscounted return", title="Episode return")
    ax.legend()

    ax = axes[2]
    if "epsilon" in df:
        ax.plot(ep, df["epsilon"], color=P.POLICY_COLORS["as"])
    ax.set(xlabel="Episode", ylabel=r"$\varepsilon$", title="Exploration schedule")

    fig.tight_layout()
    P.save_fig(fig, out, "training_diagnostics")


# ---------------------------------------------------------------------------
# (b) regret curve
# ---------------------------------------------------------------------------


def fig_regret_curve(run_dir: Path, out: Path) -> None:
    fig, ax = P.plt.subplots()
    drawn = False
    for bench in ("as", "twap"):
        df = P.read_regret(run_dir, bench)
        if df is None:
            continue
        x = df["episode"].to_numpy(float) + 1.0
        per_ep = df["regret"].to_numpy(float)
        cum = df["cum_regret"].to_numpy(float)
        lo, hi = stats.bootstrap_cumulative_band(per_ep)
        color = P.POLICY_COLORS[bench]
        ax.plot(x, cum, color=color, label=f"vs {P.POLICY_LABELS[bench]}")
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
        drawn = True
    if not drawn:
        P.plt.close(fig)
        warnings.warn(f"no regret_*.csv under {run_dir}/eval; skipping regret figure", stacklevel=2)
        return
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(xlabel="Eval episode", ylabel=r"Cumulative regret $\mathrm{PRegret}(T)$",
           title="Cumulative regret vs benchmarks (bootstrap 95% CI)")
    ax.legend()
    fig.tight_layout()
    P.save_fig(fig, out, "regret_curve")


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

    if legacy_style:
        fig, ax = P.plt.subplots()
        means = [float(by[p]["return_undisc"].mean()) for p in policies]
        errs = [float(by[p]["return_undisc"].std(ddof=1)) for p in policies]
        ax.bar(labels, means, yerr=errs, color=colors, capsize=4)
        ax.set(ylabel="Mean undiscounted return", title="Evaluation returns (mean ± std)")
        fig.tight_layout()
        P.save_fig(fig, out, "eval_distributions_bars")
        return

    fig, axes = P.plt.subplots(1, 3, figsize=(13.0, 4.0))

    ax = axes[0]
    data = [by[p]["return_undisc"].to_numpy(float) for p in policies]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20)
    ax.set(ylabel="Undiscounted return", title="Return distribution")

    ax = axes[1]
    learned = [p for p in policies if p not in ("initial", "as", "twap")]
    diff_data, diff_labels = [], []
    for p in learned:
        for bench in ("as", "twap"):
            if bench in by:
                m = by[p].merge(by[bench], on="episode", suffixes=("", "_b"))
                diff_data.append((m["return_undisc"] - m["return_undisc_b"]).to_numpy(float))
                diff_labels.append(f"{P.POLICY_LABELS[p]}\n- {P.POLICY_LABELS[bench]}")
    if diff_data:
        ax.boxplot(diff_data, tick_labels=diff_labels, showmeans=True)
        ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(ylabel="Paired return difference (CRN)", title="Paired differences")

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
        # mean over tickers per algo (synthetic has a single group).
        algo_vals: dict[str, list[float]] = {}
        for r in runs:
            if r.setting != setting:
                continue
            learned = r.records[r.records["policy"] == "dqn"]["return_undisc"].to_numpy(float)
            if learned.size:
                algo_vals.setdefault(r.algo, []).extend(learned.tolist())
        algos = [a for a in P.ALGO_ORDER if a in algo_vals]
        means, los, his, colors, labels = [], [], [], [], []
        for a in algos:
            pt, lo, hi = stats.bootstrap_ci(np.asarray(algo_vals[a]))
            means.append(pt); los.append(pt - lo); his.append(hi - pt)
            colors.append(P.POLICY_COLORS[a]); labels.append(P.POLICY_LABELS[a])
        ax.bar(labels, means, yerr=[los, his], color=colors, capsize=4)
        ax.set(ylabel="Mean undiscounted return", title=setting)
    fig.suptitle("Algorithm comparison (mean return, bootstrap 95% CI)")
    fig.tight_layout()
    P.save_fig(fig, out, "algorithm_comparison")


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
    parser.add_argument("--policy", default="dqn", help="policy for the episode-anatomy figure")
    parser.add_argument("--episode", type=int, default=0, help="traced episode index to plot")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    P.apply_style()
    run_dirs = [Path(d) for d in args.run_dir]
    primary = run_dirs[0]
    out = Path(args.out) if args.out else primary / "figures"
    out.mkdir(parents=True, exist_ok=True)

    funcs = [
        ("training_diagnostics", lambda: fig_training_diagnostics(primary, out)),
        ("regret_curve", lambda: fig_regret_curve(primary, out)),
        ("episode_anatomy",
         lambda: fig_episode_anatomy(primary, out, policy=args.policy, episode=args.episode)),
        ("benchmark_anatomy", lambda: fig_benchmark_anatomy(primary, out, episode=args.episode)),
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
