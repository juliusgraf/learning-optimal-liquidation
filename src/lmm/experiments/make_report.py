"""Focused research report from saved artifacts only; no training or environment stepping.

The expected economic objective calls for a mean, not a trimmed mean that changes
that estimand. Episode-paired differences are averaged within each training seed;
intervals resample whole seed blocks. All seed estimates remain visible.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lmm.experiments import plotting as P, publication
from lmm.experiments.make_figures import _reportable_best_episode

SYNTHETIC = "synthetic_rough_heston"
ALGOS = P.ALGO_ORDER
POLICIES = [*ALGOS, "as", "twap"]
KEY = ["episode", "env_seed"]
BOOTSTRAP_REPLICATES = 10000
CAPTIONS = {
    "economic_performance": "Held-out economic performance. Points and 95% percentile bootstrap intervals summarize the mean of training-seed test means. Small points show every seed. AS and TWAP are stylized reference policies. All outcomes are basis points of initial notional, not dollar forecasts. Each market is reported separately; horizontal scales may differ.",
    "learning": "Economic validation during training on shaped J. Thin lines show each seed; stars mark the actual selected mature checkpoint. Heavy lines and pointwise 95% seed-bootstrap bands use only checkpoints observed for every requested seed. Individual traces continue after that common support ends. There is no smoothing, best-so-far envelope, or extrapolation after early stopping. Episode zero is the untrained diagnostic. Test outcomes are never used in these curves.",
    "auction_mechanism": "Closing-auction contribution with the CLOB trajectory held fixed. Total value equals signed execution price edge minus cancellation fees plus terminal inventory-risk relief, relative to submitting no auction orders. The cash component excludes risk relief. Inventory panels report mean absolute exposure as a percentage of initial inventory at auction open and after clearing. Historical panels average the fixed reported tickers equally within each seed before resampling seeds. This is a policy decomposition, not the effect of retraining without an auction.",
    "credit_assignment": "Does dense auction guidance accelerate learning? Curves show paired economic validation differences between auction-potential-on and auction-potential-off policies at the same training budget. Both optimize the same economic objective with common CLOB conditioning and observations. Thin traces retain every seed; mean bands use common observed support only. This isolates delayed auction credit assignment rather than extra manuscript preferences.",
    "treatments": "Paired synthetic treatment effects on held-out economic objective, first condition minus second. Differences are paired by environment seed before averaging within training seed; intervals resample those seed means. Forecast information, bounded action anchoring, combined manuscript preferences, invariant auction credit assignment and auction access are separated. Information and access contrasts hold the economic training objective fixed; preference contrasts retain common dense conditioning. Negative effects are retained. These are pointwise, exploratory intervals, without familywise significance claims.",
}
TABLE_CAPTIONS = {
    "economic_performance": "Held-out net PnL and economic objective in basis points of initial notional. Cells show equal-seed means with 95% seed-bootstrap intervals below. Objective differences against DQN, AS and TWAP are paired by episode and environment seed. PnL includes fees; objective additionally subtracts terminal inventory penalty. Neither includes training shaping.",
    "auction_mechanism": "Auction contribution by market and learned policy. Total value equals execution price edge minus fees (Cash) plus terminal inventory-risk relief, relative to no auction orders with the CLOB trajectory fixed. Open and final inventory are mean absolute exposure as a percentage of initial inventory. Cells show equal-seed means with 95% seed-bootstrap intervals below. Contributions are in basis points of initial notional.",
    "treatments": "Paired synthetic treatment effects on the held-out economic objective in basis points of initial notional. Positive values favor the first condition. Cells show equal-seed means with pointwise 95% seed-bootstrap intervals below. Forecast information is compared at a fixed anchor and economic objective. Anchoring holds information and shaped J fixed. Dense credit, combined preferences and auction access have matched controls. No familywise significance claim is made.",
}
METHODS = """The estimand is expected risk-adjusted PnL (economic bar-J-lambda after subtracting initial inventory value), in basis points of initial notional. Net PnL includes cancellation fees and excludes shaping; objective subtracts terminal inventory penalty. Shaped J is used only for training. For each market and policy, average test episodes within each training seed, then give each seed equal weight. Pair comparisons by episode AND environment seed before averaging. Intervals are deterministic 95% percentile bootstrap intervals of seed means (10,000 resamples). Evaluation paths are not independent training replications. Reference policies are checked across algorithm runs and included once per seed. Historical macro summaries keep the observed ticker set fixed and resample seed blocks jointly, preserving cross-ticker dependence. They do not estimate performance on unseen stocks or dates. Small seed counts limit precision; show every seed, avoid significance stars and universal rankings. One-seed development reports omit uncertainty intervals. Learning bands are pointwise and stop when common observed support ends. Algorithm comparisons include the declared preprocessing differences. All figures come from saved records; no policy is retrained, reselected or re-evaluated by this report.

Reporting choices follow Agarwal et al., NeurIPS 2021 (https://papers.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) on interval estimates and uncertainty, and Patterson et al., JMLR 2024 (https://www.jmlr.org/papers/v25/23-0183.html) on empirical design. A mean is retained here because it is the stated economic estimand; no poor seeds are trimmed out. The five historical tickers are not a broad RL benchmark suite requiring a performance-profile plot.
"""


def experiment_scope(runs):
    """Expose training and held-out scope from the resolved run specifications."""
    mechanisms = sorted({r.cfg.auction_flow.clearing_mechanism for r in runs})
    text = (f"Auction clearing mechanism: {', '.join(mechanisms)}. "
            "These results apply only to the recorded mechanism; legacy nearest-tick "
            "results are not evidence for revised volume-maximizing clearing. ")
    historical = [r for r in runs if r.setting == P.HISTORICAL_SETTING]
    if not historical:
        return text
    pooled = [r for r in historical if r.cfg.midprice.historical.training_pool == "all_symbols"]
    text += ("Historical training pools all configured stocks on training dates only. "
            "Validation and evaluation remain stock-specific on their separate date partitions. "
            if len(pooled) == len(historical) else
            "Historical training-pool choices are recorded in each resolved configuration. ")
    if any(r.cfg.rl.test_seed_namespace in (18001, 19001) for r in historical):
        text += ("The campaign uses a separate simulated evaluation namespace, but the "
                 "August 24–28, 2026 historical test dates were previously inspected in v17 and v18. "
                 "These dates are a reused holdout, not a new untouched historical sample. "
                 "Additional training seeds do not add independent historical dates.")
    return text


def mean_interval(values, *, n_boot=BOOTSTRAP_REPLICATES):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("estimates require nonempty finite seed values")
    point = float(values.mean())
    if len(values) == 1:
        return point, float("nan"), float("nan")
    idx = np.random.default_rng(0).integers(len(values), size=(n_boot, len(values)))
    lo, hi = np.quantile(values[idx].mean(axis=1), [.025, .975])
    return point, float(lo), float(hi)


def market(run):
    return run.symbol if run.setting == P.HISTORICAL_SETTING else "Synthetic"


def frame(run, policy):
    out = run.records.loc[run.records.policy == policy].sort_values(KEY).reset_index(drop=True)
    if out.empty or out.duplicated(KEY).any():
        raise ValueError(f"{run.run_dir}: missing or duplicate policy {policy}")
    return out


def require_paired(a, b):
    if not a[KEY].equals(b[KEY]):
        raise ValueError("CRN violation: episode/environment seed pairs differ")


def economic_rows(run, policy):
    """Recover the same-CLOB no-order counterfactual exactly from settlement."""
    f = frame(run, policy)
    columns = ["initial_mid", "initial_inventory", "pnl", "risk_adjusted_pnl",
               "clob_exec_qty", "auction_exec_qty", "I_final", "S_cl",
               "residual_liquidation_price", "cancel_cost", "inventory_penalty"]
    if not set(columns).issubset(f.columns) or not np.isfinite(f[columns].to_numpy(float)).all():
        raise ValueError(f"{run.run_dir}: missing/nonfinite economic accounting fields")
    notional = f.initial_mid * f.initial_inventory
    if (notional <= 0).any():
        raise ValueError("initial notional must be positive")
    scale = 10000 / notional
    opened = f.initial_inventory - f.clob_exec_qty
    closed = f.I_final
    if not np.allclose(opened - f.auction_exec_qty, closed, rtol=1e-9, atol=1e-8):
        raise ValueError("signed inventory conservation failed")
    penalty = run.cfg.reward.lambda_inv * closed ** 2
    if not np.allclose(f.inventory_penalty, penalty) or not np.allclose(f.risk_adjusted_pnl, f.pnl - penalty):
        raise ValueError("economic objective/terminal penalty accounting failed")
    edge = f.auction_exec_qty * (f.S_cl - f.residual_liquidation_price)
    relief = run.cfg.reward.lambda_inv * (opened ** 2 - closed ** 2)
    out = f[KEY].copy()
    out["objective_bps"] = f.risk_adjusted_pnl * scale
    out["pnl_bps"] = f.pnl * scale
    out["auction_edge_bps"] = edge * scale
    out["auction_fees_bps"] = f.cancel_cost * scale
    out["auction_cash_bps"] = (edge - f.cancel_cost) * scale
    out["auction_risk_relief_bps"] = relief * scale
    out["auction_value_bps"] = (edge - f.cancel_cost + relief) * scale
    out["open_abs_inventory_pct"] = 100 * abs(opened) / f.initial_inventory
    out["close_abs_inventory_pct"] = 100 * abs(closed) / f.initial_inventory
    out["short_probability_pct"] = 100 * (closed < 0).astype(float)
    # Verify precomputed normalized fields when available, never mix scales.
    for saved, derived in [("risk_adjusted_pnl_bps", "objective_bps"), ("pnl_bps", "pnl_bps")]:
        if saved in f and not np.allclose(f[saved], out[derived]):
            raise ValueError(f"{run.run_dir}: inconsistent {saved}")
    return out


def seed_outcomes(runs):
    rows, groups = [], {}
    for run in runs:
        key = (market(run), run.seed)
        groups.setdefault(key, {})[run.algo] = run
    for (name, seed), group in sorted(groups.items()):
        if set(group) != set(ALGOS):
            raise ValueError(f"incomplete algorithm block: {name}, seed {seed}")
        reference = group["dqn"]
        refs = {p: economic_rows(reference, p) for p in ("as", "twap")}
        dqn_reference = economic_rows(reference, "dqn")
        for algo in ALGOS:
            run = group[algo]
            learned = economic_rows(run, algo)
            for policy, expected in refs.items():
                actual = economic_rows(run, policy)
                require_paired(expected, actual)
                if not np.allclose(expected.drop(columns=KEY), actual.drop(columns=KEY), rtol=1e-10, atol=1e-9):
                    raise ValueError(f"reference policy differs between algorithms: {name}/{seed}/{policy}")
            for p in ([algo, "as", "twap"] if algo == "dqn" else [algo]):
                f = learned if p == algo else refs[p]
                row = {"market": name, "policy": p, "seed": seed, "n_episodes": len(f)}
                row.update(f.drop(columns=KEY).mean().to_dict())
                require_paired(f, dqn_reference)
                row["gap_dqn_bps"] = float((f.objective_bps - dqn_reference.objective_bps).mean())
                for bench, b in refs.items():
                    require_paired(f, b)
                    row[f"gap_{bench}_bps"] = float((f.objective_bps - b.objective_bps).mean())
                if p in ALGOS:
                    initial = economic_rows(run, "initial")
                    require_paired(f, initial)
                    row["gap_initial_bps"] = float((f.objective_bps - initial.objective_bps).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame, keys, metrics):
    rows = []
    for identity, group in frame.groupby(keys, sort=True):
        if len(keys) == 1:
            identity = (identity,) if not isinstance(identity, tuple) else identity
        for metric in metrics:
            point, lo, hi = mean_interval(group[metric])
            rows.append(dict(zip(keys, identity), metric=metric, mean=point, ci_low=lo, ci_high=hi, n_seeds=len(group)))
    return pd.DataFrame(rows)


def learning_rows(runs):
    rows = []
    for run in runs:
        metrics = P.read_metrics(run.run_dir)
        if metrics is None or "eval_risk_adjusted_pnl_mean" not in metrics:
            raise ValueError(f"{run.run_dir}: missing economic validation series")
        initial = yaml.safe_load((run.run_dir / "checkpoints/initial_validation.yaml").read_text())
        if initial.get("metric") != "risk_adjusted_pnl":
            raise ValueError("initial validation metric must be economic")
        best = _reportable_best_episode(run)
        if best is None:
            raise ValueError(f"{run.run_dir}: no selected checkpoint")
        factor = 10000 / (run.cfg.grid.S0 * run.cfg.grid.I0)
        vals = metrics.loc[metrics.eval_risk_adjusted_pnl_mean.notna(), ["episode", "eval_risk_adjusted_pnl_mean"]]
        if vals.empty or vals.episode.duplicated().any() or best not in set(vals.episode):
            raise ValueError("selected checkpoint must have an observed validation value")
        observations = [(-1, float(initial["value"])), *vals.itertuples(index=False, name=None)]
        for episode, value in observations:
            if not np.isfinite(value):
                raise ValueError("nonfinite validation value")
            rows.append({"market": market(run), "policy": run.algo, "seed": run.seed,
                         "episodes_completed": int(episode) + 1, "objective_bps": value * factor,
                         "selected": int(episode) == best})
    return pd.DataFrame(rows)


def learning_summary(data):
    """Only estimate means on common observed support; never forward-fill."""
    rows = []
    for (name, policy), group in data.groupby(["market", "policy"]):
        n = group.seed.nunique()
        for episode, block in group.groupby("episodes_completed"):
            if len(block) != n:
                continue
            mean, lo, hi = mean_interval(block.objective_bps)
            rows.append(dict(market=name, policy=policy, episodes_completed=episode,
                             mean=mean, ci_low=lo, ci_high=hi, n_seeds=n))
    return pd.DataFrame(rows)


def auction_groups(seed_data):
    parts = [seed_data.loc[seed_data.market == "Synthetic"].copy()]
    historical = seed_data.loc[seed_data.market != "Synthetic"]
    if not historical.empty:
        # Equal ticker weights inside each seed, not independent ticker resampling.
        h = historical.groupby(["policy", "seed"], as_index=False).mean(numeric_only=True)
        h["market"] = "Historical (fixed-ticker mean)"
        parts.append(h)
    return pd.concat(parts, ignore_index=True)


def _axes(names, *, height=3.2, max_cols=3):
    cols = min(max_cols, len(names))
    fig, axes = P.plt.subplots(math.ceil(len(names) / cols), cols,
                              figsize=(4.4 * cols, height * math.ceil(len(names) / cols)), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    return fig, list(axes.flat[:len(names)])


def _interval(ax, values, y, color, *, marker="o", offset=0, seeds=False):
    point, lo, hi = mean_interval(values)
    ax.plot(point, y + offset, marker=marker, color=color, ms=5, zorder=4)
    if np.isfinite(lo):
        ax.plot([lo, hi], [y + offset] * 2, color=color, lw=1.6, zorder=3)
    if seeds:
        ax.scatter(values, y + offset + np.linspace(-.10, .10, len(values)), s=10,
                   color=color, alpha=.4, zorder=2)


def _finish(fig, out, name, note):
    fig.text(.5, .01, note, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1))
    P.save_fig(fig, out, name)


def performance_figure(data, out):
    names = sorted(data.market.unique(), key=lambda x: (x != "Synthetic", x))
    fig, axes = _axes(names)
    for name, ax in zip(names, axes):
        for y, policy in enumerate(POLICIES):
            values = data.loc[(data.market == name) & (data.policy == policy), "objective_bps"]
            _interval(ax, values, y, P.POLICY_COLORS[policy], seeds=True)
        ax.axvline(0, color=".5", lw=.8)
        ax.set(yticks=range(6), yticklabels=[P.POLICY_LABELS[p] for p in POLICIES],
               xlabel="Test economic objective (bps)", title=name)
        ax.invert_yaxis()
    _finish(fig, out, "economic_performance", "Mean and 95% seed-bootstrap CI; small points = seed means. Higher is better.")


def learning_figure(data, aggregate, out):
    names = sorted(data.market.unique(), key=lambda x: (x != "Synthetic", x))
    fig, axes = _axes(names)
    for name, ax in zip(names, axes):
        for policy in ALGOS:
            group = data.loc[(data.market == name) & (data.policy == policy)]
            for _, seed in group.groupby("seed"):
                ax.plot(seed.episodes_completed, seed.objective_bps, color=P.POLICY_COLORS[policy], alpha=.22, lw=.8)
                selected = seed.loc[seed.selected]
                ax.scatter(selected.episodes_completed, selected.objective_bps, marker="*", s=45,
                           color=P.POLICY_COLORS[policy], zorder=4)
            curve = aggregate.loc[(aggregate.market == name) & (aggregate.policy == policy)]
            ax.plot(curve.episodes_completed, curve["mean"], color=P.POLICY_COLORS[policy], label=policy.upper())
            ax.fill_between(curve.episodes_completed, curve.ci_low, curve.ci_high, color=P.POLICY_COLORS[policy], alpha=.10)
        ax.axhline(0, color=".5", lw=.8)
        ax.set(title=name, xlabel="Training episodes completed", ylabel="Validation economic objective (bps)")
    axes[0].legend(ncol=2, fontsize=8)
    _finish(fig, out, "learning", "Thin lines = individual seeds; stars = selected checkpoints; aggregate bands stop at common observed support.")


def auction_figure(data, out):
    grouped = auction_groups(data)
    names = list(grouped.market.unique())
    fig, axes = P.plt.subplots(2, len(names), figsize=(6 * len(names), 6.8), squeeze=False)
    for col, name in enumerate(names):
        for row, specs in enumerate([
            [("auction_value_bps", "Total economic value", "o", -.12), ("auction_cash_bps", "Price edge minus fees", "s", .12)],
            [("open_abs_inventory_pct", "At auction open", "o", -.12), ("close_abs_inventory_pct", "After clearing", "s", .12)],
        ]):
            ax = axes[row, col]
            for y, policy in enumerate(ALGOS):
                sub = grouped.loc[(grouped.market == name) & (grouped.policy == policy)]
                for metric, label, marker, offset in specs:
                    _interval(ax, sub[metric], y, P.POLICY_COLORS[policy], marker=marker, offset=offset)
            for _, label, marker, _ in specs:
                ax.plot([], [], color=".35", marker=marker, ls="", label=label)
            ax.axvline(0, color=".5", lw=.8)
            ax.set(yticks=range(4), yticklabels=[p.upper() for p in ALGOS], title=name if row == 0 else "Residual exposure",
                   xlabel="Contribution vs no auction orders (bps)" if row == 0 else "Mean absolute inventory (% of initial)")
            ax.invert_yaxis()
            ax.legend(fontsize=8, loc="best")
    _finish(fig, out, "auction_mechanism", "Same CLOB trajectory; signed auction fills. Historical means keep the reported ticker set fixed. 95% seed-bootstrap CIs.")


def treatment_figure(data, out):
    labels = {
        "auction_credit": "Dense auction credit on − off\nEconomic objective held fixed",
        "h_feature": "H observation on − off\nFixed anchor, economic training",
        "h_anchor": "Indicative − frozen-mid anchor\nH observation and shaped J held fixed",
        "combined_preferences": "Combined manuscript preferences on − off",
        "auction_access": "Auction access on − off\nMatched information, economic training",
    }
    if set(data.contrast_key) != set(labels):
        raise ValueError("treatment figure must cover every recorded contrast exactly")
    fig, axes = _axes(list(labels), height=3.1, max_cols=2)
    for (key, label), ax in zip(labels.items(), axes):
        for y, algo in enumerate(ALGOS):
            sub = data.loc[(data.contrast_key == key) & (data.algorithm == algo)]
            _interval(ax, sub.mean_difference, y, P.POLICY_COLORS[algo], seeds=True)
        ax.axvline(0, color=".5", lw=.8)
        ax.set(yticks=range(4), yticklabels=[a.upper() for a in ALGOS], title=label,
               xlabel="Paired test objective difference (bps)")
        ax.invert_yaxis()
    _finish(fig, out, "treatments", "Mean and pointwise 95% seed-bootstrap CI; small points = paired seed means. Positive favors the first condition.")


def credit_learning_rows(runs):
    """Pair raw validation curves at common budgets; never select a time point."""
    arms = {}
    path_sets = {}
    for suffix in ("mechanism_economic_dense", "mechanism_economic_sparse"):
        selected = [r for r in runs if r.setting == SYNTHETIC + "__" + suffix]
        arms[suffix] = learning_rows(selected)
        for r in selected:
            meta = yaml.safe_load((r.run_dir/'checkpoints/initial_validation.yaml').read_text())
            paths = meta.get('validation_seeds')
            if not paths:
                raise ValueError('credit learning comparison requires recorded validation seeds')
            key = (r.algo, r.seed)
            if key in path_sets and path_sets[key] != paths:
                raise ValueError('credit learning validation CRN mismatch')
            path_sets[key] = paths
    key = ['market', 'policy', 'seed', 'episodes_completed']
    left, right = arms.values()
    d = left.merge(right, on=key, suffixes=('_dense', '_sparse'), validate='one_to_one')
    d['objective_bps'] = d.objective_bps_dense-d.objective_bps_sparse
    return d


def credit_learning_figure(data, out):
    # The paired difference directly answers whether guidance accelerates
    # economic learning. Untrained initialization stays visible at zero.
    fig, axes = _axes(ALGOS, height=3.1, max_cols=2)
    summary = learning_summary(data)
    for algo, ax in zip(ALGOS, axes):
        sub = data[data.policy == algo]
        for _, block in sub.groupby('seed'):
            block = block.sort_values('episodes_completed')
            ax.plot(block.episodes_completed, block.objective_bps, color=P.POLICY_COLORS[algo], alpha=.18, lw=.7)
        block = summary[summary.policy == algo].sort_values('episodes_completed')
        ax.plot(block.episodes_completed, block['mean'], color=P.POLICY_COLORS[algo])
        ax.fill_between(block.episodes_completed, block.ci_low, block.ci_high, color=P.POLICY_COLORS[algo], alpha=.2)
        ax.axhline(0, color='.5', lw=.8)
        ax.set(title=algo.upper(), xlabel='Training episodes completed', ylabel='Validation difference (bps)')
    _finish(fig, out, 'credit_assignment', 'Dense − sparse auction credit at common budgets; pointwise seed-bootstrap intervals. Raw validation, with no checkpoint envelope.')


def write_table(summary, out, name, keys, metrics, caption):
    summary.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
    display = []
    for identity, group in summary.groupby(keys, sort=True):
        if len(keys) == 1:
            identity = (identity,) if not isinstance(identity, tuple) else identity
        row = dict(zip(keys, identity))
        for key in ("policy", "algorithm"):
            if key in row:
                row[key] = str(row[key]).upper()
        for metric, label in metrics.items():
            cell = group.loc[group.metric == metric].iloc[0]
            row[label] = (f"{cell['mean']:.2f} [{cell.ci_low:.2f}, {cell.ci_high:.2f}]"
                          if np.isfinite(cell.ci_low) else f"{cell['mean']:.2f}")
        display.append(row)
    if keys[0] == "market":
        display.sort(key=lambda row: (row["market"] != "Synthetic", row["market"],
                                     POLICIES.index(row["policy"].lower())))
    elif name == "treatments":
        labels = [label for _, label, _, _ in publication.TREATMENT_CONTRASTS]
        display = [{"contrast": label, **{
            algo.upper(): next(row["Effect (bps)"] for row in display
                               if row["contrast"] == label and row["algorithm"] == algo.upper())
            for algo in ALGOS}} for label in labels]
        keys = ["contrast"]
        metrics = {algo: algo.upper() for algo in ALGOS}
    # Fixed-width columns and two-line estimates fit the manuscript's portrait
    # text width. Captions are not escaped by pandas.to_latex.
    key_widths = [.11, .08] if keys[0] == "market" else [.32]
    cell_width = (.90 - sum(key_widths)) / len(metrics)
    widths = [*key_widths, *([cell_width] * len(metrics))]
    columns = "@{}" + "".join(f"p{{{width:.3f}\\linewidth}}" for width in widths) + "@{}"
    headers = [key.title() for key in keys] + list(metrics.values())
    header = " & ".join(r"\raggedright " + tex(h) for h in headers) + r" \tabularnewline"
    lines = [r"\begingroup", r"\small", r"\setlength{\tabcolsep}{2pt}",
             r"\renewcommand{\arraystretch}{1.15}", f"\\begin{{longtable}}{{{columns}}}",
             f"\\caption{{{tex(caption)}}}\\label{{tab:{name}}}\\\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead", r"\toprule", header,
             r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for row in display:
        cells = []
        for key, value in row.items():
            text = tex(value)
            if key not in keys and " [" in text:
                point, interval = text.split(" [", 1)
                text = point + r"\newline{\scriptsize [" + interval + "}"
            cells.append(text)
        lines.append(" & ".join(r"\raggedright " + c for c in cells) + r" \tabularnewline")
    lines.extend([r"\end{longtable}", r"\endgroup"])
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(c, c) for c in str(value))


def discover(root, seeds, *, complete=False, symbol=None, treatments=False):
    settings = {SYNTHETIC, P.HISTORICAL_SETTING}
    if treatments:
        settings.update(s["setting"] for s in publication.TREATMENT_SPECS.values())
    directories = []
    for setting in sorted(settings):
        for path in sorted((root / setting).glob("*/config_resolved.yaml")):
            rd = path.parent
            if rd.name.startswith("_"):
                continue
            seed_path = rd / "seed.txt"
            if not seed_path.exists() or int(seed_path.read_text()) not in seeds:
                continue
            if symbol and setting == P.HISTORICAL_SETTING:
                meta = P.read_metadata(rd)
                if meta.get("symbol") != symbol:
                    continue
            directories.append(rd)
    runs = P.collect_runs(directories)
    if len(runs) != len(directories) or not runs:
        raise ValueError("no complete matching run artifacts")
    headline = [r for r in runs if r.setting in (SYNTHETIC, P.HISTORICAL_SETTING)]
    expected_symbols = [symbol] if symbol else ["MSFT", "JPM", "PG", "GOOGL", "CAT"]
    names = {market(r) for r in headline}
    if complete and names != {"Synthetic", *expected_symbols}:
        raise ValueError("incomplete headline market matrix")
    expected = {(name, algo, seed) for name in names for algo in ALGOS for seed in seeds}
    observed = {(market(r), r.algo, r.seed) for r in headline}
    if observed != expected:
        raise ValueError("unbalanced headline market/algorithm/seed matrix")
    synthetic = [r for r in runs if r.setting != P.HISTORICAL_SETTING]
    if treatments:
        publication.validate_treatment_run_configs(synthetic)
    return runs, headline, synthetic


def generate(root, seeds, *, publication_mode=False, complete=False, symbol=None, treatments=False):
    if publication_mode:
        if symbol or set(seeds) != set(publication.PUBLICATION_SEEDS):
            raise ValueError("publication requires the canonical seeds and all historical tickers")
        complete = treatments = True
    runs, headline, synthetic = discover(root, seeds, complete=complete, symbol=symbol, treatments=treatments)
    from lmm.experiments.mature_reporting import load_protocol, RELATIVE_PATH
    protocol = load_protocol(runs)
    if publication_mode:
        publication.validate_publication_runs(synthetic)
        publication.validate_publication_runs([r for r in runs if r.setting == P.HISTORICAL_SETTING])
    data = seed_outcomes(headline)
    learning = learning_rows(headline)
    curves = learning_summary(learning)
    performance_metrics = {"objective_bps": "Objective (bps)", "pnl_bps": "Net PnL (bps)",
                           "gap_dqn_bps": "Difference vs DQN (bps)",
                           "gap_as_bps": "Difference vs AS (bps)", "gap_twap_bps": "Difference vs TWAP (bps)"}
    auction_metrics = {"auction_value_bps": "Total (bps)", "auction_cash_bps": "Cash (bps)",
                       "auction_risk_relief_bps": "Risk relief (bps)", "auction_fees_bps": "Fees (bps)",
                       "open_abs_inventory_pct": "Open inventory (%)", "close_abs_inventory_pct": "Final inventory (%)"}
    performance = summarize(data, ["market", "policy"], performance_metrics)
    auction = summarize(data.loc[data.policy.isin(ALGOS)], ["market", "policy"], auction_metrics)
    contrasts = publication.build_treatment_contrast_records(synthetic, metric="risk_adjusted_pnl_bps") if treatments else None
    if contrasts is not None:
        contrast_summary = summarize(contrasts.rename(columns={"mean_difference": "effect_bps"}),
                                     ["contrast", "algorithm"], ["effect_bps"])
    bundle = "_publication" if publication_mode else (f"_single_seed{seeds[0]}" if len(seeds) == 1 else "_development")
    destination = root / bundle
    P.apply_style()
    # Stage everything before replacing any report files. Reports never touch
    # eval inputs or their completion manifests, nor any manuscript source.
    with tempfile.TemporaryDirectory(prefix=".report-", dir=root) as tmp:
        stage = Path(tmp)
        figures, tables, audit = [stage / name for name in ("figures", "tables", "audit")]
        for path in (figures, tables, audit):
            path.mkdir()
        disclosure = ""
        if protocol is not None:
            selection_rows = [{"run": name, **{k: entry[k] for k in (
                "selected_episode", "validation_score", "initial_validation_score",
                "improved_over_initial", "evaluated_checkpoint")}}
                for name, entry in sorted(protocol["runs"].items())]
            pd.DataFrame(selection_rows).to_csv(audit / "checkpoint_selection.csv", index=False)
            (audit / "reporting_protocol.json").write_text((root / RELATIVE_PATH).read_text())
            non_improving = [r for r in selection_rows if not r["improved_over_initial"]]
            disclosure = protocol["disclosure"] + f" {len(non_improving)} of {len(selection_rows)} runs did not improve on initialization."
            for row in non_improving:
                label = row['run'].split('/')[-1].replace('__', ', ').replace('_', ' ')
                disclosure += (f" {label}: selected validation objective {row['validation_score']:.6f},"
                               f" initial {row['initial_validation_score']:.6f}; selected episode {row['selected_episode'] + 1}.")
            (audit / "reporting_amendment.tex").write_text(tex(disclosure) + "\n")
        data.to_csv(audit / "economic_by_seed.csv", index=False)
        learning.to_csv(audit / "validation_by_seed.csv", index=False)
        curves.to_csv(audit / "validation_common_support.csv", index=False)
        performance_figure(data, figures)
        learning_figure(learning, curves, figures)
        auction_figure(data, figures)
        write_table(performance, tables, "economic_performance", ["market", "policy"], performance_metrics, TABLE_CAPTIONS["economic_performance"])
        write_table(auction, tables, "auction_mechanism", ["market", "policy"], auction_metrics, TABLE_CAPTIONS["auction_mechanism"])
        names = ["economic_performance", "learning", "auction_mechanism"]
        if contrasts is not None:
            contrasts.to_csv(audit / "treatments_by_seed.csv", index=False)
            treatment_figure(contrasts, figures)
            write_table(contrast_summary, tables, "treatments", ["contrast", "algorithm"],
                        {"effect_bps": "Effect (bps)"}, TABLE_CAPTIONS["treatments"])
            names.append("treatments")
            credit = credit_learning_rows(synthetic)
            credit.to_csv(audit/'credit_learning_by_seed.csv', index=False)
            learning_summary(credit).to_csv(audit/'credit_learning_common_support.csv', index=False)
            credit_learning_figure(credit, figures)
            names.append('credit_assignment')
        status = "Publication matrix" if publication_mode else "DEVELOPMENT ONLY — not publication evidence"
        intro = f"{status}. {len(runs)} runs; master seeds {', '.join(map(str, seeds))}."
        methods = METHODS + "\n" + experiment_scope(runs)
        readme = f"# Research results\n\n{intro}\n\n{methods}\n"
        if disclosure:
            readme += f"\n## Reporting protocol amendment\n\n{disclosure}\n\nSee audit/checkpoint_selection.csv for all 220 decisions and audit/reporting_protocol.json for the bound provenance. Include audit/reporting_amendment.tex in the paper when using these results.\n"
        for name in names:
            readme += f"\n## {name.replace('_', ' ').title()}\n\n{CAPTIONS[name]}\n\n![{name}](figures/{name}.png)\n"
        readme += "\nTables are longtable/booktabs LaTeX and numeric long-form CSV (mean, CI limits and seed count). Include with \\input; longtable cannot be nested inside a table float. CSVs under audit contain every contributing seed estimate and validation point. Source configs, raw episode records, selection details and checkpoint hashes remain in the run directories listed in manifest.json. Report generation only writes to the requested output directory.\n"
        (stage / "README.md").write_text(readme)
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Research results</title><style>body{font:16px/1.55 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#20242b}img{width:100%;height:auto}h1,h2{line-height:1.2}p{max-width:100ch}a{color:#0072B2}</style><h1>Research results</h1><p>' + html.escape(intro) + '</p><p>' + html.escape(methods).replace('\n\n', '</p><p>') + '</p>'
        if disclosure:
            page += '<h2>Reporting protocol amendment</h2><p>' + html.escape(disclosure) + '</p><p><a href="audit/checkpoint_selection.csv">All checkpoint decisions</a> · <a href="audit/reporting_amendment.tex">LaTeX disclosure</a></p>'
        for name in names:
            page += f'<h2>{name.replace("_", " ").title()}</h2><p>{html.escape(CAPTIONS[name])}</p><a href="figures/{name}.pdf">Vector PDF</a>'
            if name not in ("learning", "credit_assignment"):
                page += f' · <a href="tables/{name}.tex">LaTeX table</a> · <a href="tables/{name}.csv">Numeric CSV</a>'
            page += f'<img src="figures/{name}.png" alt="{name.replace("_", " ")}">'
        page += '<p><a href="manifest.json">Input/output provenance</a> · <a href="README.md">Methods and inclusion notes</a></p></html>'
        (stage / "index.html").write_text(page)
        inputs = []
        for run in sorted(runs, key=lambda r: str(r.run_dir)):
            paths = ["config_resolved.yaml", "seed.txt", "git_sha.txt", "metrics.csv", "eval/records.csv", "eval/metadata.yaml",
                     "checkpoints/initial_validation.yaml", "checkpoints/best_selection.yaml",
                     "checkpoints/best_mature_selection.yaml", publication.COMPLETION_MANIFEST_NAME]
            inputs.append({"run_dir": str(run.run_dir.resolve()), "algorithm": run.algo, "seed": run.seed,
                           "setting": run.setting, "symbol": run.symbol,
                           "files": {p: publication._sha256_file(run.run_dir / p) for p in paths if (run.run_dir / p).is_file()}})
        manifest = {"schema": "lmm-focused-report-v1", "publication": publication_mode,
                    "reporting_amendment": disclosure or None,
                    "seeds": seeds, "runs": inputs, "estimand": "equal-seed mean of episode means",
                    "interval": "95% percentile bootstrap of training-seed blocks", "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": 0, "captions": {n: CAPTIONS[n] for n in names},
                    "report_source_sha256": publication._sha256_file(Path(__file__)),
                    "outputs": {str(p.relative_to(stage)): publication._sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        destination.mkdir(exist_ok=True)
        (destination / "manifest.json").unlink(missing_ok=True)
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                target = destination / p.relative_to(stage)
                target.parent.mkdir(exist_ok=True)
                os.replace(p, target)
        os.replace(stage / "manifest.json", destination / "manifest.json")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--include-treatments", action="store_true")
    parser.add_argument("--symbol", choices=["MSFT", "JPM", "PG", "GOOGL", "CAT"])
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("seeds must be distinct nonnegative integers")
    try:
        path = generate(args.root, sorted(args.seeds), publication_mode=args.publication,
                        complete=args.require_complete, symbol=args.symbol, treatments=args.include_treatments)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Report generation failed: {exc}\n")
    print(f"Research report: {path / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
