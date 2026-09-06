"""Run only the H-anchored cash-flow follow-up and compare with saved headlines.

The frozen v19 matrix/report remain unchanged. Production adds 40 runs; smoke
also trains its own four headline controls, using a separate nonpublication seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from lmm.config import load_config, to_dict
from lmm.experiments import plotting as P, publication
from lmm.experiments.make_report import economic_rows, mean_interval, require_paired
from lmm.experiments.protocol import PUBLICATION_SEEDS, RESULTS_ROOT
from lmm.experiments.run_matrix import Job, root_lock, run_queue, worker_environment

ARM = "mechanism_economic_cashflow"
SETTING = "synthetic_rough_heston"
DEFAULT_ROOT = Path("results/revision_v19_cashflow")
NOTE = (
    "Shaped headline training minus unshaped economic cash-flow training; both "
    "retain H observations and H-anchored auction quotes. The contrast changes "
    "manuscript preferences AND reward conditioning. It does not isolate either "
    "component. Economic test performance excludes shaping. Positive favors the "
    "headline. Pointwise 95% bootstrap intervals resample training-seed means. "
    "This is a separately declared follow-up to v19, not part of its original matrix."
)


def jobs_for(seeds, *, smoke=False):
    blocks = ("synthetic", ARM) if smoke else (ARM,)
    return [Job(seed, algo, block) for block in blocks for seed in seeds
            for algo in P.ALGO_ORDER]


def run_path(root, algo, seed, *, cashflow):
    suffix = "__" + ARM if cashflow else ""
    return root / (SETTING + suffix) / f"{algo}{suffix}_seed{seed}"


def read_run(path):
    if not (path / "config_resolved.yaml").is_file():
        raise ValueError(f"missing completed run: {path}")
    runs = P.collect_runs([path])
    if len(runs) != 1:
        raise ValueError(f"cannot read exactly one run: {path}")
    run = runs[0]
    # Existing headline runs predate this extension. Bind each to its own saved
    # source identity and completion hashes rather than requiring identical HEAD.
    publication.validate_completed_run(
        run, run.cfg, expected_symbol=None,
        expected_git_sha=(path / "git_sha.txt").read_text().strip())
    return run


def validate_pair(headline, cashflow):
    expected = load_config(
        str(headline.run_dir / "config_resolved.yaml"),
        f"configs/treatment/{ARM}.yaml",
        overrides=[f"experiment.name={cashflow.cfg.experiment.name}",
                   f"experiment.results_root={cashflow.cfg.experiment.results_root}"])
    if to_dict(expected) != to_dict(cashflow.cfg):
        raise ValueError("cash-flow control differs beyond the declared reward intervention")
    if headline.cfg.actions.auction_anchor != "indicative" or not headline.cfg.rl.h_cl_feature_enabled:
        raise ValueError("comparator must be the H-visible, H-anchored headline")
    metrics = pd.read_csv(cashflow.run_dir / "metrics.csv")
    zeros = ["potential_adjustment", "market_baseline_adjustment", "reward_baseline_adjustment",
             "clob_shaping_adjustment", "auction_interim_shaping", "auction_shaping_clawback",
             "auction_terminal_shaping"]
    required = zeros + ["training_return", "replay_return_unscaled", "economic_objective",
                        "initial_mid", "initial_inventory"]
    if metrics.empty or not np.isfinite(metrics[required].to_numpy(float)).all():
        raise ValueError("missing/nonfinite cash-flow reward audit")
    if not np.allclose(metrics[zeros], 0, rtol=0, atol=1e-10):
        raise ValueError("cash-flow training contains reward shaping or conditioning")
    if not np.allclose(metrics.training_return, metrics.replay_return_unscaled, rtol=0, atol=1e-8):
        raise ValueError("replay changes cash-flow rewards")
    economic = metrics.training_return - metrics.initial_mid * metrics.initial_inventory
    if not np.allclose(economic, metrics.economic_objective, rtol=0, atol=1e-8):
        raise ValueError("cash-flow rewards do not sum to economic wealth")


def generate(root, headline_root, seeds):
    rows, inputs = [], []
    for algo in P.ALGO_ORDER:
        for seed in seeds:
            h = read_run(run_path(headline_root, algo, seed, cashflow=False))
            c = read_run(run_path(root, algo, seed, cashflow=True))
            if any(run.seed != seed or run.algo != algo for run in (h, c)):
                raise ValueError('run identity differs from requested algorithm/seed')
            validate_pair(h, c)
            a, b = economic_rows(h, algo), economic_rows(c, algo)
            require_paired(a, b)
            rows.append(dict(algo=algo, seed=seed,
                             headline_bps=a.objective_bps.mean(),
                             cashflow_bps=b.objective_bps.mean(),
                             difference_bps=(a.objective_bps-b.objective_bps).mean()))
            for run in (h, c):
                inputs.append(publication._completion_manifest_payload(run.run_dir))
    data = pd.DataFrame(rows)
    summary = []
    for algo in P.ALGO_ORDER:
        group = data[data.algo == algo]
        mean, low, high = mean_interval(group.difference_bps)
        summary.append(dict(algo=algo, headline_bps=group.headline_bps.mean(),
                            cashflow_bps=group.cashflow_bps.mean(), difference_bps=mean,
                            ci_low=low, ci_high=high, training_seeds=len(group)))
    table = pd.DataFrame(summary)
    out = root / "_comparison"
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "by_seed.csv", index=False)
    table.to_csv(out / "comparison.csv", index=False)
    P.apply_style()
    fig, ax = P.plt.subplots(figsize=(7, 3.5))
    for i, row in enumerate(summary):
        color = P.POLICY_COLORS[row['algo']]
        values = data[data.algo == row['algo']].difference_bps.to_numpy()
        ax.scatter(values, np.full(len(values), i), s=12, alpha=.35, color=color)
        ax.plot(row['difference_bps'], i, 'o', color=color)
        if np.isfinite([row['ci_low'], row['ci_high']]).all():
            ax.hlines(i, row['ci_low'], row['ci_high'], color=color, linewidth=2)
    ax.axvline(0, color='.5', linewidth=.8)
    ax.set_yticks(range(4), [a.upper() for a in P.ALGO_ORDER])
    ax.set_xlabel('Economic performance difference (bps): shaped headline minus cash-flow training')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f"comparison.{ext}", dpi=180)
    P.plt.close(fig)
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Economic cash-flow comparison</title>'
        '<h1>H-anchored training comparison</h1><p>' + NOTE + '</p>'
        + ('<p><strong>Smoke test only: not performance evidence. No uncertainty interval '
           'is estimated from one training seed.</strong></p>' if len(seeds) == 1 else '')
        + table.to_html(index=False, float_format=lambda x: f'{x:.3f}')
        + '<img src="comparison.png" alt="Paired economic training comparison">')
    (out / "manifest.json").write_text(json.dumps(
        dict(note=NOTE, seeds=list(seeds), inputs=inputs), indent=2) + '\n')
    return out / "index.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--headline-root', type=Path, default=Path(RESULTS_ROOT))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--threads-per-job', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.threads_per_job < 1:
        parser.error('worker and thread counts must be positive')
    if args.smoke and args.root == DEFAULT_ROOT:
        parser.error('smoke requires an explicit separate --root')
    seeds = (91031,) if args.smoke else PUBLICATION_SEEDS
    repo = Path(__file__).resolve().parents[3]
    root = args.root.resolve()
    headline_root = root if args.smoke else args.headline_root.resolve()
    if not args.smoke and (root == headline_root or headline_root in root.parents):
        parser.error('follow-up root must be outside the frozen headline results tree')
    jobs = jobs_for(seeds, smoke=args.smoke)
    if args.dry_run:
        print(json.dumps(dict(runs=len(jobs), root=str(root), headline_root=str(headline_root),
                              commands=[j.command(repo, args.smoke) for j in jobs]), indent=2))
        return 0
    try:
        if not args.report_only:
            if not args.smoke:
                if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                                           cwd=repo, text=True).strip():
                    parser.error('full follow-up requires a clean git worktree')
                # Fail before expensive training if saved controls are incomplete.
                for seed in seeds:
                    for algo in P.ALGO_ORDER:
                        read_run(run_path(headline_root, algo, seed, cashflow=False))
            env = worker_environment(args.threads_per_job, root)
            with root_lock(root / '_orchestration'):
                code = run_queue(jobs, workers=args.jobs, env=env,
                                 log_dir=root / '_orchestration', cwd=repo,
                                 command=lambda j: j.command(repo, args.smoke))
                if code:
                    return code
                print(generate(root, headline_root, seeds))
        else:
            print(generate(root, headline_root, seeds))
        return 0
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; active workers stopped.\n')


if __name__ == '__main__':
    raise SystemExit(main())
