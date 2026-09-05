"""Summarize the saved repair checks; never train or step an environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = {
    'synthetic': ['dqn_final615', 'ddpg_clip_final', 'td3_clip_final', 'sac_clip_final'],
    'historical_MSFT': ['dqn_historical616', 'ddpg_historical_clip', 'td3_clip_history', 'sac_clip_history'],
}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('results/pathology_diagnostic_20260905'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs/verification'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(91832)

    def ci(values):
        values = np.asarray(values)
        means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
        return np.quantile(means, [.025, .975])

    summaries, confirmations, validations, training, manifest = [], [], [], [], []
    for setting, names in RUNS.items():
        for index, name in enumerate(names):
            root = args.results_root / name
            algo = name.split('_')[0]
            protocol = json.loads((root / 'protocol.json').read_text())
            selection = json.loads((root / 'selection.json').read_text())
            data = pd.read_csv(root / 'confirmation.csv')
            baseline = data[data.policy == 'as'].set_index('env_seed').objective
            noop = pd.read_csv(root / 'auction_noop.csv').set_index('env_seed')
            for policy in ([algo, 'as', 'twap'] if index == 0 else [algo]):
                frame = data[data.policy == policy].set_index('env_seed').sort_index()
                difference = frame.objective - baseline.loc[frame.index]
                low, high = ci(frame.objective)
                dlo, dhi = ci(difference)
                row = dict(setting=setting, seed=protocol['seed'], policy=policy, episodes=len(frame),
                           selected_episode=selection['episode'] if policy == algo else 0,
                           pnl=frame.pnl.mean(), objective=frame.objective.mean(),
                           objective_ci_low=low, objective_ci_high=high,
                           difference_as=difference.mean(), difference_ci_low=dlo, difference_ci_high=dhi,
                           inventory=frame.inventory.mean(), inventory_rms=np.sqrt((frame.inventory**2).mean()),
                           auction_qty=frame.auction_qty.mean(), cancels=frame.cancels.mean())
                if policy == algo:
                    paired = noop.loc[frame.index]
                    np.testing.assert_allclose(frame.clob_qty, paired.clob_qty, atol=1e-10, rtol=0)
                    gain = frame.objective - paired.objective
                    alo, ahi = ci(gain)
                    row.update(auction_gain=gain.mean(), auction_gain_ci_low=alo,
                               auction_gain_ci_high=ahi, opening_inventory=paired.inventory.mean(),
                               auction_pnl_gain=(frame.pnl-paired.pnl).mean())
                summaries.append(row)
                confirmations.append(frame.reset_index().assign(setting=setting, seed=protocol['seed']))
            val = pd.DataFrame(json.loads((root / 'validation.json').read_text()))
            val['selected'] = val.episode == selection['episode']
            validations.append(val.drop(columns='updates').assign(setting=setting, policy=algo))
            train = pd.read_csv(root / 'training.csv')
            columns = [c for c in train if c in ('episode', 'pnl', 'objective', 'training_return',
                       'replay_return', 'potential_adjustment', 'clob_shaping', 'auction_shaping')
                       or c.startswith(('loss_', 'grad_norm_'))]
            training.append(train[columns].assign(setting=setting, policy=algo))
            manifest.append(dict(setting=setting, algorithm=algo, source=str(root),
                                 selection=selection, protocol=protocol,
                                 artifact_sha256={file: hashlib.sha256((root/file).read_bytes()).hexdigest()
                                    for file in ('config.yaml', 'best.pt', 'confirmation.csv',
                                                 'auction_noop.csv', 'training.csv', 'validation.json')}))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / 'summary.csv', index=False)
    pd.concat(confirmations).to_csv(args.output_dir / 'confirmation.csv', index=False)
    validation = pd.concat(validations)
    validation.to_csv(args.output_dir / 'validation.csv', index=False)
    pd.concat(training).to_csv(args.output_dir / 'training.csv', index=False)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    colors = dict(dqn='#2459a6', ddpg='#d56b20', td3='#15836d', sac='#8c489f')
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True, layout='constrained')
    for row, setting in enumerate(RUNS):
        for col, (algo, color) in enumerate(colors.items()):
            ax = axes[row, col]
            data = validation[(validation.setting == setting) & (validation.policy == algo)]
            ax.plot(data.episode, data.objective, 'o-', color=color, markersize=4)
            selected = data[data.selected]
            ax.scatter(selected.episode, selected.objective, marker='*', s=150, color='black', zorder=5)
            ax.axhline(0, color='gray', lw=.8)
            # Symmetric log preserves every large negative early/late value.
            ax.set_yscale('symlog', linthresh=50)
            ax.set_ylim(-3000, 80)
            ax.set_yticks([-1000, -100, -50, 0, 50], labels=['−1000', '−100', '−50', '0', '50'])
            ax.set_xticks([0, 60, 120, 180])
            ax.set_title(f'{algo.upper()} · {"synthetic" if row == 0 else "MSFT replay"}')
            ax.set_xlabel('Training episodes')
            if col == 0:
                ax.set_ylabel('Economic validation objective\n(symmetric log beyond ±50)')
            ax.grid(alpha=.2)
    fig.suptitle('Bounded learning checks: all validation checkpoints\nBlack stars: selected mature joint policies; no convergence claim', fontsize=14)
    fig.savefig(args.output_dir / 'learning.png', dpi=170)
    plt.close(fig)
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
