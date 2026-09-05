"""Rebuild the complete development ledger and selected-policy evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SELECTED = {
    'synthetic_rough_heston': ['dqn_verified', 'ddpg_paper_actor', 'td3_verified', 'sac_verified'],
    'historical_sp500_midquotes': ['dqn_verified_history', 'ddpg_paper_actor_history',
                                 'td3_verified_history', 'sac_verified_history'],
}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('results/revised_objective'))
    p.add_argument('--output', type=Path, default=Path('docs/verification_v16'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(482199)
    def interval(x):
        x = np.asarray(x, dtype=float)
        return np.quantile(x[rng.integers(len(x), size=(4000, len(x)))].mean(axis=1), [.025, .975])
    ledger, validation, manifest = [], [], []
    for root in sorted(args.root.iterdir()):
        if not (root/'confirmation.csv').exists(): continue
        protocol = json.loads((root/'protocol.json').read_text())
        cfg = yaml.safe_load((root/'config.yaml').read_text())
        algo = cfg['algo']['name']
        data = pd.read_csv(root/'confirmation.csv')
        frame = data[data.policy == algo].set_index('env_seed').sort_index()
        as_ = data[data.policy == 'as'].set_index('env_seed').loc[frame.index]
        twap = data[data.policy == 'twap'].set_index('env_seed').loc[frame.index]
        noop = pd.read_csv(root/'auction_noop.csv').set_index('env_seed').loc[frame.index]
        np.testing.assert_allclose(frame.clob_qty, noop.clob_qty, rtol=0, atol=1e-10)
        val = pd.DataFrame(json.loads((root/'validation.json').read_text()))
        selected = json.loads((root/'selection.json').read_text())['episode']
        gain = frame.objective-noop.objective
        difference = frame.objective-as_.objective
        row = dict(run=root.name, algorithm=algo, setting=protocol['setting'], seed=protocol['seed'],
                   training_episodes=protocol['episodes'], selected_episode=selected,
                   n_confirmation=len(frame), pnl=frame.pnl.mean(), objective=frame.objective.mean(),
                   as_objective=as_.objective.mean(), twap_objective=twap.objective.mean(),
                   difference_as=difference.mean(), difference_twap=(frame.objective-twap.objective).mean(),
                   auction_gain=gain.mean(), auction_pnl_gain=(frame.pnl-noop.pnl).mean(),
                   opening_inventory=noop.inventory.mean(), terminal_inventory=frame.inventory.mean(),
                   inventory_rms=np.sqrt(np.mean(frame.inventory**2)), auction_qty=frame.auction_qty.mean(),
                   auction_abs_qty=frame.auction_qty.abs().mean(), cancels=frame.cancels.mean(),
                   validation_initial=val.objective.iloc[0], validation_last=val.objective.iloc[-1],
                   validation_selected=val.loc[val.episode == selected, 'objective'].iloc[0],
                   alpha=cfg['grid']['alpha'], initial_inventory=cfg['grid']['I0'],
                   beta=cfg['actions']['beta'], child_cap=cfg['actions']['V_max'],
                   lambda_inv=cfg['reward']['lambda_inv'], auction_weight=cfg['reward']['auction_shaping_weight'],
                   reward_scale=cfg['algo']['hyperparams']['reward_scale'],
                   hidden_width=cfg['algo']['hyperparams']['hidden_layers'][0],
                   clob_warmup=cfg['algo']['hyperparams']['min_buffer_clob'],
                   auction_warmup=cfg['algo']['hyperparams']['min_buffer_auction'])
        for label, values in [('objective', frame.objective), ('difference_as', difference),
                              ('auction_gain', gain), ('pnl', frame.pnl)]:
            row[label+'_ci_low'], row[label+'_ci_high'] = interval(values)
        ledger.append(row)
        val['selected'] = val.episode == selected
        validation.append(val.assign(run=root.name, algorithm=algo, setting=protocol['setting']))
        manifest.append(dict(run=root.name, protocol=protocol,
                             files={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                    for name in ['config.yaml', 'best.pt', 'confirmation.csv',
                                                 'validation.json', 'auction_noop.csv', 'training.csv']}))
    ledger = pd.DataFrame(ledger)
    ledger.to_csv(args.output/'all_attempts.csv', index=False)
    all_validation = pd.concat(validation)
    all_validation.to_csv(args.output/'all_validation.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    names = sum(SELECTED.values(), [])
    if set(names).issubset(set(ledger.run)):
        ledger[ledger.run.isin(names)].to_csv(args.output/'selected_development.csv', index=False)
        val = all_validation[all_validation.run.isin(names)]
        val.to_csv(args.output/'selected_validation.csv', index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = ['#2463a9', '#bd5c18', '#188771', '#854fa2']
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.3), layout='constrained')
        for i, (setting, runs) in enumerate(SELECTED.items()):
            for j, name in enumerate(runs):
                a = val[val.run == name]
                ax = axes[i, j]
                ax.plot(a.episode, a.objective, 'o-', color=colors[j], markersize=4)
                ax.axhline(a.objective.iloc[0], color='#888888', ls=':', lw=1)
                ax.axhline(0, color='#777777', lw=.6)
                chosen = a[a.selected]
                ax.scatter(chosen.episode, chosen.objective, marker='*', color='black', s=100, zorder=4)
                ax.set_title(f'{a.algorithm.iloc[0].upper()} · {"synthetic" if i == 0 else "MSFT replay"}')
                ax.set_xlabel('Training episodes')
                ax.set_xticks([0, 60, 120, 180])
                ax.grid(alpha=.2)
                if j == 0: ax.set_ylabel('Economic validation objective')
        fig.suptitle('All validation checkpoints of the selected configuration\nDotted: initial policy; star: selected mature policy; axes retain every observation')
        fig.savefig(args.output/'learning.png', dpi=160)
        plt.close(fig)
    confirmation = args.root/'frozen_confirmation/records.csv'
    if confirmation.exists():
        data = pd.read_csv(confirmation)
        summaries = []
        for setting in SELECTED:
            sample = data[data.setting == setting]
            baseline = sample[sample.policy == 'as'].set_index('env_seed')
            twap = sample[sample.policy == 'twap'].set_index('env_seed')
            for algo in ['dqn', 'ddpg', 'td3', 'sac', 'as', 'twap']:
                frame = sample[sample.policy == algo].set_index('env_seed').sort_index()
                difference = frame.objective-baseline.loc[frame.index].objective
                row = dict(setting=setting, policy=algo, episodes=len(frame), pnl=frame.pnl.mean(),
                           objective=frame.objective.mean(), difference_as=difference.mean(),
                           difference_twap=(frame.objective-twap.loc[frame.index].objective).mean(),
                           opening_inventory=frame.opening_inventory.mean(), inventory=frame.inventory.mean(),
                           inventory_rms=np.sqrt(np.mean(frame.inventory**2)), auction_qty=frame.auction_qty.mean(),
                           auction_abs_qty=frame.auction_qty.abs().mean(), cancels=frame.cancels.mean())
                for label, values in [('pnl', frame.pnl), ('objective', frame.objective), ('difference_as', difference)]:
                    row[label+'_ci_low'], row[label+'_ci_high'] = interval(values)
                if algo not in ['as', 'twap']:
                    noop = sample[sample.policy == algo+'_auction_noop'].set_index('env_seed').loc[frame.index]
                    np.testing.assert_allclose(frame.clob_qty, noop.clob_qty, rtol=0, atol=1e-10)
                    gain = frame.objective-noop.objective
                    row.update(auction_gain=gain.mean(), auction_pnl_gain=(frame.pnl-noop.pnl).mean())
                    row['auction_gain_ci_low'], row['auction_gain_ci_high'] = interval(gain)
                summaries.append(row)
        pd.DataFrame(summaries).to_csv(args.output/'frozen_summary.csv', index=False)
        data.to_csv(args.output/'frozen_records.csv', index=False)
        (args.output/'frozen_protocol.json').write_text((confirmation.parent/'protocol.json').read_text())
    inventory = args.root/'inventory_verified/records.csv'
    if inventory.exists():
        data = pd.read_csv(inventory)
        summaries = []
        for (name, reserve), group in data.groupby(['run', 'reserve']):
            enabled = group[group.enabled].set_index('env_seed').sort_index()
            disabled = group[~group.enabled].set_index('env_seed').loc[enabled.index]
            gain = enabled.objective-disabled.objective
            lo, hi = interval(gain)
            summaries.append(dict(run=name, reserve=reserve, opening_inventory=enabled.opening_inventory.mean(),
                                  auction_gain=gain.mean(), gain_ci_low=lo, gain_ci_high=hi,
                                  pnl_gain=(enabled.pnl-disabled.pnl).mean(),
                                  risk_reduction=.01*(disabled.final_inventory**2-enabled.final_inventory**2).mean(),
                                  signed_quantity=enabled.auction_quantity.mean(),
                                  absolute_quantity=enabled.auction_quantity.abs().mean()))
        pd.DataFrame(summaries).to_csv(args.output/'inventory_probes.csv', index=False)
        data.to_csv(args.output/'inventory_records.csv', index=False)
        (args.output/'inventory_protocol.json').write_text((inventory.parent/'protocol.json').read_text())
        if confirmation.exists():
            frozen = pd.read_csv(args.output/'frozen_summary.csv')
            probes = pd.DataFrame(summaries)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
            algos = ['dqn', 'ddpg', 'td3', 'sac']
            for offset, setting, color, label in [(-.1, 'synthetic_rough_heston', '#2563a5', 'Synthetic'),
                                                  (.1, 'historical_sp500_midquotes', '#bd5c18', 'MSFT replay')]:
                a = frozen[frozen.setting == setting].set_index('policy').loc[algos]
                axes[0].errorbar(np.arange(4)+offset, a.auction_gain,
                    yerr=[a.auction_gain-a.auction_gain_ci_low, a.auction_gain_ci_high-a.auction_gain],
                    fmt='o', color=color, capsize=4, label=label)
            axes[0].set_xticks(range(4), [a.upper() for a in algos])
            axes[0].set_title('Frozen policies: same CLOB, auction on versus no orders')
            axes[0].legend()
            for j, name in enumerate(SELECTED['synthetic_rough_heston']):
                a = probes[probes.run == name].sort_values('reserve')
                axes[1].errorbar(a.reserve, a.auction_gain,
                    yerr=[a.auction_gain-a.gain_ci_low, a.gain_ci_high-a.auction_gain],
                    fmt='o-', color=colors[j], capsize=3, label=algos[j].upper())
            axes[1].set_title('Synthetic: learned auction after feasible CLOB probes')
            axes[1].set_xlabel('Minimum inventory left by the CLOB probe')
            axes[1].set_xticks([0, 10, 20])
            axes[1].legend(ncol=2)
            for ax in axes:
                ax.set_ylabel('Paired economic auction contribution')
                ax.axhline(0, color='#777777', lw=.8)
                ax.grid(alpha=.2)
            fig.suptitle('Closing-auction value depends on inventory and policy\n95% path-bootstrap intervals; no forced auction trading')
            fig.savefig(args.output/'auction.png', dpi=160)
            plt.close(fig)
    print(ledger[['run','pnl','objective','difference_as','auction_gain','validation_last']].round(4).to_string(index=False))


if __name__ == '__main__': main()
