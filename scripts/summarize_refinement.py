"""Rebuild v17 development evidence from completed, immutable run artifacts."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmm.config import load_config


ROOT = Path('results/refinement_v17')
OUT = Path('docs/verification_v17')
OLD = dict(dqn='dqn_verified', ddpg='ddpg_paper_actor', td3='td3_verified', sac='sac_verified')


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(635001)
    means = values[rng.integers(0,len(values),(4000,len(values)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames = [pd.read_csv(ROOT/f'frozen_{name}'/'records.csv') for name in ('synthetic','history')]
    fresh = pd.concat(frames, ignore_index=True)
    fresh.to_csv(OUT/'frozen_records.csv',index=False)
    for name in ('synthetic','history'):
        shutil.copyfile(ROOT/f'frozen_{name}'/'protocol.json', OUT/f'frozen_{name}_protocol.json')
    for directory in ('calibration','market_units','frozen_audit'):
        (OUT/directory).mkdir(exist_ok=True)
        for path in (ROOT/directory).iterdir():
            if path.suffix in ('.csv','.json') and path.name != 'steps.csv':
                shutil.copyfile(path, OUT/directory/path.name)
    records, contracts = [], []
    for setting, frame in fresh.groupby('setting'):
        historical = setting=='historical_sp500_midquotes'
        suffix = '_history' if historical else ''
        reference = frame[frame.policy=='as'].set_index('env_seed')
        for algo in ('dqn','ddpg','td3','sac'):
            previous = OLD[algo]+suffix
            current = algo+'_asinh'+suffix if algo in ('ddpg','td3') else previous
            old = frame[(frame.run==previous)&(frame.policy==algo)].set_index('env_seed')
            for version, run in [('v16',previous),('v17',current)]:
                rows = frame[(frame.run==run)&(frame.policy==algo)].set_index('env_seed')
                noop = frame[(frame.run==run)&(frame.policy==algo+'_auction_noop')].set_index('env_seed')
                assert rows.index.equals(old.index) and rows.index.equals(noop.index) and rows.index.equals(reference.index)
                auction = rows.objective-noop.objective
                change = rows.objective-old.objective
                difference_as = rows.objective-reference.objective
                result = dict(setting=setting, algorithm=algo, version=version, run=run,
                    n=len(rows), pnl=rows.pnl.mean(), objective=rows.objective.mean(),
                    auction_gain=auction.mean(), auction_pnl_gain=(rows.pnl-noop.pnl).mean(),
                    change=change.mean(), difference_as=difference_as.mean(),
                    opening_inventory=rows.opening_inventory.mean(), inventory_rms=np.sqrt(np.mean(rows.inventory**2)))
                for name, values in [('auction_gain',auction),('change',change),('difference_as',difference_as),('pnl',rows.pnl)]:
                    result[name+'_lo'], result[name+'_hi'] = interval(values)
                records.append(result)
            directory = (ROOT/current if algo in ('ddpg','td3') else Path('results/revised_objective')/current)
            cfg = load_config(directory/'config.yaml')
            active = load_config('configs/base.yaml',f'configs/{setting}.yaml',f'configs/algo/{algo}.yaml')
            fields = ('grid','clob_flow','auction_flow','actions','algo1','reward','features','benchmark','algo')
            for field in fields: assert getattr(cfg,field)==getattr(active,field), (current,field)
            learning_diffs = {key:[value,asdict(active.rl)[key]] for key,value in asdict(cfg.rl).items() if value!=asdict(active.rl)[key]}
            assert set(learning_diffs) <= {'checkpoint_require_initial_improvement'}, (current,learning_diffs)
            validation = json.loads((directory/'validation.json').read_text())
            selected = json.loads((directory/'selection.json').read_text())
            assert selected['objective']>validation[0]['objective']
            contracts.append(dict(run=str(directory), matched_sections=list(fields), rl_changes=learning_diffs,
                selection=selected, checkpoint_sha256=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest()))
    summary = pd.DataFrame(records)
    summary.to_csv(OUT/'frozen_summary.csv',index=False)
    (OUT/'active_config_check.json').write_text(json.dumps(contracts,indent=2))
    attempts, validations, seed_pairs = [], [], []
    for path in sorted(ROOT.glob('*/validation.json')):
        root=path.parent; cfg=load_config(root/'config.yaml')
        for row in json.loads(path.read_text()): validations.append(dict(run=root.name, **row))
        record=dict(run=root.name, algorithm=cfg.algo.name, setting=cfg.experiment.setting,
            seed=cfg.experiment.master_seed, phase_normalization=cfg.rl.phase_normalization,
            inventory_asinh=cfg.rl.auction_inventory_asinh, episodes=cfg.experiment.episodes)
        if (root/'confirmation.csv').exists():
            rows=pd.read_csv(root/'confirmation.csv');policy=rows[rows.policy==cfg.algo.name].set_index('env_seed')
            noop=pd.read_csv(root/'auction_noop.csv').set_index('env_seed')
            record.update(objective=policy.objective.mean(), pnl=policy.pnl.mean(),
                          auction_gain=(policy.objective-noop.objective).mean())
        attempts.append(record)
    pd.DataFrame(attempts).to_csv(OUT/'all_attempts.csv',index=False)
    pd.DataFrame(validations).to_csv(OUT/'all_validation.csv',index=False)
    for algo in ('ddpg','td3'):
        for seed in (628,8841):
            names=(OLD[algo], algo+'_asinh') if seed==628 else (f'{algo}_seed8841_v16',f'{algo}_seed8841_asinh')
            dirs=(Path('results/revised_objective')/names[0],ROOT/names[1]) if seed==628 else (ROOT/names[0],ROOT/names[1])
            policies=[]
            for directory in dirs:
                f=pd.read_csv(directory/'confirmation.csv');policies.append(f[f.policy==algo].set_index('env_seed'))
            delta=policies[1].objective-policies[0].objective
            seed_pairs.append(dict(algorithm=algo, seed=seed, before=policies[0].objective.mean(),
                after=policies[1].objective.mean(), change=delta.mean(), change_lo=interval(delta)[0],change_hi=interval(delta)[1]))
    pd.DataFrame(seed_pairs).to_csv(OUT/'paired_training_seeds.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
    for row, historical in enumerate((False,True)):
        suffix='_history' if historical else ''
        for col,algo in enumerate(('ddpg','td3')):
            ax=axes[row,col]
            for label,directory,color in [('v16',Path('results/revised_objective')/(OLD[algo]+suffix),'#737373'),
                                          ('v17',ROOT/(algo+'_asinh'+suffix),'#167488')]:
                data=pd.DataFrame(json.loads((directory/'validation.json').read_text()))
                ax.plot(data.episode,data.objective,'o-',color=color,label=label,markersize=4)
                s=json.loads((directory/'selection.json').read_text())
                ax.scatter(s['episode'],s['objective'],marker='*',color=color,s=140,zorder=4)
            ax.set_title(algo.upper()+(' · MSFT replay' if historical else ' · synthetic'))
            ax.set_xlabel('Training episode');ax.set_ylabel('Economic validation objective')
            ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(OUT/'learning_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    algorithms=['dqn','ddpg','td3','sac']
    for ax,setting in zip(axes,['synthetic_rough_heston','historical_sp500_midquotes']):
        for version,offset,color in [('v16',-.1,'#737373'),('v17',.1,'#167488')]:
            rows=summary[(summary.setting==setting)&(summary.version==version)].set_index('algorithm').loc[algorithms]
            ax.errorbar(np.arange(4)+offset,rows.auction_gain,
                yerr=[rows.auction_gain-rows.auction_gain_lo, rows.auction_gain_hi-rows.auction_gain],
                fmt='o',capsize=3,color=color,label=version)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),[x.upper() for x in algorithms])
        ax.set_title('Synthetic' if setting.startswith('synthetic') else 'MSFT replay')
        ax.set_ylabel('Paired auction contribution (bps of initial notional)')
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.savefig(OUT/'auction_comparison.png',dpi=160);plt.close(fig)
    print(summary[summary.version=='v17'].round(4).to_string(index=False))


if __name__ == '__main__': main()
