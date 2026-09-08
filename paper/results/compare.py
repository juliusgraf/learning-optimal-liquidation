"""Compare the archived pre-replacement v20 evidence with the refreshed paper.

Read saved evaluations only. Campaign differences include refitting/retraining;
seed labels do not establish coupled Brownian paths or isolate mesh error.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from lmm.experiments.make_report import mean_interval

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ROOT=REPO/'results/revision_v20'
ARCHIVE=Path('_superseded/synthetic_refinement_v1')
inputs={}
def read(path):
    inputs[str(path.relative_to(REPO))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)
rows=[]
def add(exhibit,metric,identity,old,new):
    rows.append(dict(exhibit=exhibit,metric=metric,identity=identity,
                     before=float(old),latest=float(new),change=float(new-old)))
# The archived and replacement publication tables have the same estimands.
for name,keys in [('economic_performance',['market','policy','metric']),
                  ('auction_mechanism',['market','policy','metric']),
                  ('treatments',['contrast','algorithm','metric'])]:
    before=read(ROOT/ARCHIVE/'_publication/tables'/f'{name}.csv').set_index(keys).sort_index()
    latest=read(ROOT/'_publication/tables'/f'{name}.csv').set_index(keys).sort_index()
    pd.testing.assert_index_equal(before.index,latest.index)
    if name!='treatments':
        historical=before.index.get_level_values('market')!='Synthetic'
        pd.testing.assert_frame_equal(before.loc[historical],latest.loc[historical],check_exact=True)
    for index in before.index:
        if name!='treatments' and index[0]!='Synthetic':continue
        for metric in ('mean','ci_low','ci_high'):
            add(name,metric,' / '.join(index),before.loc[index,metric],latest.loc[index,metric])
for key in ('cashflow','economic_dense_h'):
    root=REPO/f'results/revision_v20_{key}'
    old=read(root/ARCHIVE/'_comparison/comparison.csv').set_index('algo')
    new=read(root/'_comparison/comparison.csv').set_index('algo')
    for algo in old.index:
        for metric in ('difference_bps','ci_low','ci_high'):
            add(key,metric,algo,old.loc[algo,metric],new.loc[algo,metric])
# Reconstruct the old paper's episode-first forecast estimand from archived logs.
blocks=[]
for path in sorted((ROOT/ARCHIVE/'synthetic_rough_heston').glob('*/eval/h_forecasts.csv')):
    algo,seed=path.parents[1].name.split('_seed')
    d=read(path)
    d=d[d.policy==algo].copy()
    assert d.time.max()<150 and d.episode.nunique()==100
    for k in ('h','mid'):
        d[k+'_ae']=d[k+'_error'].abs();d[k+'_se']=d[k+'_error']**2
    b=d.groupby(['phase','episode'])[['h_ae','mid_ae','h_se','mid_se']].mean()
    b=b.groupby('phase').mean().reset_index();b['seed']=int(seed);blocks.append(b)
assert len(blocks)==40
b=pd.concat(blocks).groupby(['phase','seed']).mean(numeric_only=True).reset_index()
new=read(HERE/'forecast_summary.csv').query("market == 'Synthetic'").set_index('phase')
for phase,g in b.groupby('phase'):
    gain,lo,hi=mean_interval(g.mid_ae-g.h_ae)
    old=dict(h_mae=g.h_ae.mean(),mid_mae=g.mid_ae.mean(),mae_gain=gain,ci_low=lo,ci_high=hi,
             h_rmse=np.sqrt(g.h_se.mean()),mid_rmse=np.sqrt(g.mid_se.mean()))
    for metric,value in old.items():add('forecast',metric,phase,value,new.loc[phase,metric])
result=pd.DataFrame(rows)
result.to_csv(HERE/'refinement_comparison.csv',index=False)
(HERE/'comparison_inputs.json').write_text(json.dumps(dict(
    before=str((ROOT/ARCHIVE).relative_to(REPO)),latest='revision_v20, maximum step 0.25 minutes',
    interpretation='Descriptive changes between recalibrated and retrained campaigns; not fixed-policy coupled-mesh errors. Historical report table values agree exactly.',
    inputs=inputs,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
lines=['# Synthetic v20: pre-replacement versus latest results','',
       'Before: archived decision-grid generator. Latest: maximum internal step 0.25 minutes, new training-only calibration and retrained/reselected policies. Historical report table values agree exactly. These are descriptive campaign changes, not coupled-mesh numerical errors or causal estimates of refinement alone.','',
       'All shortfalls and treatment effects below are in basis points. Intervals are pointwise 95% seed bootstrap intervals within each campaign; no interval for the between-campaign change is claimed.','',
       '## Headline inventory-penalized shortfall (lower is better)','',
       '| Policy | Before | Latest | Latest − before |','|---|---:|---:|---:|']
for algo in ('dqn','ddpg','td3','sac','as','twap'):
    r=result[(result.exhibit=='economic_performance')&(result.metric=='mean')&(result.identity==f'Synthetic / {algo} / objective_bps')].iloc[0]
    lines.append(f'| {algo.upper()} | {-r.before:.2f} | {-r.latest:.2f} | {-r.change:+.2f} |')
lines+=['','## Matched treatment effects (positive favors first condition)','',
        '| Contrast / learner | Before [95% CI] | Latest [95% CI] |','|---|---:|---:|']
for (exhibit,identity),g in result[result.exhibit.isin(['treatments','cashflow','economic_dense_h'])].groupby(['exhibit','identity'],sort=False):
    g=g.set_index('metric');key='mean' if exhibit=='treatments' else 'difference_bps'
    label=identity.replace(' / effect_bps','') if exhibit=='treatments' else f'{exhibit} / {identity.upper()}'
    vals=[f'{g.loc[key,col]:.2f} [{g.loc["ci_low",col]:.2f}, {g.loc["ci_high",col]:.2f}]' for col in ('before','latest')]
    lines.append(f'| {label} | '+ ' | '.join(vals)+' |')
lines+=['','## Synthetic forecast errors (model-price units)','',
        '| Phase / metric | Before | Latest |','|---|---:|---:|']
for r in result[(result.exhibit=='forecast')&(~result.metric.isin(['ci_low','ci_high']))].itertuples():
    lines.append(f'| {r.identity} / {r.metric} | {r.before:.4f} | {r.latest:.4f} |')
lines+=['','The CSV includes every synthetic economic, auction-mechanism and treatment mean and interval endpoint, plus forecast metrics. `comparison_inputs.json` binds all source files. Reproduce with `.venv/bin/python paper/results/compare.py`.','']
(HERE/'refinement_comparison.md').write_text('\n'.join(lines))
print(f'Compared {len(rows)} synthetic values; historical report tables are unchanged.')
