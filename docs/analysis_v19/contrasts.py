"""Paired cash comparisons and seed-sign summaries from publication audit."""
from pathlib import Path
import pandas as pd
from lmm.experiments.make_report import mean_interval
out=Path('docs/analysis_v19');root=Path('results/revision_v19/_publication')
e=pd.read_csv(root/'audit/economic_by_seed.csv');t=pd.read_csv(root/'audit/treatments_by_seed.csv')
s=t.groupby(['contrast_key','algorithm']).mean_difference.agg(mean='mean',median='median',minimum='min',maximum='max',positive_seed_pairs=lambda x:(x>0).sum())
s.to_csv(out/'treatment_seed_signs.csv')
rows=[]
for market,g in e.groupby('market'):
 reference=g[g.policy=='as'].set_index('seed')
 for algo in ['dqn','ddpg','td3','sac']:
  d=g[g.policy==algo].set_index('seed');mu,lo,hi=mean_interval(d.pnl_bps-reference.pnl_bps)
  rows.append(dict(market=market,algorithm=algo,pnl_gap_mean=mu,ci_low=lo,ci_high=hi,
     risk_penalty_reduction=(reference.pnl_bps-reference.objective_bps-(d.pnl_bps-d.objective_bps)).mean()))
pd.DataFrame(rows).to_csv(out/'cash_vs_as.csv',index=False)
