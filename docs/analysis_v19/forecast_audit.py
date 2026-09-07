"""Forecast errors: equal episode, algorithm and seed weights within market."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lmm.experiments.make_report import mean_interval
out=Path('docs/analysis_v19'); manifest=json.loads(Path('results/revision_v19/_publication/manifest.json').read_text())
blocks=[]
for e in manifest['runs']:
 if e['setting'] not in ['synthetic_rough_heston','historical_sp500_midquotes']:continue
 d=pd.read_csv(Path(e['run_dir'])/'eval/h_forecasts.csv',usecols=['policy','episode','env_seed','phase','time','h_error','mid_error'])
 d=d[d.policy==e['algorithm']].copy()
 for n in ['h','mid']:
  d[n+'_ae']=d[n+'_error'].abs();d[n+'_se']=d[n+'_error']**2
 b=d.groupby(['phase','episode'])[['h_ae','mid_ae','h_se','mid_se']].mean().groupby('phase').mean().reset_index()
 b['algorithm']=e['algorithm'];b['market']=e['symbol'] or 'Synthetic';b['seed']=e['seed'];blocks.append(b)
b=pd.concat(blocks,ignore_index=True);b.to_csv(out/'forecast_by_algorithm_seed.csv',index=False)
b=b.groupby(['market','phase','seed'])[['h_ae','mid_ae','h_se','mid_se']].mean().reset_index();b['mae_gain']=b.mid_ae-b.h_ae
r=[]
for (market,phase),g in b.groupby(['market','phase']):
 mu,lo,hi=mean_interval(g.mae_gain)
 r.append(dict(market=market,phase=phase,h_mae=g.h_ae.mean(),mid_mae=g.mid_ae.mean(),mae_gain=mu,ci_low=lo,ci_high=hi,h_rmse=np.sqrt(g.h_se.mean()),mid_rmse=np.sqrt(g.mid_se.mean())))
s=pd.DataFrame(r);s.to_csv(out/'forecast_summary.csv',index=False);print(s.to_string(index=False))
