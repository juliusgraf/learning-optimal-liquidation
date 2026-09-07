"""Rebuild paper evidence from saved v20 runs; never step or fit the simulator."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from lmm.experiments.make_report import mean_interval

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REPORT=REPO/'results/revision_v20/_publication'
manifest=json.loads((REPORT/'manifest.json').read_text())
assert manifest['publication'] and len(manifest['runs'])==440
inputs={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tracked(p):
 inputs[str(p.relative_to(REPO))]=sha(p)
 return p
for rel,digest in manifest['outputs'].items():
 assert sha(REPORT/rel)==digest,rel
tracked(REPORT/'manifest.json')
blocks=[]
for e in manifest['runs']:
 if e['setting'] not in ('synthetic_rough_heston','historical_sp500_midquotes'):continue
 rd=Path(e['run_dir'])
 path=rd/'eval/h_forecasts.csv'
 completion=json.loads((rd/'pipeline_complete.json').read_text())
 assert sha(path)==completion['files']['eval/h_forecasts.csv'],path
 tracked(path)
 d=pd.read_csv(path,usecols=['policy','episode','env_seed','phase','time','h_error','mid_error'])
 d=d[d.policy==e['algorithm']].copy()
 assert set(d.phase)=={'clob','auction'} and d.time.max()<150 and d.time.min()==0
 assert np.isfinite(d[['h_error','mid_error']].values).all()
 assert d[d.phase=='auction'].time.min()==120
 pairs=d[['episode','env_seed']].drop_duplicates()
 assert len(pairs)==100 and not pairs.episode.duplicated().any()
 for n in ('h','mid'):
  d[n+'_ae']=d[n+'_error'].abs();d[n+'_se']=d[n+'_error']**2
 b=d.groupby(['phase','episode'])[['h_ae','mid_ae','h_se','mid_se']].mean()
 assert b.groupby(level='phase').size().eq(100).all()
 b=b.groupby('phase').mean().reset_index()
 b['algorithm']=e['algorithm'];b['market']=e['symbol'] or 'Synthetic';b['seed']=e['seed'];blocks.append(b)
b=pd.concat(blocks,ignore_index=True)
b.to_csv(OUT/'forecast_by_algorithm_seed.csv',index=False)
b=b.groupby(['market','phase','seed'])[['h_ae','mid_ae','h_se','mid_se']].mean().reset_index()
b['mae_gain']=b.mid_ae-b.h_ae
rows=[]
for (market,phase),g in b.groupby(['market','phase']):
 assert len(g)==10
 mu,lo,hi=mean_interval(g.mae_gain)
 rows.append(dict(market=market,phase=phase,h_mae=g.h_ae.mean(),mid_mae=g.mid_ae.mean(),mae_gain=mu,ci_low=lo,ci_high=hi,h_rmse=np.sqrt(g.h_se.mean()),mid_rmse=np.sqrt(g.mid_se.mean())))
f=pd.DataFrame(rows);f.to_csv(OUT/'forecast_summary.csv',index=False)
e=pd.read_csv(tracked(REPORT/'audit/economic_by_seed.csv'))
rows=[]
for market,d in [('Synthetic',e[e.market=='Synthetic']),('Historical',e[e.market!='Synthetic'].groupby(['policy','seed']).mean(numeric_only=True).reset_index())]:
 for policy,g in d.groupby('policy'):
  # Flip the reported score interval's endpoints; reuse the same bootstrap
  # draws as the figures rather than drawing a new finite bootstrap after negation.
  mu,lo,hi=mean_interval(g.objective_bps)
  rows.append(dict(market=market,policy=policy,IS=-g.pnl_bps.mean(),IS_lambda=-mu,ci_low=-hi,ci_high=-lo))
s=pd.DataFrame(rows);s.to_csv(OUT/'shortfall_summary.csv',index=False)
credits=pd.read_csv(tracked(REPORT/'audit/credit_learning_common_support.csv'))
print('FORECAST\n'+f.round(6).to_string(index=False))
print('SHORTFALL\n'+s.round(6).to_string(index=False))
print('CREDIT COLUMNS',credits.columns.tolist());print(credits.to_string(index=False))
for setting in ('synthetic_rough_heston','historical_sp500_midquotes'):
 p=tracked(REPO/f'results/revision_v20/_forecasts/{setting}/protocol.json')
 print('WEIGHTS',setting,json.loads(p.read_text())['weights'])
(OUT/'analysis_inputs.json').write_text(json.dumps(dict(campaign='revision_v20',report=str(REPORT.relative_to(REPO)),inputs=inputs,forecast_estimand='equal decision weight within phase/episode; equal episodes, algorithms and seeds; exclude terminal; RMSE=sqrt(mean squared error)',bootstrap='10000 sorted-seed percentile resamples, seed 0',source_script_sha256=sha(Path(__file__))),indent=2)+'\n')
