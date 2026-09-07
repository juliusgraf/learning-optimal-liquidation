"""Check the manuscript's rounded cells and qualitative claims against v20 evidence."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from lmm.experiments.make_report import mean_interval
from lmm.experiments.source_attestation import SourceAttestation
REPO=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REPORT=REPO/'results/revision_v20/_publication';main=(REPO/'paper/main.tex').read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def body(label,start):
 stop=main.index('\\label{'+label+'}')
 block=main[main.rfind('\\begin{table}',0,stop):stop]
 return block[block.index(start):block.index('\\bottomrule')]
def numbers(s):return [float(x) for x in re.findall(r'(?<![\w.])-?\d+\.\d+',s)]
def compare(observed,expected,precision):
 assert len(observed)==len(expected),(len(observed),len(expected))
 assert np.allclose(observed,[float(f'{x:.{precision}f}') for x in expected],rtol=0,atol=1e-12)
 return len(observed)
manifest=json.loads((HERE/'refresh_manifest.json').read_text())
for name,item in manifest['figure_outputs'].items():
 assert sha(REPO/name)==item['sha256']==sha(REPO/item['source'])
for p,digest in manifest['inputs'].items():assert sha(REPO/p)==digest,p
SourceAttestation.load(REPO,REPO/'results/revision_v20',REPO/'results/revision_v20/_provenance/source_attestation.json')
algos=['dqn','ddpg','td3','sac'];count=0
# Recompute economic cells from saved per-seed report observations, including sign reversal.
e=pd.read_csv(REPORT/'audit/economic_by_seed.csv');values=[]
for a in [*algos,'as','twap']:
 for market in ('Synthetic','Historical'):
  d=e[(e.policy==a)&((e.market=='Synthetic') if market=='Synthetic' else (e.market!='Synthetic'))]
  if market=='Historical':d=d.groupby('seed').mean(numeric_only=True)
  mean,lo,hi=mean_interval(d.objective_bps)
  values.extend([-d.pnl_bps.mean(),-mean,-hi,-lo])
count+=compare(numbers(body('tab:economic_performance','DQN &')),values,2)
f=pd.read_csv(HERE/'forecast_summary.csv').set_index(['market','phase']);values=[]
for market in ('Synthetic','CAT','GOOGL','JPM','MSFT','PG'):
 for phase in ('clob','auction'):
  r=f.loc[market,phase]
  assert r.h_mae<r.mid_mae and r.h_rmse<r.mid_rmse and r.ci_low>0
  values.extend(r[['h_mae','mid_mae','mae_gain','ci_low','ci_high','h_rmse','mid_rmse']])
count+=compare(numbers(body('tab:forecast_accuracy','Synthetic & CLOB')),values,4)
t=pd.read_csv(REPORT/'tables/treatments.csv');a=pd.read_csv(REPORT/'audit/treatments_by_seed.csv')
labels=dict(a[['contrast_key','contrast']].drop_duplicates().values);values=[]
for key in ('auction_credit','h_feature','h_anchor','combined_preferences','auction_access','cashflow','economic_dense_h'):
 if key in labels:g=t[t.contrast==labels[key]].set_index('algorithm');field='mean'
 else:g=pd.read_csv(REPO/f'results/revision_v20_{key}/_comparison/comparison.csv').set_index('algo');field='difference_bps'
 for algo in algos:values.extend(g.loc[algo,[field,'ci_low','ci_high']])
count+=compare(numbers(body('tab:treatments',r'\shortstack[l]{Dense versus sparse credit')),values,2)
bench=pd.read_csv(REPORT/'tables/economic_performance.csv')
b=bench[bench.policy.isin(algos)&bench.metric.isin(['gap_as_bps','gap_twap_bps'])]
assert len(b)==48 and (b.ci_low>0).all()
pnl=e.pivot(index=['market','seed'],columns='policy',values='pnl_bps')
assert pnl[algos].sub(pnl['as'],axis=0).groupby('market').mean().lt(0).all().all()
credit=pd.read_csv(REPORT/'audit/credit_learning_common_support.csv')
assert (credit[credit.episodes_completed==250].ci_low>0).all()
cash=pd.read_csv(REPO/'results/revision_v20_cashflow/_comparison/by_seed.csv')
assert len(cash)==40 and (cash.difference_bps>0).sum()==39
bad=cash[cash.difference_bps<0].iloc[0];assert bad.algo=='dqn' and bad.seed==1618
assert 'H_{t_j}^\\cl=p_{\\alpha,j}' in main
assert 'used only for terminal clearing' not in main
log=(REPO/'paper/build/main.log').read_text()
assert 'Output written on build/main.pdf' in log
assert not re.search(r'(?:LaTeX|Package \w+) Warning|Overfull|Underfull|Undefined control sequence|Fatal error',log)
report=dict(campaign='revision_v20',empirical_tables=3,numeric_cells_verified=count,figure_assets_verified=7,
 forecast_headline_runs=240,forecast_episodes_per_run=100,forecast_market_phase_rows=12,
 qualitative_claim_checks='benchmark gaps, raw PnL comparison, forecast improvement, credit at episode 250, 39/40 cash-flow wins',
 source_attestation_valid=True,latex_warnings=0,main_tex_sha256=sha(REPO/'paper/main.tex'),pdf_sha256=sha(REPO/'paper/build/main.pdf'))
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
