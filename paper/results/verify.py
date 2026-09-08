"""Check the manuscript's rounded cells and qualitative claims against v20 evidence."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
import yaml
from lmm.experiments.make_report import mean_interval
from lmm.experiments.retained_history import RetainedHistory
from lmm.experiments.publication import validate_completion_manifest
from lmm.experiments.cashflow_comparison import run_path
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
history=RetainedHistory.load(REPO,REPORT.parent)
assert history.state['status']=='complete' and history.state['max_step_minutes']==0.25
history.revalidate()
publication=json.loads((REPORT/'manifest.json').read_text())
assert publication['publication'] and len(publication['runs'])==440
assert publication['retained_historical_provenance']['synthetic_training_revision']==history.state['launch_git_sha']
for rel,digest in publication['outputs'].items():assert sha(REPORT/rel)==digest,rel
synthetic=set()
for run in publication['runs']:
 rd=Path(run['run_dir'])
 for rel,digest in run['files'].items():assert sha(rd/rel)==digest,(rd,rel)
 if run['setting']!='historical_sp500_midquotes':synthetic.add(rd)
for key in ('cashflow','economic_dense_h'):
 root=REPO/f'results/revision_v20_{key}'
 m=json.loads((root/'_comparison/manifest.json').read_text())
 observed=[]
 for algo in ('dqn','ddpg','td3','sac'):
  for seed in m['seeds']:
   for base,control in ((REPORT.parent,False),(root,True)):
    rd=run_path(base,algo,seed,cashflow=control,conditioned=key=='economic_dense_h')
    observed.append(json.loads((rd/'pipeline_complete.json').read_text()))
    synthetic.add(rd)
 assert observed==m['inputs'],key
 by=pd.read_csv(root/'_comparison/by_seed.csv')
 summary=pd.read_csv(root/'_comparison/comparison.csv').set_index('algo')
 for algo,g in by.groupby('algo'):
  assert np.allclose(mean_interval(g.difference_bps),summary.loc[algo,['difference_bps','ci_low','ci_high']].to_numpy(float))
assert len(synthetic)==320
for rd in sorted(synthetic):
 validate_completion_manifest(rd)
 cfg=yaml.safe_load((rd/'config_resolved.yaml').read_text())
 assert cfg['midprice']['rough_heston']['rough_heston_max_step_minutes']==0.25
 assert (rd/'git_sha.txt').read_text().strip()==history.state['launch_git_sha']
analysis=json.loads((HERE/'analysis_inputs.json').read_text())
for rel,digest in analysis['inputs'].items():assert sha(REPO/rel)==digest,rel
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
# Market-specific benchmark evidence is checked independently against unrounded seed data.
supp_manifest=json.loads((HERE/'supplement_manifest.json').read_text())
for section in ('inputs','outputs'):
 for rel,digest in supp_manifest[section].items():assert sha(REPO/rel)==digest,rel
assert supp_manifest['source_script_sha256']==sha(HERE/'supplement.py')
assert main.index(r'\input{results/benchmark_supplement}') < main.index(r'\section{Conclusion}')
assert r'\appendix' not in main
assert main.count(r'\ref{tab:benchmark_supplement}')>=3
supp=pd.read_csv(HERE/'benchmark_supplement.csv').set_index(['market','policy'])
assert len(supp)==30 and not supp.index.duplicated().any()
values=[];paired_count=0
for market in ('Synthetic','CAT','GOOGL','JPM','MSFT','PG'):
 for algo in [*algos,'as']:
  g=e[(e.market==market)&(e.policy==algo)]
  assert len(g)==10 and g.seed.nunique()==10 and g.n_episodes.eq(100).all()
  mu,lo,hi=mean_interval(g.pnl_bps)
  expected=[-mu,-hi,-lo]
  assert np.allclose(supp.loc[(market,algo),['ordinary_mean','ordinary_ci_low','ordinary_ci_high']].to_numpy(float),expected,rtol=0,atol=1e-10)
  values.extend(expected)
  if algo in algos:
   if market=='MSFT':assert -mu>0 and -hi<0<-lo
   for reference in ('as','twap'):
    # These saved gaps were formed from matched episode-level comparisons,
    # independently of the supplement generator's difference of seed means.
    expected=mean_interval(g[f'gap_{reference}_bps'])
    assert expected[1]>0
    assert np.allclose(supp.loc[(market,algo),[f'{reference}_mean',f'{reference}_ci_low',f'{reference}_ci_high']].to_numpy(float),expected,rtol=0,atol=1e-10)
    values.extend(expected);paired_count+=1
assert paired_count==48
supp_tex=(HERE/'benchmark_supplement.tex').read_text()
supp_body=supp_tex[supp_tex.index(r'\midrule'):supp_tex.index(r'\bottomrule')]
supp_count=compare(numbers(supp_body),values,2)
assert supp_count==234
credit=pd.read_csv(REPORT/'audit/credit_learning_common_support.csv')
assert (credit[credit.episodes_completed==250].ci_low>0).all()
cash=pd.read_csv(REPO/'results/revision_v20_cashflow/_comparison/by_seed.csv')
assert len(cash)==40 and (cash.difference_bps>0).sum()==39
bad=cash[cash.difference_bps<0].iloc[0];assert bad.algo=='dqn' and bad.seed==314
assert round(-bad.difference_bps,2)==0.15
# Explicit expected findings force a prose review if a later campaign changes them.
expected={'auction_credit':(set(algos),set()), 'auction_access':(set(algos),set()),
          'h_feature':(set(),{'dqn'}), 'h_anchor':(set(),{'ddpg','td3','sac'}),
          'combined_preferences':(set(),set())}
for key,(positive,negative) in expected.items():
 g=t[t.contrast==labels[key]]
 assert set(g.loc[g.ci_low>0,'algorithm'])==positive
 assert set(g.loc[g.ci_high<0,'algorithm'])==negative
prefs=pd.read_csv(REPO/'results/revision_v20_economic_dense_h/_comparison/comparison.csv')
assert set(prefs.loc[prefs.ci_high<0,'algo'])=={'sac'} and not (prefs.ci_low>0).any()
auction=pd.read_csv(REPORT/'tables/auction_mechanism.csv')
uncertain=auction[(auction.metric=='auction_value_bps')&(auction.ci_low<=0)]
assert list(zip(uncertain.market,uncertain.policy))==[('Synthetic','td3')]
assert (auction[auction.metric=='auction_value_bps']['mean']>0).all()
p=e[e.market=='Synthetic'].pivot(index='seed',columns='policy',values='objective_bps')
assert p[algos].mean().idxmax()=='sac'
for algo in ('dqn','ddpg','td3'):
 _,lo,hi=mean_interval(p.sac-p[algo]);assert lo<0<hi
for snippet in ('SAC has the lowest mean','explicit access to H worsens DQN',
                'indicative anchoring worsens penalized shortfall for DDPG, TD3 and SAC',
                'DQN at seed 314','Synthetic TD3 is the only case',
                'auction access lowers penalized shortfall for all four learners',
                'worsens SAC penalized shortfall by $0.16$'):
 assert snippet in main,snippet
for setting in ('synthetic_rough_heston','historical_sp500_midquotes'):
 weights=json.loads((REPORT.parent/'_forecasts'/setting/'protocol.json').read_text())['weights']
 assert '('+','.join(f'{x:.3f}' for x in weights)+')' in main
assert 'H_{t_j}^\\cl=p_{\\alpha,j}' in main
assert 'used only for terminal clearing' not in main
log=(REPO/'paper/build/main.log').read_text()
assert 'Output written on build/main.pdf' in log
assert not re.search(r'(?:LaTeX|Package \w+) Warning|Overfull|Underfull|Undefined control sequence|Fatal error',log)
assert sha(REPO/'paper/main.pdf')==sha(REPO/'paper/build/main.pdf')
report=dict(campaign='revision_v20',empirical_tables=4,numeric_cells_verified=count+supp_count,figure_assets_verified=7,
 benchmark_table_numeric_cells_verified=supp_count,
 paired_reference_intervals_verified=paired_count,ordinary_shortfall_levels_verified=30,
 forecast_headline_runs=240,forecast_episodes_per_run=100,forecast_market_phase_rows=12,
 qualitative_claim_checks='benchmark gaps, raw PnL comparison, forecast improvement, credit at episode 250, 39/40 cash-flow wins and exception, all treatment interval signs, auction uncertainty, synthetic ranking, fitted weights',
 refined_synthetic_runs_verified=320,retained_historical_runs_verified=200,
 retained_historical_provenance_valid=True,max_step_minutes=0.25,
 latex_warnings=0,main_tex_sha256=sha(REPO/'paper/main.tex'),pdf_sha256=sha(REPO/'paper/build/main.pdf'))
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
