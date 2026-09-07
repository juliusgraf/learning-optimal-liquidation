"""Read-only analysis of frozen v19 inputs; writes only to this directory."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import yaml
from lmm.experiments.make_report import mean_interval, discover
from lmm.experiments.publication import build_treatment_contrast_records

ROOT=Path('results/revision_v19'); OUT=Path('docs/analysis_v19')
report=ROOT/'_publication'
manifest=json.loads((report/'manifest.json').read_text())
economic=pd.read_csv(report/'audit/economic_by_seed.csv')
validation=pd.read_csv(report/'audit/validation_by_seed.csv')
rows=[]; failures=[]; digest_errors=[];shas=set(); record_checks=[]
max_reward_residual=0.;max_replay_residual=0.
for entry in manifest['runs']:
 p=Path(entry['run_dir']);name=p.name
 for rel,digest in entry['files'].items():
  f=p/rel
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
   digest_errors.append(str(f))
 shas.add((p/'git_sha.txt').read_text().strip())
 cfg=yaml.safe_load((p/'config_resolved.yaml').read_text())
 metrics=pd.read_csv(p/'metrics.csv')
 # The logged auction_interim_shaping is already net of cancellation clawbacks.
 shaped=metrics.risk_adjusted_pnl+metrics.clob_shaping_adjustment+metrics.auction_interim_shaping+metrics.auction_terminal_shaping
 max_reward_residual=max(max_reward_residual,float(abs(metrics.training_return-shaped).max()))
 replay=metrics.training_return+metrics.potential_adjustment+metrics.market_baseline_adjustment
 max_replay_residual=max(max_replay_residual,float(abs(metrics.replay_return_unscaled-replay).max()))
 selected=yaml.safe_load((p/'checkpoints/best_selection.yaml').read_text())
 initial=yaml.safe_load((p/'checkpoints/initial_validation.yaml').read_text())
 meta=yaml.safe_load((p/'eval/metadata.yaml').read_text())
 records=pd.read_csv(p/'eval/records.csv')
 factor=1e4/(cfg['grid']['S0']*cfg['grid']['I0'])
 for field in ['training_return','pnl','risk_adjusted_pnl','I_final']:
  if not np.isfinite(metrics[field]).all():failures.append((name,'training',field))
 for field in ['loss_clob','loss_auction']:
  v=metrics[field].dropna()
  if not np.isfinite(v).all():failures.append((name,field))
 for field in ['return_undisc','pnl','risk_adjusted_pnl','I_final']:
  if not np.isfinite(records[field]).all():failures.append((name,'eval',field))
 assert selected['eligibility']['eligible']
 assert selected['metric']=='risk_adjusted_pnl'
 assert len(initial['validation_seeds'])==128
 assert len(set(records.env_seed))==100
 assert set(initial['validation_seeds']).isdisjoint(set(records.env_seed))
 assert meta['policy_evaluation_split']=='test'
 assert np.allclose(records.return_undisc,records.risk_adjusted_pnl,atol=1e-7,rtol=0)
 assert np.allclose(records.initial_inventory-records.clob_exec_qty-records.auction_exec_qty,records.I_final,atol=1e-7,rtol=0)
 assert np.allclose(records.pnl-records.inventory_penalty,records.risk_adjusted_pnl,atol=1e-7,rtol=0)
 assert (records[['clob_shaping_adjustment','auction_interim_shaping','auction_shaping_clawback','auction_terminal_shaping']].abs().max()<1e-10).all()
 curve=metrics.dropna(subset=['eval_risk_adjusted_pnl_mean'])
 last=curve.iloc[-1]
 row=dict(run=str(p),setting=entry['setting'],market=entry['symbol'] or 'Synthetic',algorithm=entry['algorithm'],seed=entry['seed'],
  episodes=len(metrics),selected_episode=selected['episode']+1,
  initial=initial['value']*factor, selected=selected['value']*factor,last=last.eval_risk_adjusted_pnl_mean*factor,
  selected_gain=(selected['value']-initial['value'])*factor,
  last_gain=(last.eval_risk_adjusted_pnl_mean-initial['value'])*factor,
  selected_to_last=(selected['value']-last.eval_risk_adjusted_pnl_mean)*factor,
  clob_max_loss=metrics.loss_clob.max(),auction_max_loss=metrics.loss_auction.max(),
  clob_last_loss=metrics.loss_clob.tail(50).mean(),auction_last_loss=metrics.loss_auction.tail(50).mean())
 rows.append(row)
 assert cfg['rl']['test_seed_namespace']==19001
 assert len(records)==400
 # Strict headline reward checks; all trials retain the same economic parameters.
 assert cfg['reward']['lambda_inv']==.01 and cfg['reward']['q']==0
 if entry['setting'] in ('synthetic_rough_heston','historical_sp500_midquotes'):
  assert cfg['reward']['shaping_enabled'] and cfg['reward']['auction_shaping_weight']==.0001
  assert cfg['actions']['auction_anchor']=='indicative'
  if entry['algorithm']=='ddpg':assert cfg['algo']['hyperparams']['critic_layer_norm']

assert len(manifest['runs'])==440 and len(shas)==1 and not failures and not digest_errors
assert max_reward_residual<1e-7 and max_replay_residual<1e-7
_,_,treatment_runs=discover(ROOT,manifest['seeds'],complete=True,treatments=True)
rebuilt=build_treatment_contrast_records(treatment_runs,metric='risk_adjusted_pnl_bps')
saved=pd.read_csv(report/'audit/treatments_by_seed.csv')
keys=['contrast_key','algorithm','master_seed']
assert np.allclose(rebuilt.sort_values(keys).mean_difference,saved.sort_values(keys).mean_difference,atol=1e-10)
learning=pd.DataFrame(rows);learning.to_csv(OUT/'learning_audit.csv',index=False)
headline=learning[learning.setting.isin(['synthetic_rough_heston','historical_sp500_midquotes'])]
summary=[]
for algo,d in headline.groupby('algorithm'):
 summary.append(dict(algorithm=algo,n=len(d),selected_improves=int((d.selected_gain>0).sum()),last_improves=int((d.last_gain>0).sum()),
  selected_to_last_mean=d.selected_to_last.mean(),selected_to_last_max=d.selected_to_last.max(),regress_gt2=int((d.selected_to_last>2).sum()),
  median_episodes=d.episodes.median(),max_clob_loss=d.clob_max_loss.max(),max_auction_loss=d.auction_max_loss.max()))
pd.DataFrame(summary).to_csv(OUT/'learning_summary.csv',index=False)
print('LEARNING',pd.DataFrame(summary).to_string(index=False))
print('WORST REGRESSIONS',headline.nlargest(12,'selected_to_last')[['market','algorithm','seed','initial','selected','last','selected_to_last']].to_string(index=False))
print('INITIAL FAILURES',headline[headline.selected_gain<=0][['market','algorithm','seed','initial','selected','last']].to_string(index=False))
print('TEST INITIAL',economic[economic.policy.isin(['dqn','ddpg','td3','sac'])].groupby('policy').agg(n=('gap_initial_bps','size'),improved=('gap_initial_bps',lambda x:(x>0).sum()),mean_gain=('gap_initial_bps','mean')).to_string())
# Historical macro effects preserve all ticker outcomes inside each seed.
h=economic[economic.market!='Synthetic'].groupby(['policy','seed']).mean(numeric_only=True).reset_index()
macro=[]
for algo,d in h.groupby('policy'):
 for col in ['objective_bps','pnl_bps','gap_dqn_bps','gap_as_bps','auction_value_bps','auction_cash_bps','auction_risk_relief_bps']:
  mu,lo,hi=mean_interval(d[col]);macro.append(dict(policy=algo,metric=col,mean=mu,ci_low=lo,ci_high=hi))
pd.DataFrame(macro).to_csv(OUT/'historical_macro.csv',index=False)
print('MACRO',pd.DataFrame(macro)[pd.DataFrame(macro).metric.isin(['objective_bps','gap_dqn_bps','auction_value_bps'])].to_string(index=False))
# Paired direct auction-use gaps, distinct from total economic advantage.
auc=[]
for group,table in [('Synthetic',economic[economic.market=='Synthetic']),('Historical fixed-ticker mean',h)]:
 base=table[table.policy=='dqn'].set_index('seed').auction_value_bps
 for algo in ['ddpg','td3','sac']:
  diff=table[table.policy==algo].set_index('seed').auction_value_bps-base
  mu,lo,hi=mean_interval(diff);auc.append(dict(market=group,policy=algo,mean=mu,ci_low=lo,ci_high=hi))
pd.DataFrame(auc).to_csv(OUT/'auction_gaps_vs_dqn.csv',index=False)
print('AUCTION GAPS',pd.DataFrame(auc).to_string(index=False))
print('AUCTION NEGATIVE SEEDS',economic[economic.policy.isin(['dqn','ddpg','td3','sac'])].groupby(['market','policy']).auction_value_bps.agg(mean='mean',min='min',negative=lambda x:(x<0).sum()).to_string())
verification=dict(runs=440,headline_runs=len(headline),training_episodes=int(learning.episodes.sum()),evaluations_per_policy=100,training_seeds=10,git_shas=sorted(shas),nonfinite_fields=failures,report_input_hash_mismatches=digest_errors,accounting_checks='passed',selection_eligibility='passed',validation_test_seed_separation='passed')
verification.update(shaped_reward_identity_max_residual=max_reward_residual,replay_identity_max_residual=max_replay_residual,matched_treatment_estimates_reproduced=len(rebuilt))
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(verification)
