"""Phase-specific H diagnostics on training/validation paths, never final test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lmm.agents.benchmarks import TWAPBenchmarkAgent
from lmm.config import load_config, economic_evaluation_config, environment_contract, save_resolved
from lmm.env.mdp import make_env
from lmm.rl.loops import run_episode


class NoOrders(TWAPBenchmarkAgent):
    def act(self, obs, mask, phase, *, eval_mode=False):
        return 0


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--setting',default='synthetic_rough_heston')
    p.add_argument('--symbol')
    p.add_argument('--episodes',type=int,default=256)
    p.add_argument('--config', action='append', default=[], help='additional overlays after setting and algorithm')
    p.add_argument('--override', '-o', action='append', default=[])
    a=p.parse_args()
    if not 1<=a.episodes<=512:p.error('At most 512 forecast paths per split')
    cfg=economic_evaluation_config(load_config('configs/base.yaml',f'configs/{a.setting}.yaml','configs/algo/dqn.yaml', *a.config,
        overrides=[*a.override, 'algo1.clob_forecast_weights=[]']))
    a.output.mkdir(parents=True,exist_ok=False)
    save_resolved(cfg, a.output/'config_resolved.yaml')
    rows=[]
    for split,seed in [('train',194713),('validation',194719)]:
        env=make_env(cfg,symbol=a.symbol,data_split=split); policy=NoOrders(cfg)
        for ep,s in enumerate(np.random.default_rng(seed).integers(0,2**31-1,a.episodes)):
            policy.bind(env);r=run_episode(env,policy,int(s),chi=1.,train=False)
            rows.extend(dict(split=split,episode=ep,env_seed=int(s),**x) for x in r.forecast_records)
    d=pd.DataFrame(rows); d['bin']=np.minimum((4*d.time/cfg.grid.tau_op).astype(int),4)
    # Restricted least squares: a convex combination of mid and raw H,
    # trained on full paths only. No intercept estimates a favorable drift.
    weights=[]
    for b in range(4):
        x=d[(d.split=='train')&(d.phase=='clob')&(d.bin==b)]
        delta=(x.h_cl-x.s_mid).to_numpy(); y=(x.s_cl-x.s_mid).to_numpy()
        weights.append(float(np.clip(delta@y/max(delta@delta,1e-12),0,1)))
    d['calibrated_h']=d.s_mid+np.array(weights+[1.])[d.bin.to_numpy()]*(d.h_cl-d.s_mid)
    for name,col in [('raw','h_cl'),('mid','s_mid'),('calibrated','calibrated_h')]:
        d[name+'_ae']=abs(d[col]-d.s_cl); d[name+'_se']=(d[col]-d.s_cl)**2
    d.to_csv(a.output/'forecasts.csv',index=False)
    blocks=d.groupby(['split','phase','episode'])[[c for c in d if c.endswith(('_ae','_se'))]].mean()
    blocks.to_csv(a.output/'error_by_episode.csv')
    summary=blocks.groupby(['split','phase']).mean()
    summary.to_csv(a.output/'summary.csv')
    d.groupby(['split','phase'])[[c for c in d if c.endswith(('_ae','_se'))]].mean().to_csv(a.output/'decision_weighted_summary.csv')
    (a.output/'aggregation.json').write_text(json.dumps(dict(
        summary='equal path weights within split and phase',
        decision_weighted_summary='equal observation weights; retained for comparison',
        coefficients='unchanged restricted least squares on training observations'),indent=2))
    (a.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__,episodes_per_split=a.episodes,
        clearing_mechanism=cfg.auction_flow.clearing_mechanism,
        environment_contract=environment_contract(cfg),artifact_schema_version=cfg.experiment.artifact_schema_version,
        setting=a.setting,symbol=a.symbol,weights=weights,train_seed=194713,validation_seed=194719,
        fit='no-intercept restricted least squares, four equal CLOB bins, training only',
        clob_bin_width=cfg.grid.tau_op/4,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    (a.output/'forecast_overlay.yaml').write_text(yaml.safe_dump({'algo1':{
        'clob_forecast_weights':weights, 'clob_forecast_mechanism':cfg.auction_flow.clearing_mechanism}}))
    print('weights',weights);print(summary.to_string())


if __name__=='__main__':main()
