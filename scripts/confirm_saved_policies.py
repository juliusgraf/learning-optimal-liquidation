"""Fresh development paths for frozen checkpoints; no optimization or selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.config import load_config, economic_evaluation_config
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent, wrap_env_for_agent
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=256)
    p.add_argument('--seed', type=int, default=791483)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = np.random.default_rng(args.seed).integers(0, 2**31-1, args.episodes).tolist()
    rows, manifest, done_benchmarks = [], [], set()
    for root in args.run:
        cfg = load_config(root/'config.yaml')
        protocol = json.loads((root/'protocol.json').read_text())
        setting, symbol = protocol['setting'], protocol.get('symbol')
        seeds = seed_everything(protocol['seed'], SEED_COMPONENTS)
        agent = make_agent(cfg, seeds)
        agent.load(root/'best.pt')
        ecfg = economic_evaluation_config(cfg)
        env = wrap_env_for_agent(make_env(ecfg, symbol=symbol, data_split='validation'), cfg)
        class NoAuction:
            def __getattr__(self, name): return getattr(agent, name)
            def act(self, obs, mask, phase, **kwargs):
                if phase == 'clob': return agent.act(obs, mask, phase, **kwargs)
                if cfg.algo.name == 'dqn': return 0
                return np.array([-1., 0., -1.], dtype=np.float32)[:agent.models['auction'].action_space.shape[0]]
        def evaluate(policy, label, environment):
            for i, seed in enumerate(paths):
                policy.bind(environment)
                policy.start_episode(i)
                result = run_episode(environment, policy, seed, chi=1., train=False)
                rows.append(dict(run=root.name, setting=setting, symbol=symbol, policy=label,
                                 env_seed=seed, pnl=result.pnl, objective=result.risk_adjusted_pnl,
                                 opening_inventory=result.initial_inventory-result.clob_exec_qty,
                                 inventory=result.i_final, auction_qty=result.auction_exec_qty,
                                 clob_qty=result.clob_exec_qty, cancels=result.cancel_count))
        evaluate(agent, cfg.algo.name, env)
        evaluate(NoAuction(), cfg.algo.name+'_auction_noop', env)
        if (setting, symbol) not in done_benchmarks:
            as_ = ASBenchmarkAgent(ecfg)
            calibration = as_.calibrate(rng_k=seeds.generators['as_calibration'])
            (args.output/f'as_{setting}.json').write_text(json.dumps(calibration, indent=2))
            for policy, label in [(as_, 'as'), (TWAPBenchmarkAgent(ecfg), 'twap')]:
                evaluate(policy, label, make_env(ecfg, symbol=symbol, data_split='validation'))
            done_benchmarks.add((setting, symbol))
        manifest.append(dict(run=str(root), config_sha256=hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest(),
                             checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
                             training_protocol=protocol))
        print(root.name, 'complete', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'records.csv', index=False)
    # A comparison may contain multiple frozen policies of the same algorithm.
    # Keep their run identities instead of averaging old and new controllers.
    frame.groupby(['run', 'setting', 'policy']).mean(numeric_only=True).to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__, seed=args.seed,
            paths=paths, data_split='validation', checkpoint_selection='frozen before these paths', runs=manifest), indent=2))


if __name__ == '__main__': main()
