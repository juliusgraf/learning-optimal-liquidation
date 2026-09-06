"""Frozen-policy transfer across historical validation stocks; no learner updates.

The checkpoints were selected on MSFT validation. Other stocks share training
dates and the training pool, so this diagnoses transfer across validation price
paths, not generalization to unseen securities or a fresh historical test week.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.config import economic_evaluation_config, load_config
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent, wrap_env_for_agent
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 128:
        parser.error('diagnostic requires 1..128 validation paths per stock')
    args.output.mkdir(parents=True, exist_ok=False)
    paths = np.random.default_rng(980917).integers(0, 2**31-1, args.episodes).tolist()
    rows, manifests = [], []
    for root in args.run:
        cfg = load_config(root/'config.yaml')
        if cfg.midprice.historical is None:
            parser.error('historical checkpoints only')
        protocol = json.loads((root/'protocol.json').read_text())
        if protocol['symbol'] != 'MSFT':
            parser.error('this diagnostic requires checkpoints selected on MSFT')
        agent = make_agent(cfg, seed_everything(792, SEED_COMPONENTS))
        agent.load(root/'best.pt')
        ecfg = economic_evaluation_config(cfg)
        reference = ASBenchmarkAgent(ecfg)
        # A common calibration stream, independent of validation paths.
        reference.calibrate(rng_k=np.random.default_rng(349201))
        for symbol in cfg.midprice.historical.symbols:
            env = wrap_env_for_agent(make_env(ecfg, symbol=symbol, data_split='validation'), cfg)
            for label, policy in [(cfg.algo.name, agent), ('as', reference), ('twap', TWAPBenchmarkAgent(ecfg))]:
                policy_env = env if label == cfg.algo.name else make_env(ecfg, symbol=symbol, data_split='validation')
                for i, seed in enumerate(paths):
                    policy.bind(policy_env)
                    policy.start_episode(i)
                    r = run_episode(policy_env, policy, seed, chi=1., train=False)
                    rows.append(dict(run=root.name, algorithm=cfg.algo.name, symbol=symbol,
                                     policy=label, env_seed=seed, pnl=r.pnl,
                                     objective=r.risk_adjusted_pnl,
                                     opening_inventory=r.initial_inventory-r.clob_exec_qty,
                                     final_inventory=r.i_final))
            print(root.name, symbol, 'complete', flush=True)
            pd.DataFrame(rows).to_csv(args.output/'records.csv', index=False)
        manifests.append(dict(run=str(root), source_protocol=protocol,
                              checkpoint_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/'records.csv', index=False)
    frame.groupby(['run','symbol','policy'])[['pnl','objective','opening_inventory','final_inventory']].mean().to_csv(args.output/'summary.csv')
    (args.output/'protocol.json').write_text(json.dumps(dict(purpose=__doc__, seeds=paths, runs=manifests), indent=2))


if __name__ == '__main__':
    main()
