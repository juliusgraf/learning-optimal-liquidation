"""Bounded late-update diagnostic on a fixed saved replay; no new training paths.

This intentionally revisits finite off-policy data much more often than the
production learner. It is a stress test of numerical stability, not a policy
comparison or a replacement for fresh held-out economic evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lmm.config import load_config, economic_evaluation_config
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=20000)
    args = p.parse_args()
    if not 1 <= args.updates <= 20000:
        p.error('bounded stress requires 1..20000 auction updates (3 CLOB per auction)')
    cfg = load_config(args.run/'config.yaml')
    if cfg.algo.name != 'dqn':
        p.error('DQN only')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.run/'protocol.json').read_text())
    agent = make_agent(cfg, seed_everything(93817, SEED_COMPONENTS))
    checkpoint = args.run/'stress_start.pt'
    agent.load(checkpoint)
    env = make_env(economic_evaluation_config(cfg), symbol=protocol.get('symbol'), data_split='validation')
    paths = protocol['confirmation_seeds'][:48]
    validations, diagnostics = [], []
    frozen_episode = agent._episode
    for step in range(args.updates+1):
        if step % 5000 == 0 or step == args.updates:
            outcomes = []
            for seed in paths:
                agent.bind(env)
                r = run_episode(env, agent, seed, chi=1., train=False)
                outcomes.append(r.risk_adjusted_pnl)
            row = dict(auction_updates=step, clob_updates=3*step,
                       objective_mean=float(np.mean(outcomes)),
                       objective_se=float(np.std(outcomes, ddof=1)/np.sqrt(len(outcomes))))
            validations.append(row)
            print(json.dumps(row), flush=True)
            pd.DataFrame(validations).to_csv(args.output/'validation.csv', index=False)
        if step == args.updates:
            break
        agent.set_train(True)
        agent.start_episode(frozen_episode)
        for phase in ['clob', 'clob', 'clob', 'auction']:
            batch = agent.replay[phase].sample(agent.hp.batch_size)
            stats = agent._gradient_step(phase, batch)
            agent._update_count[phase] += 1
            agent._sync_target_after_update(phase)
            if step % 250 == 0:
                with torch.no_grad():
                    q = agent.q[phase](torch.as_tensor(batch.obs, device=agent.device))
                diagnostics.append(dict(step=step, phase=phase,
                                        q_abs_max=float(q.abs().max()), **stats))
                pd.DataFrame(diagnostics).to_csv(args.output/'diagnostics.csv', index=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(
        purpose=__doc__, run=str(args.run), updates=args.updates,
        learning_rate_episode=frozen_episode, evaluation_split='validation',
        evaluation_seeds=paths, source_protocol=protocol,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    ), indent=2))


if __name__ == '__main__':
    main()
