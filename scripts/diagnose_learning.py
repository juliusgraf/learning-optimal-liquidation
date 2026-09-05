"""Bounded development learning checks; never touches publication/test outputs."""
from __future__ import annotations

import argparse
import json
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from lmm.agents.benchmarks import ASBenchmarkAgent, TWAPBenchmarkAgent
from lmm.config import load_config, economic_evaluation_config, save_resolved
from lmm.env.mdp import make_env
from lmm.experiments.train import make_agent, fit_feature_normalizer, wrap_env_for_agent, draw_seed
from lmm.rl.loops import SEED_COMPONENTS, run_episode
from lmm.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--algo', required=True, choices=['dqn', 'ddpg', 'td3', 'sac'])
    p.add_argument('--seed', type=int, default=613)
    p.add_argument('--episodes', type=int, default=120)
    p.add_argument('--validation-size', type=int, default=32)
    p.add_argument('--every', type=int, default=30)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--override', action='append', default=[])
    p.add_argument('--setting', default='synthetic_rough_heston')
    p.add_argument('--symbol', default=None)
    p.add_argument('--config', action='append', default=[])
    p.add_argument('--resolved-config', type=Path,
                   help='Load a saved complete config alone, without inheriting current base defaults.')
    args = p.parse_args()
    if not 1 <= args.episodes <= 200:
        p.error('This diagnostic is capped at 200 episodes; use the production trainer for full runs.')
    if args.validation_size < 1 or args.every < 1:
        p.error('validation-size and every must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.resolved_config and args.config:
        p.error('--resolved-config and --config cannot be combined')
    sources = ([args.resolved_config] if args.resolved_config else
               ['configs/base.yaml', f'configs/{args.setting}.yaml',
                f'configs/algo/{args.algo}.yaml', *args.config])
    cfg = load_config(*sources,
                      overrides=args.override + [f'experiment.episodes={args.episodes}',
                                                 f'experiment.master_seed={args.seed}'])
    if cfg.algo.name != args.algo:
        p.error('--algo differs from the saved resolved configuration')
    save_resolved(cfg, args.output / 'config.yaml')
    seeds = seed_everything(args.seed, SEED_COMPONENTS)
    agent = make_agent(cfg, seeds)
    normalizer, norm_seeds = fit_feature_normalizer(cfg, seeds, symbol=args.symbol)
    agent.set_feature_normalizer(normalizer)
    env = wrap_env_for_agent(make_env(cfg, symbol=args.symbol, data_split='train'), cfg)
    ecfg = economic_evaluation_config(cfg)
    val_env = wrap_env_for_agent(make_env(ecfg, symbol=args.symbol, data_split='validation'), cfg)
    val_seeds = [draw_seed(seeds.generators['env_eval']) for _ in range(args.validation_size)]
    # Fresh development confirmation paths; intentionally not the final-test stream.
    confirm_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 99173]))
    confirm_seeds = [draw_seed(confirm_rng) for _ in range(max(128, args.validation_size))]
    (args.output / 'protocol.json').write_text(json.dumps(dict(
        purpose='bounded development; no final test', seed=args.seed,
        setting=cfg.experiment.setting, symbol=args.symbol, evaluation_data_split='validation',
        auction_shaping_weight=cfg.reward.auction_shaping_weight,
        episodes=args.episodes, normalizer_seeds=norm_seeds,
        packages={p: version(p) for p in ('numpy', 'torch', 'gymnasium', 'stable-baselines3')},
        validation_seeds=val_seeds, confirmation_seeds=confirm_seeds,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for root in ('src/lmm', 'configs') for p in sorted(Path(root).rglob('*'))
                       if p.suffix in ('.py', '.yaml')},
    ), indent=2))

    def evaluate(policy, environment, paths, label):
        rows = []
        for i, s in enumerate(paths):
            policy.bind(environment)
            policy.start_episode(i)
            r = run_episode(environment, policy, s, chi=1.0, train=False)
            rows.append(dict(policy=label, env_seed=s, pnl=r.pnl, objective=r.risk_adjusted_pnl,
                             inventory=r.i_final, auction_qty=r.auction_exec_qty,
                             inventory_at_auction_open=r.initial_inventory-r.clob_exec_qty,
                             clob_qty=r.clob_exec_qty, cancels=r.cancel_count))
        return rows

    validations, training = [], []
    best = -float('inf')
    for e in range(args.episodes + 1):
        if e % args.every == 0 or e == args.episodes:
            rows = evaluate(agent, val_env, val_seeds, args.algo)
            mean = pd.DataFrame(rows).drop(columns=['env_seed']).select_dtypes('number').mean().to_dict()
            mean.update(episode=e, updates=agent.checkpoint_update_counts)
            validations.append(mean)
            print(json.dumps(mean), flush=True)
            eligible = (agent.checkpoint_update_counts['clob'] >= cfg.rl.checkpoint_min_clob_updates
                        and (not cfg.experiment.auction_enabled or
                             agent.checkpoint_update_counts['auction'] >= cfg.rl.checkpoint_min_auction_updates))
            eligible = eligible and (not cfg.rl.checkpoint_require_initial_improvement
                                     or mean['objective'] > validations[0]['objective'])
            if e and eligible and mean['objective'] > best:
                best = mean['objective']
                agent.save(args.output / 'best.pt')
                (args.output / 'selection.json').write_text(json.dumps(mean, indent=2))
            (args.output / 'validation.json').write_text(json.dumps(validations, indent=2))
        if e == args.episodes:
            break
        agent.start_episode(e)
        r = run_episode(env, agent, draw_seed(seeds.generators['env_train']), chi=1.0, train=True)
        assert np.isclose(r.replay_return_unscaled, r.training_return + r.potential_adjustment + r.market_baseline_adjustment)
        training.append(dict(episode=e, pnl=r.pnl, objective=r.risk_adjusted_pnl,
                             training_return=r.training_return, replay_return=r.replay_return_unscaled,
                             potential_adjustment=r.potential_adjustment,
                             market_baseline_adjustment=r.market_baseline_adjustment,
                             clob_shaping=r.clob_shaping_adjustment,
                             auction_shaping=r.auction_interim_shaping,
                             inventory_at_auction_open=r.initial_inventory-r.clob_exec_qty,
                             terminal_inventory=r.i_final, auction_qty=r.auction_exec_qty,
                             **r.diagnostics))
    pd.DataFrame(training).to_csv(args.output / 'training.csv', index=False)
    if not (args.output / 'best.pt').exists():
        raise RuntimeError('No maturity-eligible checkpoint: inspect validation.json; increase the bounded budget or explicitly use smoke maturity thresholds.')
    agent.load(args.output / 'best.pt')
    rows = evaluate(agent, val_env, confirm_seeds, args.algo)
    for cls, label in [(ASBenchmarkAgent, 'as'), (TWAPBenchmarkAgent, 'twap')]:
        policy = cls(ecfg)
        if label == 'as':
            calibration = policy.calibrate(rng_k=seeds.generators['as_calibration'])
            (args.output / 'as_calibration.json').write_text(json.dumps(calibration, indent=2))
        rows.extend(evaluate(policy, make_env(ecfg, symbol=args.symbol, data_split='validation'), confirm_seeds, label))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'confirmation.csv', index=False)
    class AuctionNoop:
        def __getattr__(self, name):
            return getattr(agent, name)

        def act(self, obs, mask, phase, **kwargs):
            if phase == 'auction':
                if args.algo == 'dqn':
                    return 0
                dim = agent.models['auction'].action_space.shape[0]
                return np.array([-1., 0., -1.], dtype=np.float32)[:dim]
            return agent.act(obs, mask, phase, **kwargs)

    counterfactual = pd.DataFrame(evaluate(AuctionNoop(), val_env, confirm_seeds, 'auction_noop'))
    counterfactual.to_csv(args.output / 'auction_noop.csv', index=False)
    print(frame.groupby('policy')[['pnl', 'objective', 'inventory', 'auction_qty', 'clob_qty', 'cancels']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
