"""Refit training-only normalization/inventory reference; audit AS calibration.

Training repeats this same seeded reference fit before initial validation and
policy learning. This separate artifact is an audit, never a legacy fit import.
AS impact calibration uses CLOB flow alone; benchmark settlement still changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from lmm.agents.benchmarks import ASBenchmarkAgent
from lmm.config import add_config_cli, config_from_args, environment_contract, save_resolved
from lmm.experiments.train import fit_feature_normalizer
from lmm.rl.loops import SEED_COMPONENTS
from lmm.utils.seeding import seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    add_config_cli(parser)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--symbol')
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    save_resolved(cfg, args.output/'config_resolved.yaml')
    seeds = seed_everything(cfg.experiment.master_seed, SEED_COMPONENTS, seed_torch=True)
    normalizer, episode_seeds = fit_feature_normalizer(cfg, seeds, symbol=args.symbol)
    calibration = ASBenchmarkAgent(cfg).calibrate(rng_k=seeds.generators['as_calibration'])
    (args.output/'reference.yaml').write_text(yaml.safe_dump(dict(
        purpose=__doc__, fit_split='train', symbol=args.symbol,
        clearing_mechanism=cfg.auction_flow.clearing_mechanism,
        environment_contract=environment_contract(cfg),
        artifact_schema_version=cfg.experiment.artifact_schema_version,
        master_seed=cfg.experiment.master_seed, episode_seeds=episode_seeds,
        seed_stream='normalizer_env', policy_seed_stream='normalizer_policy',
        state=normalizer.state_dict(), as_calibration={k: float(v) for k,v in calibration.items()},
    ), sort_keys=False))


if __name__ == '__main__':
    main()
