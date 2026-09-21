"""Regenerate P-003 from bundled, unrounded seed means; no simulator imports.

The estimator follows make_report.mean_interval and the original manuscript
supplement estimator (manuscript sources are distributed separately):
10,000 resamples, PCG64 seed 0, sorted values, equal seed weights, pointwise CIs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {42, 7, 99, 123, 2024, 314, 577, 811, 1618, 2718}
LEARNERS = ('dqn', 'ddpg', 'td3', 'sac')
MARKETS = ('Synthetic', 'CAT', 'GOOGL', 'JPM', 'MSFT', 'PG')


def interval(values):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) != 10 or not np.isfinite(values).all():
        raise ValueError('exactly ten finite seed means required')
    indices = np.random.default_rng(0).integers(10, size=(10000, 10))
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def generate(source):
    with Path(source).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cells = {}
    if len(rows) != 360:
        raise ValueError('expected 360 market/policy/seed records')
    for row in rows:
        key = row['market'], row['policy'], int(row['seed'])
        if key in cells or int(row['n_episodes']) != 100:
            raise ValueError('duplicate or incomplete paper seed cell')
        cells[key] = row
    expected = {(m, p, s) for m in MARKETS for p in (*LEARNERS, 'as', 'twap') for s in SEEDS}
    if set(cells) != expected:
        raise ValueError('paper market/policy/seed matrix mismatch')
    result = []
    for market in MARKETS:
        for policy in (*LEARNERS, 'as'):
            group = [cells[market, policy, seed] for seed in sorted(SEEDS)]
            mean, lo, hi = interval([float(r['pnl_bps']) for r in group])
            out = dict(market=market, policy=policy, n_seeds=10,
                       ordinary_mean=-mean, ordinary_ci_low=-hi, ordinary_ci_high=-lo)
            if policy in LEARNERS:
                for ref in ('as', 'twap'):
                    diff = [float(r['objective_bps']) - float(cells[market, ref, int(r['seed'])]['objective_bps']) for r in group]
                    gaps = [float(r[f'gap_{ref}_bps']) for r in group]
                    if not np.allclose(diff, gaps, rtol=0, atol=1e-10):
                        raise ValueError('paired gaps disagree with matching policy seed means')
                    mean, lo, hi = interval(diff)
                    out.update({f'{ref}_mean': mean, f'{ref}_ci_low': lo, f'{ref}_ci_high': hi})
            result.append(out)
    return result


def check_expected(rows, expected):
    with Path(expected).open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    if len(rows) != len(saved):
        raise ValueError('support table row count mismatch')
    for row, old in zip(rows, saved):
        if (row['market'], row['policy']) != (old['market'], old['policy']):
            raise ValueError('support table order mismatch')
        for key, value in row.items():
            if key not in ('market', 'policy') and not np.isclose(value, float(old[key]), rtol=0, atol=1e-10):
                raise ValueError(f'saved numerical result mismatch: {row["market"]}/{row["policy"]}/{key}')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, help='new directory outside the repository')
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error('specify --check or --output')
    source = ROOT / 'release/evidence/revision_v20/audit/economic_by_seed.csv'
    rows = generate(source)
    check_expected(rows, ROOT / 'release/evidence/benchmark_supplement.csv')
    if args.output:
        dest = args.output.resolve()
        if dest == ROOT or ROOT in dest.parents or dest.exists():
            parser.error('output must be a new directory outside the repository')
        dest.mkdir(parents=True)
        fields = list(rows[0])
        with (dest / 'benchmark_supplement.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        (dest / 'verification.json').write_text(json.dumps({
            'level': 'B', 'paired_comparisons': 48, 'ordinary_levels': 30,
            'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'procedure': '10000 percentile resamples of 10 paired seed differences; pointwise, no multiplicity adjustment',
        }, indent=2) + '\n')
    print('PASS: 48 paired intervals and 30 ordinary-shortfall levels match saved unrounded results (level B).')


if __name__ == '__main__':
    main()
