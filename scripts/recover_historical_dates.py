"""Check recovered date assignments using recorded seeds and original draw order.

No prices, environment episodes, policies, training or checkpoints are loaded.
These are recovered preparation-time metadata, not contemporaneous date logs.
"""
import json
from pathlib import Path
import numpy as np
from lmm.config import ClobFlowParams, GridParams
from lmm.market.generator import sample_episode_realization

ROOT = Path(__file__).resolve().parents[1]


def check():
    campaign = json.loads((ROOT/'release/evidence/campaign.json').read_text())
    dates = json.loads((ROOT/'release/evidence/historical_dates.json').read_text())
    pools = json.loads((ROOT/'release/evidence/data.json').read_text())['metadata']['split_session_dates']
    cache = {}; count = 0
    for run in campaign['runs']:
        if run['setting'] != 'historical_sp500_midquotes':
            continue
        cfg = run['config']; flow = ClobFlowParams(**cfg['clob_flow']); grid = GridParams(**cfg['grid'])
        for phase, binding in run['recovered_date_mapping'].items():
            split = 'train' if phase in ('training','normalizer') else phase
            symbols = cfg['midprice']['historical']['symbols'] if split == 'train' else [run['symbol']]
            pool = [(s,d) for s in symbols for d in pools[split]]
            seeds = run['seeds'][phase]
            rows = dates['assignments'][binding['key']][:binding['count']]
            if len(rows) != len(seeds) or [r[0] for r in rows] != seeds:
                raise ValueError('recorded seeds and date rows disagree')
            for seed, symbol, date in rows:
                key = (seed, json.dumps(cfg['clob_flow'],sort_keys=True), json.dumps(cfg['grid'],sort_keys=True), tuple(pool))
                if key not in cache:
                    rng = np.random.default_rng(seed)
                    sample_episode_realization(rng, flow, grid)
                    cache[key] = pool[int(rng.integers(0,len(pool)))]
                if cache[key] != (symbol,date):
                    raise ValueError('date mapping differs from original draw order')
                count += 1
    print(f'PASS: {count} recorded seed/date assignments; no market data or policy execution.')


if __name__ == '__main__':
    check()
