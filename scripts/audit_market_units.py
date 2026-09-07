"""Training-date price/depth units and minute-return scale; no fitting to rankings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from lmm.config import load_config
from lmm.experiments.calibration import calibration_scales
from lmm.market.midprice import build_midprice


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = Path('data/historical_sp500_midquotes_1m.csv')
    sidecar = Path(str(data)+'.meta.json')
    meta = json.loads(sidecar.read_text())
    dates = meta['split_session_dates']['train']
    cfg = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml')
    f = pd.read_csv(data)
    t = pd.to_datetime(f.pop('Datetime'))
    symbols = list(f)
    f['date'], f['minute'] = t.dt.strftime('%Y-%m-%d'), t.dt.hour*60+t.dt.minute
    f = f[f.date.isin(dates) & (f.minute <= 15*60+30)]
    annual_minutes = cfg.midprice.rough_heston.time_units_per_year
    rows = []
    for symbol in symbols:
        for date, group in f.groupby('date'):
            prices = group[symbol].to_numpy()
            returns = np.diff(np.log(prices))
            rows.append(dict(symbol=symbol, date=date, initial_raw_price=prices[0],
                             annual_realized_vol=np.sqrt(np.sum(returns**2)*annual_minutes/len(returns)),
                             horizon_return_bps=1e4*(prices[-1]/prices[0]-1)))
    market = build_midprice(cfg)
    seeds = np.random.default_rng(174083).integers(0, 2**31-1, 256).tolist()
    for i, seed in enumerate(seeds):
        market.reset(np.random.default_rng(seed))
        prices = np.array([cfg.grid.S0]+[market.advance_to(t) for t in range(1,cfg.grid.tau_op+1)])
        returns = np.diff(np.log(prices))
        rows.append(dict(symbol='synthetic', date=str(i), initial_raw_price=cfg.grid.S0,
                         annual_realized_vol=np.sqrt(np.sum(returns**2)*annual_minutes/len(returns)),
                         horizon_return_bps=1e4*(prices[-1]/prices[0]-1)))
    pd.DataFrame(rows).to_csv(args.output/'price_paths.csv', index=False)
    medians = pd.DataFrame(rows).groupby('symbol').median(numeric_only=True)
    medians.to_csv(args.output/'price_summary.csv')
    quote_sizes = [q[key] for session in meta['sessions'] if session['date'] in dates
                   for q in session['quote_quality'].values()
                   for key in ('median_bid_size_round_lots', 'median_ask_size_round_lots')]
    # The dated SIP erratum establishes that the frozen numeric sizes are shares.
    depth_median = cfg.clob_flow.V_inf*beta.ppf(.5,cfg.clob_flow.beta_a,cfg.clob_flow.beta_b)
    shares_per_unit = float(np.median(quote_sizes)/depth_median)
    physical = []
    for symbol in symbols:
        price_factor = medians.loc[symbol, 'initial_raw_price']/cfg.grid.S0
        physical.append(dict(symbol=symbol, raw_price_per_model_price_unit=price_factor,
            illustrative_shares_per_inventory_unit=shares_per_unit,
            dollars_per_model_objective_unit=price_factor*shares_per_unit,
            model_tick_in_raw_dollars=cfg.grid.alpha*price_factor,
            beta_in_shares_per_dollar=cfg.actions.beta*shares_per_unit/price_factor,
            lambda_in_dollars_per_share_squared=cfg.reward.lambda_inv*price_factor/shares_per_unit,
            initial_notional_dollars=cfg.grid.S0*cfg.grid.I0*price_factor*shares_per_unit))
    pd.DataFrame(physical).to_csv(args.output/'illustrative_physical_units.csv', index=False)
    audit = dict(purpose=__doc__, historical_dates=dates, synthetic_seeds=seeds,
        scales=calibration_scales(cfg),
        sources={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (data,sidecar)},
        limitation='Illustrative dimensional conversion from selected top-quote depth; no full-tape or auction fit. Ten historical dates cannot identify tail risk.',
        quote_unit_source='https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change')
    (args.output/'protocol.json').write_text(json.dumps(audit, indent=2))
    print(medians.round(5).to_string())


if __name__ == '__main__': main()
