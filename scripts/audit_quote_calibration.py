"""Training-only spread/depth-scale audit, correcting the frozen SIP unit label."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from lmm.config import load_config


def main():
    source = Path('data/historical_sp500_midquotes_1m.csv.meta.json')
    metadata = json.loads(source.read_text())
    assert metadata['quote_feed'] == 'sip'
    train_dates = metadata['split_session_dates']['train']
    assert min(train_dates) >= '2025-11-03'
    rows = []
    for session in metadata['sessions']:
        if session['date'] not in train_dates: continue
        for ticker, quality in session['quote_quality'].items():
            # Schema-3 keys were mislabeled. Values are unchanged provider
            # fields; the dated CTA/UTP announcement establishes shares.
            rows.append(dict(date=session['date'], ticker=ticker,
                             spread_bps=quality['median_spread_bps'],
                             bid_size_shares=quality['median_bid_size_round_lots'],
                             ask_size_shares=quality['median_ask_size_round_lots']))
    frame = pd.DataFrame(rows)
    output = Path('docs/verification_v16')
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'training_quote_calibration.csv', index=False)
    summary = frame.groupby('ticker').median(numeric_only=True)
    summary.to_csv(output/'training_quote_calibration_summary.csv')
    cfg = load_config('configs/base.yaml', 'configs/synthetic_rough_heston.yaml')
    model_median = cfg.clob_flow.V_inf*beta.ppf(.5, cfg.clob_flow.beta_a, cfg.clob_flow.beta_b)
    empirical_median = float(np.median(frame[['bid_size_shares', 'ask_size_shares']].to_numpy()))
    audit = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 used_dates=train_dates, corrected_size_unit='shares; numbers are not multiplied',
                 unit_source='https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change',
                 unit_label_erratum='Frozen schema-3 names say round_lots; August 2026 SIP values are shares.',
                 pooled_median_daily_selected_top_size_shares=empirical_median,
                 model_top_depth_median=float(model_median),
                 illustrative_shares_per_model_inventory_unit=empirical_median/model_median,
                 limitation='One top-depth scale anchor, not a fit of full depth, market-order flow or auction supply.')
    (output/'quote_calibration_provenance.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
