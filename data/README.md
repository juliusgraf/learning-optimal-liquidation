# Historical mid-price data

The active historical input is `historical_sp500_1m.csv`, accompanied by
`historical_sp500_1m.csv.meta.json`. It contains 20 one-minute sessions for the
five manuscript assets: CAT, PG, GOOGL, JPM, and MSFT.

The chronological pools are fixed in both the sidecar and
`configs/historical_sp500.yaml`:

- training: 2026-08-03 through 2026-08-14 (10 sessions);
- validation: 2026-08-17 through 2026-08-21 (5 sessions);
- test: 2026-08-24 through 2026-08-28 (5 sessions).

The loader verifies the CSV SHA-256 digest, dataset ID, tickers, timezone,
missing-data rule, split ranges, and nonempty split membership whenever a
historical environment is constructed. Every run also copies the verified
sidecar to `historical_data_manifest.json`.

## Data interpretation

- Source: yfinance one-minute bar closes. These are a documented mid-price
  proxy, not exchange quote midpoints.
- Window: 13:30–16:00 America/New_York, producing 151 rows per session.
- Physical units: one simulator unit is one minute.
- CLOB input: rows 0 through 120 supply the path through
  `tau_op = 120`. A realized noninteger decision time uses the latest
  observation at or before that time.
- Auction input: the row-120 mid is frozen for the 30-minute call. Later source
  rows are retained for source-window provenance but are never revealed to the
  environment.
- Normalization: each ticker/session starts at 100.
- Missing bars: forward-filled within that session only. Every filled timestamp
  is enumerated in the sidecar; there is no cross-session fill.

Order flow, CLOB depth, auction proposals, clearing, and allocation remain
synthetic, as required by the manuscript.

## Rebuilding a recent multi-session dataset

Yahoo retains one-minute data only for a short period. Run the range builder
while all requested sessions remain available, then freeze the resulting CSV
and sidecar together:

```bash
python3 -m lmm.data.load_yfinance_data \
  --tickers CAT PG GOOGL JPM MSFT \
  --start-date 2026-08-03 --end-date 2026-08-28 \
  --train-range 2026-08-03 2026-08-14 \
  --validation-range 2026-08-17 2026-08-21 \
  --test-range 2026-08-24 2026-08-28 \
  --dataset-id sp500_1m_2026-08_v1 \
  --session-start 13:30 --session-end 16:00 \
  --fill ffill \
  --out data/historical_sp500_1m.csv
```

The downloader requests sessions separately, uses an isolated writable
yfinance cache, retries partial grouped-download failures by ticker, and fails
on an empty weekday unless `--allow-skipped-sessions` is explicitly supplied.
Even when skipping is enabled, generation fails if training, validation, or
test would be empty.

Single-session mode remains available for diagnostics via `--date YYYY-MM-DD`,
but it is not the configured experiment input.
