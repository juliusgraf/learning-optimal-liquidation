# `data/` — historical mid-price datasets

This directory holds **loader-regenerated** historical mid-price CSVs. The
historical experiments do **not** read from here by default.

## What the historical experiments actually use

The committed **frozen experimental input** is [`../legacy/data.csv`](../legacy/data.csv)
— real S&P 500 1-minute mid prices, Dec 31 2025, normalized to 100 at session
start (ruling D13, `audit/AUDIT.md` A.12). `configs/historical_sp500.yaml` points
`midprice.historical.csv_path` at that file, and it is the source of truth for
every historical run. Do not edit or delete it.

## Why a frozen CSV (and not a live download)

yfinance only serves **1-minute** bars for a **~30-day lookback**. The paper's
Dec 2025 session is already outside that window, so it cannot be re-fetched. The
loader therefore exists to *regenerate* an equivalent dataset for **new** dates,
not to reproduce the frozen file byte-for-byte.

## Regenerating a dataset

```bash
python -m lmm.data.load_yfinance_data \
    --tickers CAT PG GOOGL JPM MSFT \
    --date 2025-12-31 \
    --interval 1m \
    --session-start 14:30 --session-end 17:00 \
    --clob-minutes 120 --auction-minutes 30 \
    --normalize first=100 \
    --out data/mid_prices_2025-12-31.csv
```

(equivalently the `lmm-load-data` console script). This writes the CSV plus a
JSON sidecar `data/mid_prices_<date>.csv.meta.json` recording tickers, date,
interval, source, mid proxy, normalization, fill policy, download timestamp, and
yfinance version.

To actually run experiments on a regenerated file, set
`midprice.historical.csv_path` and `midprice.historical.date` in
`configs/historical_sp500.yaml` (the **date is a config value, never a hard-coded
constant**) and pick `symbols` present in the new CSV.

## Conventions

- **Mid proxy.** yfinance 1m bars are not true midpoints; we use the bar
  **close** as the mid proxy (recorded in the sidecar).
- **Timezone.** `--session-start`/`--session-end` are wall-clock in `--timezone`
  (default `America/New_York`, i.e. Eastern — EST on Dec 31). The loader writes
  correct tz-aware Eastern `Datetime` values. (The legacy `legacy/data.csv`
  labels its Eastern timestamps `+00:00` — a known quirk, audit A.12/D13 — which
  the loader does **not** reproduce.)
- **Row → decision time (ruling D13).** Row `r` is the mid at decision time
  `t = r`. The CLOB phase consumes rows `0..tau_op-1` (`tau_op = clob_minutes`);
  the auction mid is frozen at row `tau_op-1`; clearing is at
  `tau_cl = clob_minutes + auction_minutes`. The CSV spans the full decision grid
  (`tau_cl + 1` rows); the env reads only `n_rows = tau_op` via config.
- **Completeness.** The session grid is validated bar-by-bar; missing minutes
  fail loudly (`--fill error`, default) or are forward-filled (`--fill ffill`,
  recorded in the sidecar).

Regenerated CSVs and the `.cache/` download cache are git-ignored; this README is
committed.
