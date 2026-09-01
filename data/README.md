# Historical mid-price data

## Publication candidate: true quote midpoints

The revision-v3 historical path is `historical_sp500_midquotes_1m.csv`, with
sidecar `historical_sp500_midquotes_1m.csv.meta.json`. It is generated from
timestamped bid/ask quote events, not trade bars. The price at each decision
minute is

```text
midquote = (best bid + best ask) / 2
```

using the latest valid quote at or before that timestamp. The builder rejects
nonpositive, crossed, zero-size, or stale quotes; it never interpolates from a
future observation. The publication configuration requests Alpaca's `sip`
feed (consolidated US quotes). The `iex` feed is supported only as an explicitly
tagged single-venue diagnostic.

The chronological pools are fixed in both the sidecar and
`configs/historical_sp500_midquotes.yaml`:

- training: 2026-08-03 through 2026-08-14 (10 sessions);
- validation: 2026-08-17 through 2026-08-21 (5 sessions);
- test: 2026-08-24 through 2026-08-28 (5 sessions).

The processed CSV stays in raw provider price units. Only after a session is
selected does the environment rebase its path to `S0=100`; this is a modeling
coordinate transform, not source-data normalization. Compressed raw quote
events are archived separately under `data/raw/alpaca/<dataset-id>/`.

The loader verifies the CSV SHA-256 digest, source, price type, feed,
normalization declaration, dataset ID, tickers, timezone, missing-data rule,
split ranges, and nonempty split membership whenever a historical environment
is constructed. Every run copies the verified sidecar to
`historical_data_manifest.json`.

### Build the quote artifact

Set credentials in the environment; do not put them in YAML or command-line
arguments. The account must have access to the requested historical feed.

```bash
export APCA_API_KEY_ID='...'
export APCA_API_SECRET_KEY='...'

python3 -m lmm.data.load_midquote_data \
  --tickers CAT PG GOOGL JPM MSFT \
  --start-date 2026-08-03 --end-date 2026-08-28 \
  --train-range 2026-08-03 2026-08-14 \
  --validation-range 2026-08-17 2026-08-21 \
  --test-range 2026-08-24 2026-08-28 \
  --dataset-id sp500_midquotes_sip_2026-08_v1 \
  --feed sip \
  --session-start 13:30 --session-end 16:00 \
  --max-quote-age-seconds 60 \
  --out data/historical_sp500_midquotes_1m.csv
```

The command is cache-aware and paginates until all quote events have been
retrieved. It fails closed on an incomplete weekday by default and records raw
archive digests plus quote-age, spread, and displayed-size diagnostics.

Before any training, run the policy-free simulator gate on the new artifact:

```bash
python3 -m lmm.experiments.diagnose_simulator \
  --config configs/base.yaml \
  --config configs/historical_sp500_midquotes.yaml \
  --episodes 20 --assert-ready \
  --json-out results/diagnostics_v3/midquote_simulator_gate.json
```

## Data interpretation

- Source: historical best-bid/best-ask quote events; SIP is required for the
  publication candidate.
- Window: 13:30–16:00 America/New_York, producing 151 rows per session.
- Physical units: one simulator unit is one minute.
- CLOB input: rows 0 through 120 supply the path through `tau_op = 120`. A
  realized noninteger decision time uses the latest observation at or before
  that time.
- Auction input: the row-120 midquote is frozen for the 30-minute call. Later
  source rows are retained for source-window provenance but are not revealed
  to the environment.
- Source normalization: none. The environment performs an explicit per-session
  model rebase after split selection.
- Missing/stale quotes: error; there is no cross-session or future fill.

Order flow, CLOB depth, auction proposals, clearing, and allocation remain
synthetic, as required by the manuscript.

## Legacy Yahoo artifact (reproduction only)

`historical_sp500_1m.csv` and `configs/historical_sp500.yaml` retain the old
yfinance one-minute close proxy solely to reproduce revision-v2 results. They
must not be used for new historical claims. Its exact regeneration command is:

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

Yahoo retains one-minute data only briefly. The legacy builder fails on an
empty weekday unless `--allow-skipped-sessions` is explicit, and still fails
if any chronological split would be empty.
