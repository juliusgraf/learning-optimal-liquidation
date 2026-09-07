# Historical mid-price data

## Frozen publication input: true quote midpoints

The current true-midquote historical path is `historical_sp500_midquotes_1m.csv`, with
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

Market data are not included in the current source tree. Historical experiments
require an authorized local CSV, provenance sidecar and matching raw archives
under `raw/alpaca/sp500_midquotes_sip_2026-08_v1/`. These paths are ignored by Git.
Synthetic experiments and the offline tests need no market-data credentials.
Historical integration tests skip when the private files are absent; existing
but invalid data still fail validation.

To reproduce the original runs exactly, restore the original verified artifacts
from your authorized private archive. To conduct a new historical experiment,
obtain data using the command below and validate the resulting artifact. New
retrievals are not guaranteed to be byte-identical to the original dataset.

The chronological pools are fixed in both the sidecar and
`configs/historical_sp500_midquotes.yaml`:

- training: 2026-08-03 through 2026-08-14 (10 sessions);
- validation: 2026-08-17 through 2026-08-21 (5 sessions);
- test: 2026-08-24 through 2026-08-28 (5 sessions).

The processed CSV stays in raw provider price units. Only after a session is
selected does the environment rebase its path to `S0=100`; this is a modeling
coordinate transform, not source-data normalization. Compressed raw quote
events are archived separately under `data/raw/alpaca/<dataset-id>/`.

The loader verifies the CSV and raw-archive SHA-256 digests, source, price type, feed,
normalization declaration, dataset ID, tickers, timezone, missing-data rule,
split ranges, and nonempty split membership whenever a historical environment
is constructed. Every run copies the verified sidecar to
`historical_data_manifest.json`.

### Obtain a local quote artifact

To fetch a new candidate, set credentials in the environment; do not
put them in YAML or command-line arguments. The account must have access to the
requested historical feed.

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
  --json-out results/data_checks/midquote_simulator_gate.json
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
synthetic.
# Quote-size label erratum (September 2026)

The frozen August 2026 SIP sidecar's `*_size_round_lots` names and
`quote_size_unit` label are incorrect. The stored numbers are **shares**:
Alpaca changed CTA/UTP quote-size display on November 3, 2025. Do not multiply
these values by a round-lot size. The original CSV, sidecar and raw archives
are preserved for provenance; midpoint calculations and all learning results
are unaffected. See [Alpaca's dated announcement](https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change)
and the archived training-only audit, whose recovery path is recorded in
[the cleanup manifest](../docs/cleanup_manifest.json).
Future regeneration uses schema 4, neutral provider-unit field names and
date/feed-specific units; it omits aggregate size medians across mixed units.
