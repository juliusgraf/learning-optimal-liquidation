"""Build a frozen historical dataset from true bid/ask quote events.

The publication path uses Alpaca's historical stock-quotes endpoint.  The
processed artifact contains the latest valid quote midpoint available at or
before each one-minute decision timestamp; it never uses a future quote and it
is kept in raw provider price units.  A separate gzip-compressed JSONL archive
preserves the source quote events used for every session.

Credentials are read only from environment variables and are never accepted as
command-line values.  The default ``sip`` feed represents consolidated US
quotes (NBBO); ``iex`` remains available for diagnostics but is explicitly
tagged as a single-venue feed in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import ssl
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import numpy as np
import pandas as pd

from lmm.data.historical_artifact import (
    DEFAULT_TICKERS,
    DEFAULT_TIMEZONE,
    NoSessionDataError,
    expected_grid,
    parse_session,
    sha256_file,
    validate_split_ranges,
    write_csv,
    write_metadata,
)

__all__ = [
    "ALPACA_QUOTES_URL",
    "QuoteDataError",
    "download_alpaca_quotes",
    "aggregate_quote_midpoints",
    "regenerate_range",
    "build_parser",
    "main",
]

logger = logging.getLogger(__name__)

ALPACA_QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes"
ALPACA_DOCS_URL = "https://docs.alpaca.markets/us/reference/stockquotes-1"
PRICE_TYPE = "quote_midpoint"
AGGREGATION_RULE = "latest valid quote at or before each decision timestamp"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class QuoteDataError(RuntimeError):
    """A quote response or quote-derived session violates the data contract."""


def _request_json(url: str, headers: Mapping[str, str]) -> dict[str, Any]:
    request = Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen(
            request,
            timeout=60,
            context=TLS_CONTEXT,
        ) as response:  # noqa: S310 - fixed HTTPS endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise QuoteDataError(
            f"Alpaca quote request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise QuoteDataError(f"Alpaca quote request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise QuoteDataError("Alpaca quote response must be a JSON object")
    return payload


def _credentials(
    api_key_id: str | None,
    api_secret_key: str | None,
    *,
    key_env: str,
    secret_env: str,
) -> tuple[str, str]:
    key = api_key_id or os.environ.get(key_env, "")
    secret = api_secret_key or os.environ.get(secret_env, "")
    if not key or not secret:
        raise QuoteDataError(
            f"Alpaca credentials are required; set {key_env} and {secret_env}. "
            "Do not place credentials in config files or shell history."
        )
    return key, secret


def download_alpaca_quotes(
    tickers: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    *,
    feed: str = "sip",
    limit: int = 10_000,
    api_key_id: str | None = None,
    api_secret_key: str | None = None,
    key_env: str = "APCA_API_KEY_ID",
    secret_env: str = "APCA_API_SECRET_KEY",
    endpoint: str = ALPACA_QUOTES_URL,
    requester: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Download every quote event in ``[start_ts, end_ts]`` with pagination.

    The multi-symbol endpoint is used so one HTTP stream covers a whole session.
    The returned mapping retains provider fields verbatim and adds no derived
    prices.  Tests inject ``requester``; production uses the fixed HTTPS endpoint.
    """

    symbols = [str(ticker).upper() for ticker in tickers]
    if not symbols:
        raise ValueError("at least one ticker is required")
    if feed not in ("sip", "iex"):
        raise ValueError("feed must be 'sip' or 'iex'")
    if not 1 <= int(limit) <= 10_000:
        raise ValueError("limit must be in [1, 10000]")
    key, secret = _credentials(
        api_key_id,
        api_secret_key,
        key_env=key_env,
        secret_env=secret_env,
    )
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    request_page = requester or _request_json
    events: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
    page_token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        query: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "start": pd.Timestamp(start_ts).tz_convert("UTC").isoformat(),
            "end": pd.Timestamp(end_ts).tz_convert("UTC").isoformat(),
            "limit": int(limit),
            "feed": feed,
            "sort": "asc",
        }
        if page_token:
            query["page_token"] = page_token
        payload = request_page(f"{endpoint}?{urlencode(query)}", headers)
        quote_map = payload.get("quotes")
        if not isinstance(quote_map, dict):
            raise QuoteDataError("Alpaca response is missing the per-symbol 'quotes' object")
        for symbol, rows in quote_map.items():
            normalized_symbol = str(symbol).upper()
            if normalized_symbol not in events:
                continue
            if not isinstance(rows, list):
                raise QuoteDataError(f"quotes[{normalized_symbol!r}] must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    raise QuoteDataError(f"quote row for {normalized_symbol} must be an object")
                events[normalized_symbol].append(dict(row))
        token = payload.get("next_page_token")
        if token in (None, ""):
            break
        page_token = str(token)
        if page_token in seen_tokens:
            raise QuoteDataError("Alpaca pagination repeated a page token")
        seen_tokens.add(page_token)

    if not any(events.values()):
        raise NoSessionDataError(
            f"Alpaca returned no {feed} quotes for {symbols} on {start_ts.date()}"
        )
    missing = [symbol for symbol, rows in events.items() if not rows]
    if missing:
        raise QuoteDataError(
            f"Alpaca returned no quotes for ticker(s) {missing} on {start_ts.date()}"
        )
    return events


def _provider_value(row: Mapping[str, Any], compact: str, verbose: str) -> Any:
    return row[compact] if compact in row else row.get(verbose)


def _valid_quote_frame(
    rows: Sequence[Mapping[str, Any]], symbol: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    parsed = pd.DataFrame(
        {
            "timestamp": [_provider_value(row, "t", "timestamp") for row in rows],
            "bid": [_provider_value(row, "bp", "bid_price") for row in rows],
            "ask": [_provider_value(row, "ap", "ask_price") for row in rows],
            "bid_size": [_provider_value(row, "bs", "bid_size") for row in rows],
            "ask_size": [_provider_value(row, "as", "ask_size") for row in rows],
        }
    )
    parsed["timestamp"] = pd.to_datetime(parsed["timestamp"], errors="coerce", utc=True)
    for column in ("bid", "ask", "bid_size", "ask_size"):
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")

    finite = np.isfinite(parsed[["bid", "ask", "bid_size", "ask_size"]]).all(axis=1)
    timestamp_ok = parsed["timestamp"].notna()
    positive_price = (parsed["bid"] > 0.0) & (parsed["ask"] > 0.0)
    positive_size = (parsed["bid_size"] > 0.0) & (parsed["ask_size"] > 0.0)
    not_crossed = parsed["bid"] <= parsed["ask"]
    valid = finite & timestamp_ok & positive_price & positive_size & not_crossed
    counts = {
        "n_received": int(len(parsed)),
        "n_invalid_timestamp_or_numeric": int((~(finite & timestamp_ok)).sum()),
        "n_nonpositive_price": int((finite & timestamp_ok & ~positive_price).sum()),
        "n_nonpositive_size": int((finite & timestamp_ok & positive_price & ~positive_size).sum()),
        "n_crossed": int((finite & timestamp_ok & positive_price & positive_size & ~not_crossed).sum()),
        "n_valid": int(valid.sum()),
    }
    clean = parsed.loc[valid].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if clean.empty:
        raise QuoteDataError(f"{symbol}: no valid positive, non-crossed quotes")
    return clean, counts


def aggregate_quote_midpoints(
    events: Mapping[str, Sequence[Mapping[str, Any]]],
    tickers: Sequence[str],
    grid: pd.DatetimeIndex,
    *,
    max_quote_age_seconds: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Map event quotes to decision times without interpolation or look-ahead."""

    if grid.tz is None:
        raise ValueError("decision grid must be timezone-aware")
    if max_quote_age_seconds <= 0.0:
        raise ValueError("max_quote_age_seconds must be positive")
    grid_utc = grid.tz_convert("UTC")
    # Force a common nanosecond unit: pandas 3 may preserve microsecond input
    # resolution in ``.asi8``, while parsed RFC3339 quote timestamps can carry
    # nanoseconds.  Comparing the raw integers without this conversion would
    # understate quote age by a factor of 1,000.
    grid_ns = (
        grid_utc.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    out = pd.DataFrame(index=grid)
    diagnostics: dict[str, dict[str, float | int]] = {}
    for ticker in tickers:
        symbol = str(ticker).upper()
        if symbol not in events:
            raise QuoteDataError(f"quote response is missing ticker {symbol}")
        clean, counts = _valid_quote_frame(events[symbol], symbol)
        event_ns = (
            clean["timestamp"]
            .dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        selected = np.searchsorted(event_ns, grid_ns, side="right") - 1
        if np.any(selected < 0):
            first = grid[np.flatnonzero(selected < 0)[0]]
            raise QuoteDataError(
                f"{symbol}: no quote at or before first required timestamp {first.isoformat()}"
            )
        chosen = clean.iloc[selected].reset_index(drop=True)
        age_seconds = (grid_ns - event_ns[selected]) / 1e9
        stale = age_seconds > float(max_quote_age_seconds)
        if np.any(stale):
            first_idx = int(np.flatnonzero(stale)[0])
            raise QuoteDataError(
                f"{symbol}: latest quote at {grid[first_idx].isoformat()} is "
                f"{age_seconds[first_idx]:.3f}s old, above the "
                f"{max_quote_age_seconds:g}s limit"
            )
        midpoint = 0.5 * (chosen["bid"].to_numpy() + chosen["ask"].to_numpy())
        spread = chosen["ask"].to_numpy() - chosen["bid"].to_numpy()
        spread_bps = 10_000.0 * spread / midpoint
        out[symbol] = midpoint
        diagnostics[symbol] = {
            **counts,
            "max_selected_quote_age_seconds": float(np.max(age_seconds)),
            "median_selected_quote_age_seconds": float(np.median(age_seconds)),
            "median_spread": float(np.median(spread)),
            "median_spread_bps": float(np.median(spread_bps)),
            "p95_spread_bps": float(np.quantile(spread_bps, 0.95)),
            "median_bid_size_provider_units": float(np.median(chosen["bid_size"])),
            "median_ask_size_provider_units": float(np.median(chosen["ask_size"])),
        }
    out.index.name = "Datetime"
    return out, diagnostics


def _event_cache_path(
    cache_dir: str | Path,
    session_date: str,
    feed: str,
    tickers: Sequence[str],
) -> Path:
    symbols = "-".join(str(ticker).upper() for ticker in tickers)
    return Path(cache_dir) / f"{session_date}_{feed}_{symbols}.jsonl.gz"


def _write_event_archive(
    events: Mapping[str, Sequence[Mapping[str, Any]]], path: str | Path
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"symbol": symbol, **dict(row)}
        for symbol in sorted(events)
        for row in events[symbol]
    ]
    rows.sort(key=lambda row: (str(row.get("symbol", "")), str(row.get("t", row.get("timestamp", "")))))
    with gzip.open(destination, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    return destination


def _read_event_archive(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuoteDataError(f"invalid quote cache {path} line {line_no}") from exc
            symbol = str(row.pop("symbol", "")).upper()
            if not symbol:
                raise QuoteDataError(f"quote cache {path} line {line_no} lacks symbol")
            events.setdefault(symbol, []).append(row)
    return events


def quote_size_unit(feed: str, start_date: str, end_date: str) -> str:
    """Alpaca CTA/UTP size display changed on 2025-11-03; never multiply sizes.

    Source: https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change
    The dated SIP change does not establish the units of another feed.
    """
    if feed != 'sip':
        return 'provider-native units (feed-specific; not converted)'
    if end_date < '2025-11-03':
        return 'round lots (provider bs/as fields)'
    if start_date >= '2025-11-03':
        return 'shares (provider bs/as fields)'
    return 'mixed: round lots before 2025-11-03; shares from 2025-11-03'


def _quality_summary(
    records: Sequence[Mapping[str, Any]], tickers: Sequence[str]
) -> dict[str, dict[str, float | None]]:
    fields = (
        "median_spread_bps",
        "p95_spread_bps",
        "median_bid_size_provider_units",
        "median_ask_size_provider_units",
        "max_selected_quote_age_seconds",
    )
    summary: dict[str, dict[str, float | None]] = {}
    mixed_units = len({record.get('quote_size_unit') for record in records}) > 1
    for ticker in tickers:
        symbol = str(ticker).upper()
        summary[symbol] = {
            field: None if mixed_units and 'size_provider_units' in field else float(
                statistics.median(
                    float(record["quote_quality"][symbol][field]) for record in records
                )
            )
            for field in fields
        }
    return summary


def regenerate_range(
    *,
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    train_date_range: Sequence[str],
    validation_date_range: Sequence[str],
    test_date_range: Sequence[str],
    out: str | Path,
    dataset_id: str,
    interval: str = "1m",
    session_start: str = "13:30",
    session_end: str = "16:00",
    clob_minutes: int = 120,
    auction_minutes: int = 30,
    timezone_name: str = DEFAULT_TIMEZONE,
    feed: str = "sip",
    lookback_minutes: int = 5,
    max_quote_age_seconds: float = 60.0,
    cache_dir: str | Path | None = ".cache/alpaca_quotes",
    raw_dir: str | Path | None = None,
    allow_skipped_sessions: bool = False,
    api_key_id: str | None = None,
    api_secret_key: str | None = None,
    key_env: str = "APCA_API_KEY_ID",
    secret_env: str = "APCA_API_SECRET_KEY",
    downloader: Callable[..., dict[str, list[dict[str, Any]]]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a multi-session raw-price midpoint CSV and provenance sidecar."""

    symbols = [str(ticker).upper() for ticker in tickers]
    if not dataset_id:
        raise ValueError("dataset_id is required for a publication artifact")
    if interval != "1m":
        raise ValueError("the experiment contract currently requires interval='1m'")
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be positive")
    split_ranges = validate_split_ranges(
        train_date_range,
        validation_date_range,
        test_date_range,
        dataset_start=start_date,
        dataset_end=end_date,
    )
    tau_op = int(clob_minutes)
    tau_cl = int(clob_minutes + auction_minutes)
    candidate_dates = [
        stamp.date().isoformat()
        for stamp in pd.bdate_range(start=start_date, end=end_date)
    ]
    if not candidate_dates:
        raise ValueError("dataset range contains no weekday sessions")

    fetch = downloader or download_alpaca_quotes
    sessions: list[pd.DataFrame] = []
    session_records: list[dict[str, Any]] = []
    skipped_sessions: list[dict[str, str]] = []
    for session_date in candidate_dates:
        start_ts, end_ts = parse_session(
            session_date, session_start, session_end, timezone_name
        )
        grid = expected_grid(start_ts, end_ts, interval)
        if len(grid) != tau_cl + 1:
            raise ValueError(
                f"session window {session_start}-{session_end} yields {len(grid)} rows; "
                f"expected clob+auction+1={tau_cl + 1}"
            )
        request_start = start_ts - pd.Timedelta(minutes=lookback_minutes)
        request_end = end_ts + pd.Timedelta(minutes=1)
        cache_path = (
            _event_cache_path(cache_dir, session_date, feed, symbols)
            if cache_dir is not None
            else None
        )
        try:
            if cache_path is not None and cache_path.exists():
                events = _read_event_archive(cache_path)
                cache_hit = True
            else:
                events = fetch(
                    symbols,
                    request_start,
                    request_end,
                    feed=feed,
                    api_key_id=api_key_id,
                    api_secret_key=api_secret_key,
                    key_env=key_env,
                    secret_env=secret_env,
                )
                cache_hit = False
                if cache_path is not None:
                    _write_event_archive(events, cache_path)
        except NoSessionDataError as exc:
            if not allow_skipped_sessions:
                raise
            skipped_sessions.append({"date": session_date, "reason": str(exc)})
            continue

        midpoints, quote_quality = aggregate_quote_midpoints(
            events,
            symbols,
            grid,
            max_quote_age_seconds=max_quote_age_seconds,
        )
        sessions.append(midpoints)
        raw_record: dict[str, Any] = {}
        if raw_dir is not None:
            raw_path = _write_event_archive(
                events, Path(raw_dir) / f"{session_date}_{feed}_quotes.jsonl.gz"
            )
            raw_record = {
                "raw_quote_path": str(raw_path),
                "raw_quote_sha256": sha256_file(raw_path),
            }
        session_records.append(
            {
                "date": session_date,
                "n_rows": int(len(midpoints)),
                "cache_hit": cache_hit,
                "quote_quality": quote_quality,
                "quote_size_unit": quote_size_unit(feed, session_date, session_date),
                **raw_record,
            }
        )

    if not sessions:
        raise ValueError("no complete quote sessions were downloaded")
    included_dates = [record["date"] for record in session_records]
    split_session_dates: dict[str, list[str]] = {}
    for split, (lower, upper) in split_ranges.items():
        selected = [date for date in included_dates if lower <= date <= upper]
        if not selected:
            raise ValueError(f"{split} split [{lower}, {upper}] contains no sessions")
        split_session_dates[split] = selected

    combined = pd.concat(sessions).sort_index()
    if combined.index.duplicated().any():
        duplicate = combined.index[combined.index.duplicated()][0]
        raise ValueError(f"duplicate timestamp across sessions: {duplicate}")
    out_path = write_csv(combined, out)
    metadata: dict[str, Any] = {
        "artifact_schema_version": 4,
        "dataset_id": dataset_id,
        "tickers": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "candidate_weekday_dates": candidate_dates,
        "included_session_dates": included_dates,
        "skipped_sessions": skipped_sessions,
        "n_sessions": len(session_records),
        "n_rows": int(len(combined)),
        "n_rows_per_session": tau_cl + 1,
        "sessions": session_records,
        "split_ranges": split_ranges,
        "split_session_dates": split_session_dates,
        "interval": interval,
        "time_unit": "minutes",
        "session_start": session_start,
        "session_end": session_end,
        "timezone": timezone_name,
        "clob_minutes": tau_op,
        "auction_minutes": int(auction_minutes),
        "tau_op": tau_op,
        "tau_cl": tau_cl,
        "source": "alpaca_market_data_api",
        "source_endpoint": ALPACA_QUOTES_URL,
        "source_documentation": ALPACA_DOCS_URL,
        "price_type": PRICE_TYPE,
        "price_field": "(bid_price + ask_price) / 2",
        "quote_feed": feed,
        "quote_consolidation": "consolidated NBBO" if feed == "sip" else "single-venue IEX",
        "quote_size_unit": quote_size_unit(feed, start_date, end_date),
        "quote_size_unit_source": "https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change",
        "normalize": "none",
        "model_coordinate_transform": "each selected session is rebased to grid.S0 by the environment loader",
        "fill": "error",
        "aggregation": AGGREGATION_RULE,
        "lookback_minutes": int(lookback_minutes),
        "max_quote_age_seconds": float(max_quote_age_seconds),
        "quote_quality_summary": _quality_summary(session_records, symbols),
        "row_to_decision_time": "within each session, row r is the latest valid quote midpoint "
        "known at decision time t=r; row tau_op is frozen during the call",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": sha256_file(out_path),
    }
    sidecar = write_metadata(metadata, out_path)
    logger.info(
        "wrote %s (%d quote sessions, %d rows) and %s",
        out_path,
        len(session_records),
        len(combined),
        sidecar,
    )
    return out_path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-load-midquotes",
        description="Download true historical bid/ask quotes and build a raw-price "
        "one-minute midpoint artifact with a provenance sidecar.",
    )
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--train-range", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--validation-range", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--test-range", nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--feed", choices=("sip", "iex"), default="sip")
    parser.add_argument("--session-start", default="13:30")
    parser.add_argument("--session-end", default="16:00")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--clob-minutes", type=int, default=120)
    parser.add_argument("--auction-minutes", type=int, default=30)
    parser.add_argument("--lookback-minutes", type=int, default=5)
    parser.add_argument("--max-quote-age-seconds", type=float, default=60.0)
    parser.add_argument("--allow-skipped-sessions", action="store_true")
    parser.add_argument("--key-env", default="APCA_API_KEY_ID")
    parser.add_argument("--secret-env", default="APCA_API_SECRET_KEY")
    parser.add_argument("--cache-dir", default=".cache/alpaca_quotes")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--out", default="data/historical_sp500_midquotes_1m.csv")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (5 <= len(args.tickers) <= 10):
        parser.error(f"--tickers must have 5-10 symbols, got {len(args.tickers)}")
    raw_dir = args.raw_dir or f"data/raw/alpaca/{args.dataset_id}"
    out_path, metadata = regenerate_range(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        train_date_range=args.train_range,
        validation_date_range=args.validation_range,
        test_date_range=args.test_range,
        out=args.out,
        dataset_id=args.dataset_id,
        session_start=args.session_start,
        session_end=args.session_end,
        clob_minutes=args.clob_minutes,
        auction_minutes=args.auction_minutes,
        timezone_name=args.timezone,
        feed=args.feed,
        lookback_minutes=args.lookback_minutes,
        max_quote_age_seconds=args.max_quote_age_seconds,
        cache_dir=args.cache_dir,
        raw_dir=raw_dir,
        allow_skipped_sessions=args.allow_skipped_sessions,
        key_env=args.key_env,
        secret_env=args.secret_env,
    )
    print(
        f"wrote {out_path} ({metadata['n_sessions']} sessions, "
        f"{metadata['n_rows']} raw-price midpoints) + {out_path.name}.meta.json"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
