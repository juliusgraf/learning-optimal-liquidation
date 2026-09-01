"""Build a provenance-tagged multi-session historical mid-price dataset.

The manuscript's historical setting uses five S&P 500 price paths as the
mid-price input while order flow and the auction remain simulated.  This module
downloads yfinance one-minute bar closes (a documented midpoint proxy),
regularizes each session independently, and writes one CSV containing disjoint
chronological train, validation, and test pools.

One simulator time unit is one minute.  A source session contains
``tau_cl + 1`` rows.  Rows ``0..tau_op`` supply the CLOB path through auction
open; the environment freezes the row-``tau_op`` mid throughout the call and
does not use later source bars.  Keeping the full source window makes the
session provenance explicit without leaking future bars into the policy.

Yahoo's one-minute retention is short, so ranges are downloaded one session at
a time and the resulting CSV plus JSON sidecar are the frozen local experiment
input.  The sidecar records the concrete dates that were included, any market
holidays that were skipped, missing-minute treatment, split membership, source
version, and a SHA-256 digest of the CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import date as Date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import pandas as pd

if TYPE_CHECKING:
    from lmm.config import HistoricalParams

__all__ = [
    "parse_session",
    "expected_grid",
    "validate_and_fill",
    "normalize",
    "write_csv",
    "write_metadata",
    "regenerate",
    "regenerate_range",
    "validate_split_ranges",
    "validate_historical_artifact",
    "build_parser",
    "main",
]

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ("CAT", "PG", "GOOGL", "JPM", "MSFT")  # the paper's five (data.py:1651)
DEFAULT_TIMEZONE = "America/New_York"
MID_PROXY = "yfinance 1m bar close (not a true midpoint)"


class NoSessionDataError(RuntimeError):
    """The provider returned no bars for an otherwise valid date request."""


def _interval_to_freq(interval: str) -> str:
    """Map a yfinance interval like ``1m``/``5m``/``1h`` to a pandas freq alias."""
    m = re.fullmatch(r"(\d+)(m|h)", interval)
    if not m:
        raise ValueError(f"unsupported --interval {interval!r}; use e.g. 1m, 5m, 1h")
    n, unit = int(m.group(1)), m.group(2)
    return f"{n}min" if unit == "m" else f"{n * 60}min"


def parse_session(
    date: str,
    session_start: str,
    session_end: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Localize the wall-clock session window to ``timezone_name`` (tz-aware).

    ``session_start``/``session_end`` are wall-clock (e.g. ``13:30``/``16:00``)
    in the configured timezone.
    """
    start = pd.Timestamp(f"{date} {session_start}", tz=timezone_name)
    end = pd.Timestamp(f"{date} {session_end}", tz=timezone_name)
    if end <= start:
        raise ValueError(f"session-end {session_end} must be after session-start {session_start}")
    return start, end


def expected_grid(start_ts: pd.Timestamp, end_ts: pd.Timestamp, interval: str) -> pd.DatetimeIndex:
    """Inclusive decision grid of bar timestamps (named ``Datetime``)."""
    grid = pd.date_range(start=start_ts, end=end_ts, freq=_interval_to_freq(interval))
    grid.name = "Datetime"
    return grid


def _extract_close(data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Pull the ``Close`` (mid proxy) per ticker from a yfinance download frame."""
    if isinstance(data.columns, pd.MultiIndex):
        columns: dict[str, pd.Series] = {}
        for ticker in tickers:
            if (ticker, "Close") in data.columns:
                columns[ticker] = data[(ticker, "Close")]
            elif ("Close", ticker) in data.columns:
                columns[ticker] = data[("Close", ticker)]
        close = pd.DataFrame(columns)
    else:  # single ticker -> flat columns
        close = (
            data[["Close"]].rename(columns={"Close": tickers[0]})
            if "Close" in data
            else pd.DataFrame(index=data.index)
        )
    close.index.name = "Datetime"
    return close.sort_index()


def _cache_key(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: str,
    tickers: Sequence[str],
) -> str:
    """Download-cache filename, keyed by the FULL session window.

    Must include both the start AND end time, not just the date: two intraday
    windows on the same day (e.g. ``13:30-16:00`` vs ``14:30-17:00``) otherwise
    collide, so a second run silently reuses the first window's bars and the
    missing edge is fabricated by the fill step (back-filled leading minutes show
    up as a flat run — the bug this guards against).
    """
    span = f"{start_ts.strftime('%Y%m%dT%H%M%z')}-{end_ts.strftime('%Y%m%dT%H%M%z')}"
    # Pickle avoids making pyarrow/fastparquet a hidden runtime dependency.
    return f"{span}_{interval}_{'-'.join(tickers)}.pkl"


def _download_yf(
    tickers: Sequence[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download 1-minute bars via yfinance; return a tidy frame.

    Returns a DataFrame indexed by a tz-aware ``Datetime`` index with one column
    per ticker (the bar close). yfinance is imported lazily so the rest of the
    pipeline (and the test-suite, which monkeypatches this seam) needs no network
    and no yfinance install.
    """
    tickers = list(tickers)
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_path = Path(cache_dir) / _cache_key(start_ts, end_ts, interval, tickers)
        if cache_path.exists():
            try:
                return pd.read_pickle(cache_path)
            except Exception as exc:  # pragma: no cover - corrupt/unsupported cache
                logger.warning("ignoring unreadable cache %s: %s", cache_path, exc)

    import yfinance as yf  # lazy: optional at import time, mocked in tests

    if cache_dir:
        # yfinance otherwise writes SQLite state below the user's home cache.
        # An isolated, explicitly writable location also avoids cross-process
        # cookie/ticker database locks during reproducibility runs.
        internal_cache = Path(cache_dir) / "_yfinance"
        internal_cache.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(internal_cache))

    end_excl = end_ts + pd.Timedelta(_interval_to_freq(interval))  # yfinance end is exclusive
    data = yf.download(
        tickers,
        start=start_ts,
        end=end_excl,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if data is None or len(data) == 0:
        raise NoSessionDataError(
            f"yfinance returned no {interval} bars for {tickers} on {start_ts.date()} "
            "(beyond the ~30-day intraday lookback? wrong session/date?)"
        )
    close = _extract_close(data, tickers)

    # A grouped request can occasionally return one all-NaN ticker while the
    # others succeed. Retry only those symbols sequentially; never silently
    # turn a provider failure into a filled flat path.
    missing = [
        ticker
        for ticker in tickers
        if ticker not in close or close[ticker].dropna().empty
    ]
    for ticker in missing:
        retry = yf.download(
            ticker,
            start=start_ts,
            end=end_excl,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        extracted = _extract_close(retry, [ticker])
        if ticker in extracted and not extracted[ticker].dropna().empty:
            close[ticker] = extracted[ticker]

    still_missing = [
        ticker
        for ticker in tickers
        if ticker not in close or close[ticker].dropna().empty
    ]
    if still_missing:
        raise RuntimeError(
            f"yfinance returned no usable {interval} bars for ticker(s) "
            f"{still_missing} on {start_ts.date()}"
        )
    close = close[list(tickers)]
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            close.to_pickle(cache_path)
        except Exception as exc:  # pragma: no cover - unusual filesystem failure
            logger.warning("could not write cache %s: %s", cache_path, exc)
    return close


def validate_and_fill(
    df: pd.DataFrame,
    grid: pd.DatetimeIndex,
    fill: str = "error",
) -> tuple[pd.DataFrame, list[str]]:
    """Reindex onto the session ``grid`` and enforce completeness.

    ``fill='error'`` (default): raise listing every missing minute. ``fill='ffill'``:
    forward-fill (back-fill the leading edge), returning the list of filled
    timestamps for the metadata sidecar.
    """
    df = df.copy()
    if grid.tz is not None and isinstance(df.index, pd.DatetimeIndex):
        df.index = (
            df.index.tz_convert(grid.tz)
            if df.index.tz is not None
            else df.index.tz_localize(grid.tz)
        )
    reixed = df.reindex(grid)
    missing_idx = reixed.index[reixed.isna().any(axis=1)]
    missing = [ts.isoformat() for ts in missing_idx]
    if missing:
        if fill == "error":
            raise ValueError(
                f"{len(missing)} missing minute(s) on the session grid "
                f"(use --fill ffill to forward-fill): {missing}"
            )
        if fill != "ffill":
            raise ValueError(f"unknown --fill {fill!r}; use error|ffill")
        reixed = reixed.ffill().bfill()
    if reixed.isna().any().any():
        bad = sorted(reixed.columns[reixed.isna().any(axis=0)])
        raise ValueError(f"unfillable NaNs remain for ticker(s) {bad} (no bars to fill from)")
    return reixed, missing


def normalize(df: pd.DataFrame, rule: str = "first=100") -> pd.DataFrame:
    """Apply the ``--normalize`` rule per column (mirrors ``load_mid_paths``)."""
    if rule == "none":
        return df.copy()
    m = re.fullmatch(r"first=([0-9.]+)", rule)
    if not m:
        raise ValueError(f"unknown --normalize {rule!r}; use none|first=<value>")
    target = float(m.group(1))
    first = df.iloc[0]
    if (first == 0).any():
        bad = sorted(first.index[first == 0])
        raise ValueError(f"cannot normalize: first row is 0 for ticker(s) {bad}")
    return df * (target / first)


def write_csv(df: pd.DataFrame, out: str | Path) -> Path:
    """Write the CSV in the legacy schema (``Datetime`` column + one per ticker)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df.index.name = "Datetime"
    out_df.reset_index().to_csv(out, index=False)
    return out


def write_metadata(meta: dict, out: str | Path) -> Path:
    """Write the JSON sidecar next to the CSV (``<out>.meta.json``)."""
    out = Path(out)
    side = out.parent / (out.name + ".meta.json")
    side.write_text(json.dumps(meta, indent=2) + "\n")
    return side


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split_ranges(
    train: Sequence[str],
    validation: Sequence[str],
    test: Sequence[str],
    *,
    dataset_start: str,
    dataset_end: str,
) -> dict[str, list[str]]:
    """Validate strict chronological split ranges and return ISO strings."""
    dataset_lo = pd.Timestamp(dataset_start).date()
    dataset_hi = pd.Timestamp(dataset_end).date()
    if dataset_hi < dataset_lo:
        raise ValueError("dataset end date precedes its start date")

    normalized: dict[str, list[str]] = {}
    parsed: dict[str, tuple[Date, Date]] = {}
    for name, values in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if len(values) != 2:
            raise ValueError(f"{name} split requires START END, got {list(values)!r}")
        start, end = (pd.Timestamp(value).date() for value in values)
        if end < start:
            raise ValueError(f"{name} split ends before it starts")
        if start < dataset_lo or end > dataset_hi:
            raise ValueError(
                f"{name} split [{start}, {end}] lies outside dataset range "
                f"[{dataset_lo}, {dataset_hi}]"
            )
        parsed[name] = (start, end)
        normalized[name] = [start.isoformat(), end.isoformat()]

    if not (
        parsed["train"][1] < parsed["validation"][0]
        and parsed["validation"][1] < parsed["test"][0]
    ):
        raise ValueError(
            "split ranges must be strictly chronological and nonoverlapping: "
            "train end < validation start and validation end < test start"
        )
    return normalized


def validate_historical_artifact(
    params: "HistoricalParams",
    repo_root: str | Path = ".",
    *,
    horizon: int | None = None,
) -> dict:
    """Verify the frozen CSV and sidecar against the resolved configuration."""
    csv_path = Path(params.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path(repo_root) / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"historical dataset not found: {csv_path}; run the configured data loader as "
            "documented in data/README.md"
        )
    if params.split_id.startswith("legacy_"):
        return {"dataset_id": params.split_id, "legacy_compatibility": True}

    sidecar = csv_path.parent / f"{csv_path.name}.meta.json"
    if not sidecar.exists():
        raise FileNotFoundError(
            f"historical dataset manifest not found: {sidecar}"
        )
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid historical dataset manifest {sidecar}: {exc}") from exc

    expected_ranges = {
        "train": list(params.train_date_range),
        "validation": list(params.validation_date_range),
        "test": list(params.test_date_range),
    }
    checks = {
        "dataset_id": (metadata.get("dataset_id"), params.split_id),
        "timezone": (metadata.get("timezone"), params.timezone),
        "missing-data treatment": (
            metadata.get("fill"),
            params.missing_data_treatment,
        ),
        "split ranges": (metadata.get("split_ranges"), expected_ranges),
        # Sidecars written before the explicit-clock field are unambiguously
        # one-minute artifacts because interval=1m and the lengths are stored
        # as clob_minutes/auction_minutes. Preserve those completed downloads.
        "time unit": (
            metadata.get(
                "time_unit",
                "minutes" if metadata.get("interval") == "1m" else None,
            ),
            "minutes",
        ),
    }
    provenance_requirements = {
        "source": params.source,
        "price type": params.price_type,
        "quote feed": params.quote_feed,
        "artifact normalization": params.artifact_normalization,
    }
    provenance_keys = {
        "source": "source",
        "price type": "price_type",
        "quote feed": "quote_feed",
        "artifact normalization": "normalize",
    }
    for label, expected in provenance_requirements.items():
        if expected:
            checks[label] = (metadata.get(provenance_keys[label]), expected)
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"historical dataset {label} mismatch: manifest={actual!r}, "
                f"config={expected!r}"
            )
    manifest_tickers = metadata.get("tickers")
    if not isinstance(manifest_tickers, list) or set(manifest_tickers) != set(params.symbols):
        raise ValueError(
            f"historical dataset ticker mismatch: manifest={manifest_tickers!r}, "
            f"config={list(params.symbols)!r}"
        )
    expected_digest = metadata.get("csv_sha256")
    actual_digest = _sha256(csv_path)
    if expected_digest != actual_digest:
        raise ValueError(
            f"historical dataset digest mismatch for {csv_path}: "
            f"manifest={expected_digest!r}, actual={actual_digest!r}"
        )
    if horizon is not None and int(metadata.get("n_rows_per_session", 0)) < int(horizon) + 1:
        raise ValueError(
            f"historical dataset sessions have {metadata.get('n_rows_per_session')} rows; "
            f"at least {int(horizon) + 1} are required through auction open"
        )
    split_dates = metadata.get("split_session_dates")
    if not isinstance(split_dates, dict):
        raise ValueError("historical dataset manifest lacks split_session_dates")
    for split in ("train", "validation", "test"):
        if not split_dates.get(split):
            raise ValueError(f"historical dataset manifest has an empty {split} split")
    return metadata


def _yfinance_version() -> Optional[str]:
    try:  # best-effort provenance; not required for the mocked path
        import yfinance as yf

        return getattr(yf, "__version__", None)
    except Exception:  # pragma: no cover - yfinance not installed
        return None


def regenerate(
    *,
    tickers: Sequence[str],
    date: str,
    interval: str = "1m",
    session_start: str = "13:30",
    session_end: str = "16:00",
    clob_minutes: int = 120,
    auction_minutes: int = 30,
    normalize_rule: str = "first=100",
    fill: str = "ffill",
    out: str | Path,
    cache_dir: Optional[str] = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    downloader=None,
) -> tuple[Path, dict]:
    """Download, validate, normalize, and write the historical mid CSV + sidecar.

    ``tau_op = clob_minutes`` and ``tau_cl = clob_minutes + auction_minutes``;
    the source window holds exactly ``tau_cl + 1`` bars.  The environment uses
    rows ``0..tau_op`` and freezes the row-``tau_op`` mid during the auction.
    ``downloader`` defaults to :func:`_download_yf` (tests pass a fake).
    """
    tickers = list(tickers)
    tau_op = clob_minutes
    tau_cl = clob_minutes + auction_minutes

    start_ts, end_ts = parse_session(date, session_start, session_end, timezone_name)
    grid = expected_grid(start_ts, end_ts, interval)
    if len(grid) != tau_cl + 1:
        raise ValueError(
            f"session window {session_start}-{session_end} ({interval}) yields "
            f"{len(grid)} bars but clob+auction+1 = {tau_cl + 1}; make the window "
            "match clob_minutes + auction_minutes (in bars)"
        )

    download = downloader or _download_yf
    raw = download(tickers, start_ts, end_ts, interval, cache_dir)
    missing_cols = [t for t in tickers if t not in raw.columns]
    if missing_cols:
        raise ValueError(f"download is missing column(s) for ticker(s) {missing_cols}")
    df = raw[tickers]

    filled_df, filled_minutes = validate_and_fill(df, grid, fill)
    norm_df = normalize(filled_df, normalize_rule)

    out_path = write_csv(norm_df, out)
    meta = {
        "tickers": tickers,
        "date": date,
        "interval": interval,
        "time_unit": "minutes",
        "session_start": session_start,
        "session_end": session_end,
        "timezone": timezone_name,
        "clob_minutes": clob_minutes,
        "auction_minutes": auction_minutes,
        "tau_op": tau_op,
        "tau_cl": tau_cl,
        "n_rows": int(len(norm_df)),
        "source": "yfinance",
        "mid_proxy": MID_PROXY,
        "normalize": normalize_rule,
        "fill": fill,
        "filled_minutes": filled_minutes,
        "row_to_decision_time": "row r -> mid at decision time t=r; rows 0..tau_op "
        "supply the path through auction open; row tau_op is frozen during the call",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": _yfinance_version(),
        "csv_sha256": _sha256(out_path),
    }
    meta_path = write_metadata(meta, out_path)
    logger.info("wrote %s (%d rows) and %s", out_path, len(norm_df), meta_path)
    return out_path, meta


def regenerate_range(
    *,
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    train_date_range: Sequence[str],
    validation_date_range: Sequence[str],
    test_date_range: Sequence[str],
    interval: str = "1m",
    session_start: str = "13:30",
    session_end: str = "16:00",
    clob_minutes: int = 120,
    auction_minutes: int = 30,
    normalize_rule: str = "first=100",
    fill: str = "ffill",
    out: str | Path,
    cache_dir: Optional[str] = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    dataset_id: str | None = None,
    allow_skipped_sessions: bool = False,
    downloader=None,
) -> tuple[Path, dict]:
    """Create one frozen CSV containing multiple independent sessions.

    Candidate dates are weekdays in ``[start_date, end_date]``.  Provider-empty
    dates fail by default so a transient outage cannot silently reduce a split;
    callers spanning an exchange holiday may opt into explicit skipping, which
    is recorded in the sidecar.  Regardless of that setting, partial ticker
    failures and unfillable sessions always fail.
    """
    tickers = list(tickers)
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

    download = downloader or _download_yf
    sessions: list[pd.DataFrame] = []
    session_records: list[dict] = []
    skipped_sessions: list[dict] = []
    for session_date in candidate_dates:
        start_ts, end_ts = parse_session(
            session_date, session_start, session_end, timezone_name
        )
        grid = expected_grid(start_ts, end_ts, interval)
        if len(grid) != tau_cl + 1:
            raise ValueError(
                f"session window {session_start}-{session_end} ({interval}) yields "
                f"{len(grid)} bars but clob+auction+1 = {tau_cl + 1}"
            )
        try:
            raw = download(tickers, start_ts, end_ts, interval, cache_dir)
        except NoSessionDataError as exc:
            if not allow_skipped_sessions:
                raise
            skipped_sessions.append(
                {"date": session_date, "reason": str(exc)}
            )
            continue

        missing_cols = [ticker for ticker in tickers if ticker not in raw.columns]
        if missing_cols:
            raise ValueError(
                f"download for {session_date} is missing ticker column(s) {missing_cols}"
            )
        filled_df, filled_minutes = validate_and_fill(raw[tickers], grid, fill)
        normalized = normalize(filled_df, normalize_rule)
        sessions.append(normalized)
        session_records.append(
            {
                "date": session_date,
                "n_rows": int(len(normalized)),
                "filled_minutes": filled_minutes,
            }
        )

    if not sessions:
        raise ValueError("no complete market sessions were downloaded")

    included_dates = [record["date"] for record in session_records]
    split_session_dates: dict[str, list[str]] = {}
    for split, (lower, upper) in split_ranges.items():
        selected = [date for date in included_dates if lower <= date <= upper]
        if not selected:
            raise ValueError(
                f"{split} split [{lower}, {upper}] contains no downloaded sessions"
            )
        split_session_dates[split] = selected

    combined = pd.concat(sessions).sort_index()
    if combined.index.duplicated().any():
        duplicate = combined.index[combined.index.duplicated()][0]
        raise ValueError(f"duplicate timestamp across sessions: {duplicate}")
    out_path = write_csv(combined, out)
    resolved_dataset_id = dataset_id or (
        f"sp500_{interval}_{start_date}_{end_date}"
    )
    meta = {
        "artifact_schema_version": 2,
        "dataset_id": resolved_dataset_id,
        "tickers": tickers,
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
        "source": "yfinance",
        "mid_proxy": MID_PROXY,
        "normalize": normalize_rule,
        "fill": fill,
        "row_to_decision_time": "within each session, row r -> mid at t=r; "
        "rows 0..tau_op supply the path through auction open; row tau_op is "
        "frozen during the call",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": _yfinance_version(),
        "csv_sha256": _sha256(out_path),
    }
    sidecar = write_metadata(meta, out_path)
    logger.info(
        "wrote %s (%d sessions, %d rows) and %s",
        out_path,
        len(session_records),
        len(combined),
        sidecar,
    )
    return out_path, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-load-data",
        description="Download one or more sessions of 1-minute S&P 500 price "
        "inputs and write a normalized CSV plus provenance sidecar.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="ticker symbols (5-10; default: the paper's five CAT PG GOOGL JPM MSFT)",
    )
    parser.add_argument("--date", default=None, help="one session date (single-session mode)")
    parser.add_argument("--start-date", default=None, help="first date (multi-session mode)")
    parser.add_argument("--end-date", default=None, help="last date (multi-session mode)")
    parser.add_argument("--train-range", nargs=2, metavar=("START", "END"))
    parser.add_argument("--validation-range", nargs=2, metavar=("START", "END"))
    parser.add_argument("--test-range", nargs=2, metavar=("START", "END"))
    parser.add_argument("--dataset-id", default=None, help="stable provenance identifier")
    parser.add_argument("--interval", default="1m", help="yfinance bar interval (default 1m)")
    parser.add_argument("--session-start", default="13:30", help="session start (wall-clock, --timezone)")
    parser.add_argument("--session-end", default="16:00", help="session end (wall-clock, --timezone)")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="session timezone (legacy: Eastern)")
    parser.add_argument("--clob-minutes", type=int, default=120, help="CLOB length -> tau_op")
    parser.add_argument(
        "--auction-minutes", type=int, default=30, help="auction length -> tau_cl - tau_op"
    )
    parser.add_argument(
        "--normalize",
        default="first=100",
        choices=["none", "first=100"],
        help="normalization rule (default: first row scaled to 100)",
    )
    parser.add_argument(
        "--fill",
        default="ffill",
        choices=["error", "ffill"],
        help="missing-minute policy (default: ffill, with every filled minute recorded)",
    )
    parser.add_argument(
        "--allow-skipped-sessions",
        action="store_true",
        help="record and skip provider-empty weekdays (for ranges containing holidays)",
    )
    parser.add_argument("--out", default=None, help="output CSV")
    parser.add_argument("--cache-dir", default=".cache/yfinance", help="yfinance download cache dir")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (5 <= len(args.tickers) <= 10):
        parser.error(f"--tickers must have 5-10 symbols, got {len(args.tickers)}")

    range_values = (
        args.start_date,
        args.end_date,
        args.train_range,
        args.validation_range,
        args.test_range,
    )
    range_mode = any(value is not None for value in range_values)
    if range_mode:
        if args.date is not None:
            parser.error("--date cannot be combined with multi-session range options")
        if not all(value is not None for value in range_values):
            parser.error(
                "multi-session mode requires --start-date, --end-date, "
                "--train-range, --validation-range, and --test-range"
            )
        out = args.out or "data/historical_sp500_1m.csv"
        out_path, meta = regenerate_range(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            train_date_range=args.train_range,
            validation_date_range=args.validation_range,
            test_date_range=args.test_range,
            interval=args.interval,
            session_start=args.session_start,
            session_end=args.session_end,
            clob_minutes=args.clob_minutes,
            auction_minutes=args.auction_minutes,
            normalize_rule=args.normalize,
            fill=args.fill,
            out=out,
            cache_dir=args.cache_dir,
            timezone_name=args.timezone,
            dataset_id=args.dataset_id,
            allow_skipped_sessions=args.allow_skipped_sessions,
        )
        print(
            f"wrote {out_path} ({meta['n_sessions']} sessions, {meta['n_rows']} rows) "
            f"+ {out_path.name}.meta.json"
        )
    else:
        if args.date is None:
            parser.error("provide --date or the complete multi-session range options")
        out = args.out or f"data/mid_prices_{args.date}.csv"
        out_path, meta = regenerate(
            tickers=args.tickers,
            date=args.date,
            interval=args.interval,
            session_start=args.session_start,
            session_end=args.session_end,
            clob_minutes=args.clob_minutes,
            auction_minutes=args.auction_minutes,
            normalize_rule=args.normalize,
            fill=args.fill,
            out=out,
            cache_dir=args.cache_dir,
            timezone_name=args.timezone,
        )
        print(f"wrote {out_path} ({meta['n_rows']} rows) + {out_path.name}.meta.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
