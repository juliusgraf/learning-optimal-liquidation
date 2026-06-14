"""CLI regenerating the historical mid-price CSV (ruling D13; Phase 6).

The committed ``legacy/data.csv`` IS real data: S&P 500 1-minute mid prices,
Dec 31 2025, normalized to 100 at session start — treated as the FROZEN
experimental input (ruling D13, audit A.12). The historical experiments default
to that committed CSV; this loader exists to *regenerate* a schema-compatible
dataset for chosen assets/dates from yfinance 1-minute bars.

Row -> decision-time mapping (the env, ``HistoricalMidPrice``, reproduces it):
row ``r`` is the mid at decision time ``t = r``. The CLOB phase consumes rows
``0..tau_op-1`` (``tau_op = clob_minutes``); the auction mid is frozen at row
``tau_op-1``; the terminal clearing is at ``tau_cl = clob_minutes +
auction_minutes``. The written CSV spans the full decision grid
(``tau_cl + 1`` rows); the env reads only ``n_rows = tau_op`` via config.

Mid proxy: yfinance bars are NOT true midpoints — we use the bar *close* as a
mid proxy and record this in the metadata sidecar.

Timezone: ``--session-start``/``--session-end`` are wall-clock times in
``--timezone`` (default ``America/New_York``; EST on Dec 31). yfinance returns
1m bars only for a ~30-day lookback, which is why old dates cannot be
re-fetched and the historical experiments keep using the frozen CSV. The legacy
CSV labels its Eastern timestamps ``+00:00`` (a known quirk, audit A.12/D13);
this loader writes correct tz-aware Eastern ``Datetime`` instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

__all__ = [
    "parse_session",
    "expected_grid",
    "validate_and_fill",
    "normalize",
    "write_csv",
    "write_metadata",
    "regenerate",
    "build_parser",
    "main",
]

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ("CAT", "PG", "GOOGL", "JPM", "MSFT")  # the paper's five (data.py:1651)
DEFAULT_TIMEZONE = "America/New_York"
MID_PROXY = "yfinance 1m bar close (not a true midpoint)"


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

    ``session_start``/``session_end`` are wall-clock (e.g. ``14:30``/``17:00``);
    the legacy CSV uses Eastern times (ruling D13), hence the default tz.
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
        close = pd.DataFrame({t: data[t]["Close"] for t in tickers})
    else:  # single ticker -> flat columns
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.index.name = "Datetime"
    return close


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
    return f"{span}_{interval}_{'-'.join(tickers)}.parquet"


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
                return pd.read_parquet(cache_path)
            except Exception as exc:  # pragma: no cover - corrupt/unsupported cache
                logger.warning("ignoring unreadable cache %s: %s", cache_path, exc)

    import yfinance as yf  # lazy: optional at import time, mocked in tests

    end_excl = end_ts + pd.Timedelta(_interval_to_freq(interval))  # yfinance end is exclusive
    data = yf.download(
        tickers,
        start=start_ts,
        end=end_excl,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if data is None or len(data) == 0:
        raise RuntimeError(
            f"yfinance returned no {interval} bars for {tickers} on {start_ts.date()} "
            "(beyond the ~30-day intraday lookback? wrong session/date?)"
        )
    close = _extract_close(data, tickers)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            close.to_parquet(cache_path)
        except Exception as exc:  # pragma: no cover - parquet engine missing
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
    session_start: str = "14:30",
    session_end: str = "17:00",
    clob_minutes: int = 120,
    auction_minutes: int = 30,
    normalize_rule: str = "first=100",
    fill: str = "error",
    out: str | Path,
    cache_dir: Optional[str] = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    downloader=None,
) -> tuple[Path, dict]:
    """Download, validate, normalize, and write the historical mid CSV + sidecar.

    Reproduces the legacy row -> ``t_j`` mapping (ruling D13): ``tau_op =
    clob_minutes``, ``tau_cl = clob_minutes + auction_minutes``; the session
    window must hold exactly ``tau_cl + 1`` bars. ``downloader`` defaults to the
    module-level :func:`_download_yf` (tests pass a fake or monkeypatch it).
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
        "row_to_decision_time": "row r -> mid at decision time t=r; rows 0..tau_op-1 "
        "are the CLOB phase; auction mid frozen at row tau_op-1 (ruling D13)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": _yfinance_version(),
    }
    meta_path = write_metadata(meta, out_path)
    logger.info("wrote %s (%d rows) and %s", out_path, len(norm_df), meta_path)
    return out_path, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-load-data",
        description="Download 1-minute mid prices via yfinance and write the "
        "normalized historical CSV + metadata sidecar (ruling D13).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="ticker symbols (5-10; default: the paper's five CAT PG GOOGL JPM MSFT)",
    )
    parser.add_argument("--date", default="2025-12-31", help="session date, e.g. 2025-12-31")
    parser.add_argument("--interval", default="1m", help="yfinance bar interval (default 1m)")
    parser.add_argument("--session-start", default="14:30", help="session start (wall-clock, --timezone)")
    parser.add_argument("--session-end", default="17:00", help="session end (wall-clock, --timezone)")
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
        default="error",
        choices=["error", "ffill"],
        help="missing-minute policy (default: fail loudly)",
    )
    parser.add_argument("--out", default=None, help="output CSV (default data/mid_prices_<date>.csv)")
    parser.add_argument("--cache-dir", default=".cache/yfinance", help="yfinance download cache dir")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if not (5 <= len(args.tickers) <= 10):
        build_parser().error(f"--tickers must have 5-10 symbols, got {len(args.tickers)}")
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
