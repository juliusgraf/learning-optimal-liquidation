"""Shared contracts for the frozen true-midquote historical artifact.

This module contains only provider-independent timestamp, persistence, split,
and provenance validation helpers.  The sole active downloader lives in
``load_midquote_data`` and obtains bid/ask quote events from Alpaca.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date as Date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:
    from lmm.config import HistoricalParams

__all__ = [
    "DEFAULT_TICKERS",
    "DEFAULT_TIMEZONE",
    "NoSessionDataError",
    "parse_session",
    "expected_grid",
    "write_csv",
    "write_metadata",
    "sha256_file",
    "validate_split_ranges",
    "validate_historical_artifact",
]

DEFAULT_TICKERS = ("CAT", "PG", "GOOGL", "JPM", "MSFT")
DEFAULT_TIMEZONE = "America/New_York"


class NoSessionDataError(RuntimeError):
    """The provider returned no quotes for an otherwise valid session."""


def _interval_to_freq(interval: str) -> str:
    match = re.fullmatch(r"(\d+)(m|h)", interval)
    if not match:
        raise ValueError(f"unsupported interval {interval!r}; use e.g. 1m, 5m, 1h")
    count, unit = int(match.group(1)), match.group(2)
    return f"{count}min" if unit == "m" else f"{count * 60}min"


def parse_session(
    date: str,
    session_start: str,
    session_end: str,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Localize a wall-clock session window to ``timezone_name``."""
    start = pd.Timestamp(f"{date} {session_start}", tz=timezone_name)
    end = pd.Timestamp(f"{date} {session_end}", tz=timezone_name)
    if end <= start:
        raise ValueError(
            f"session-end {session_end} must be after session-start {session_start}"
        )
    return start, end


def expected_grid(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: str,
) -> pd.DatetimeIndex:
    """Inclusive decision grid of quote-selection timestamps."""
    grid = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq=_interval_to_freq(interval),
    )
    grid.name = "Datetime"
    return grid


def write_csv(df: pd.DataFrame, out: str | Path) -> Path:
    """Write ``Datetime`` plus one raw midpoint column per ticker."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    out_df.index.name = "Datetime"
    out_df.reset_index().to_csv(out, index=False)
    return out


def write_metadata(meta: dict, out: str | Path) -> Path:
    """Write the JSON manifest next to the CSV as ``<csv>.meta.json``."""
    out = Path(out)
    sidecar = out.parent / f"{out.name}.meta.json"
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    return sidecar


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _sha256_at_stat(path: str, size: int, mtime_ns: int) -> str:
    """Hash one immutable artifact version, cached across env construction."""
    del size, mtime_ns  # included in the cache key
    return sha256_file(path)


def _validated_digest(path: Path) -> str:
    stat = path.stat()
    return _sha256_at_stat(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


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


def _resolve_manifest_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def validate_historical_artifact(
    params: "HistoricalParams",
    repo_root: str | Path = ".",
    *,
    horizon: int | None = None,
) -> dict:
    """Fail closed unless a raw-price, true-quote artifact matches the config."""
    root = Path(repo_root)
    csv_path = Path(params.csv_path)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"historical dataset not found: {csv_path}; build it as documented "
            "in data/README.md"
        )

    sidecar = csv_path.parent / f"{csv_path.name}.meta.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"historical dataset manifest not found: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid historical dataset manifest {sidecar}: {exc}") from exc

    schema = metadata.get("artifact_schema_version")
    if not isinstance(schema, int) or schema < 3:
        raise ValueError(
            "historical dataset must use true-midquote artifact_schema_version >= 3"
        )

    expected_ranges = {
        "train": list(params.train_date_range),
        "validation": list(params.validation_date_range),
        "test": list(params.test_date_range),
    }
    checks = {
        "dataset_id": (metadata.get("dataset_id"), params.split_id),
        "timezone": (metadata.get("timezone"), params.timezone),
        "missing-data treatment": (metadata.get("fill"), params.missing_data_treatment),
        "split ranges": (metadata.get("split_ranges"), expected_ranges),
        "time unit": (metadata.get("time_unit"), "minutes"),
        "source": (metadata.get("source"), params.source),
        "price type": (metadata.get("price_type"), params.price_type),
        "quote feed": (metadata.get("quote_feed"), params.quote_feed),
        "artifact normalization": (
            metadata.get("normalize"),
            params.artifact_normalization,
        ),
    }
    for label, (actual, expected) in checks.items():
        if not expected or actual != expected:
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
    actual_digest = _validated_digest(csv_path)
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

    sessions = metadata.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != metadata.get("n_sessions"):
        raise ValueError("historical dataset manifest has inconsistent session records")
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("historical dataset manifest contains an invalid session record")
        raw_path = session.get("raw_quote_path")
        raw_digest = session.get("raw_quote_sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_digest, str):
            raise ValueError("historical session lacks raw quote archive provenance")
        archive = _resolve_manifest_path(raw_path, root)
        if not archive.exists():
            raise FileNotFoundError(f"raw historical quote archive not found: {raw_path}")
        actual_raw_digest = _validated_digest(archive)
        if actual_raw_digest != raw_digest:
            raise ValueError(
                f"raw historical quote archive digest mismatch for {raw_path}: "
                f"manifest={raw_digest!r}, actual={actual_raw_digest!r}"
            )
    return metadata
