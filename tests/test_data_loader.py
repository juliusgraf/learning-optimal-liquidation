"""Phase 6 data-pipeline tests (ruling D13).

No network in CI: the yfinance seam (``_download_yf``) is replaced by a fake
returning a synthetic tidy bars frame. One ``@pytest.mark.network`` test does a
real download and is deselected by ``-m "not network"``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from helpers import REPO_ROOT, load_historical_cfg

from lmm.config import HistoricalParams
from lmm.data import load_yfinance_data as ld
from lmm.market.midprice import HistoricalMidPrice, build_midprice, load_mid_paths

DATE = "2025-12-31"
TICKERS = ("CAT", "PG", "GOOGL", "JPM", "MSFT")
LEGACY_CSV = REPO_ROOT / "legacy" / "data.csv"


def _grid():
    start, end = ld.parse_session(DATE, "14:30", "17:00")
    return ld.expected_grid(start, end, "1m")


def _synthetic_closes(grid, tickers=TICKERS) -> pd.DataFrame:
    """A smooth, distinct-per-ticker close path on the full grid (no NaNs)."""
    n = len(grid)
    data = {
        t: 100.0 + i + np.sin(np.arange(n) / 7.0) * (i + 1) * 0.1
        for i, t in enumerate(tickers)
    }
    df = pd.DataFrame(data, index=grid)
    df.index.name = "Datetime"
    return df


def make_downloader(closes: pd.DataFrame, drop_positions=()):
    """A fake ``_download_yf`` returning ``closes`` minus any dropped minutes."""

    def _dl(tickers, start_ts, end_ts, interval, cache_dir=None):
        df = closes[list(tickers)].copy()
        if drop_positions:
            df = df.drop(index=[closes.index[k] for k in drop_positions])
        return df

    return _dl


# --------------------------------------------------------------------------- #
# grid alignment + row -> decision-time mapping
# --------------------------------------------------------------------------- #
def test_grid_alignment_and_row_mapping(tmp_path):
    grid = _grid()
    closes = _synthetic_closes(grid)
    out = tmp_path / "mid.csv"
    out_path, meta = ld.regenerate(
        tickers=TICKERS,
        date=DATE,
        normalize_rule="none",
        out=out,
        downloader=make_downloader(closes),
    )

    assert meta["tau_op"] == 120 and meta["tau_cl"] == 150
    assert meta["n_rows"] == 151  # tau_cl + 1, the full decision grid

    written = pd.read_csv(out_path)
    assert list(written.columns) == ["Datetime"] + list(TICKERS)
    assert len(written) == 151
    dt = pd.to_datetime(written["Datetime"], utc=True)
    assert dt.is_monotonic_increasing
    assert (dt.diff().dropna() == pd.Timedelta(minutes=1)).all()
    # row r holds the close at minute r (mid proxy), un-normalized
    np.testing.assert_allclose(written["CAT"].to_numpy(), closes["CAT"].to_numpy())

    # the env consumes rows 0..tau_op-1 and normalizes by 100/row0 (ruling D13)
    params = HistoricalParams(
        csv_path=out_path, symbols=("CAT",), normalize_first=100.0, n_rows=120
    )
    path = load_mid_paths(params, repo_root=tmp_path)["CAT"]
    assert len(path) == 120
    raw = closes["CAT"].to_numpy()[:120]
    np.testing.assert_allclose(path, raw * (100.0 / raw[0]))


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def test_normalize_first_100(tmp_path):
    grid = _grid()
    closes = _synthetic_closes(grid)
    out_path, _ = ld.regenerate(
        tickers=TICKERS, date=DATE, normalize_rule="first=100",
        out=tmp_path / "n.csv", downloader=make_downloader(closes),
    )
    written = pd.read_csv(out_path)
    first = written[list(TICKERS)].iloc[0].to_numpy()
    np.testing.assert_allclose(first, 100.0)
    # ratios preserved column-wise
    np.testing.assert_allclose(
        written["CAT"].to_numpy(), closes["CAT"].to_numpy() * (100.0 / closes["CAT"].iloc[0])
    )


def test_normalize_none_is_identity(tmp_path):
    grid = _grid()
    closes = _synthetic_closes(grid)
    out_path, _ = ld.regenerate(
        tickers=TICKERS, date=DATE, normalize_rule="none",
        out=tmp_path / "raw.csv", downloader=make_downloader(closes),
    )
    written = pd.read_csv(out_path)
    np.testing.assert_allclose(written["MSFT"].to_numpy(), closes["MSFT"].to_numpy())


# --------------------------------------------------------------------------- #
# completeness validation
# --------------------------------------------------------------------------- #
def test_missing_minute_fails_loudly(tmp_path):
    grid = _grid()
    closes = _synthetic_closes(grid)
    missing_iso = grid[50].isoformat()
    with pytest.raises(ValueError, match="missing minute"):
        ld.regenerate(
            tickers=TICKERS, date=DATE, fill="error",
            out=tmp_path / "x.csv", downloader=make_downloader(closes, drop_positions=(50,)),
        )
    # and the offending timestamp is named
    with pytest.raises(ValueError, match=re_escape(missing_iso)):
        ld.regenerate(
            tickers=TICKERS, date=DATE, fill="error",
            out=tmp_path / "x.csv", downloader=make_downloader(closes, drop_positions=(50,)),
        )


def test_missing_minute_ffill(tmp_path):
    grid = _grid()
    closes = _synthetic_closes(grid)
    out_path, meta = ld.regenerate(
        tickers=TICKERS, date=DATE, fill="ffill", normalize_rule="none",
        out=tmp_path / "f.csv", downloader=make_downloader(closes, drop_positions=(50,)),
    )
    assert grid[50].isoformat() in meta["filled_minutes"]
    written = pd.read_csv(out_path)
    assert len(written) == 151
    # forward-filled row equals the previous minute
    assert written["CAT"].iloc[50] == pytest.approx(closes["CAT"].iloc[49])


# --------------------------------------------------------------------------- #
# metadata sidecar
# --------------------------------------------------------------------------- #
def test_metadata_sidecar(tmp_path):
    grid = _grid()
    closes = _synthetic_closes(grid)
    out_path, meta = ld.regenerate(
        tickers=TICKERS, date=DATE, out=tmp_path / "m.csv",
        downloader=make_downloader(closes),
    )
    side = out_path.parent / (out_path.name + ".meta.json")
    assert side.exists()
    disk = json.loads(side.read_text())
    assert disk == meta
    for key in (
        "tickers", "date", "interval", "session_start", "session_end", "timezone",
        "clob_minutes", "auction_minutes", "tau_op", "tau_cl", "n_rows", "source",
        "mid_proxy", "normalize", "fill", "filled_minutes", "row_to_decision_time",
        "downloaded_at_utc", "yfinance_version",
    ):
        assert key in disk
    assert disk["source"] == "yfinance"
    assert "close" in disk["mid_proxy"]
    assert disk["tickers"] == list(TICKERS)
    assert disk["date"] == DATE
    # download timestamp is a valid ISO instant
    pd.Timestamp(disk["downloaded_at_utc"])


# --------------------------------------------------------------------------- #
# schema contract: generated CSV matches legacy/data.csv's structure
# --------------------------------------------------------------------------- #
def assert_schema(path, *, first_value=None):
    df = pd.read_csv(path)
    assert df.columns[0] == "Datetime"
    assert len(df.columns) >= 2
    dt = pd.to_datetime(df["Datetime"], utc=True)  # parses without error
    assert dt.is_monotonic_increasing
    for col in df.columns[1:]:
        assert pd.api.types.is_float_dtype(df[col]), f"{col} not float"
    if first_value is not None:
        np.testing.assert_allclose(df[df.columns[1:]].iloc[0].to_numpy(), first_value)


def test_frozen_csv_schema_contract(tmp_path):
    # the frozen experimental input satisfies the contract
    assert_schema(LEGACY_CSV, first_value=100.0)
    # and so does a freshly regenerated CSV
    grid = _grid()
    closes = _synthetic_closes(grid)
    out_path, _ = ld.regenerate(
        tickers=TICKERS, date=DATE, normalize_rule="first=100",
        out=tmp_path / "s.csv", downloader=make_downloader(closes),
    )
    assert_schema(out_path, first_value=100.0)


# --------------------------------------------------------------------------- #
# path_policy (Phase 6: fixed only; bootstrap reserved)
# --------------------------------------------------------------------------- #
def test_path_policy_fixed_replays(tmp_path):
    cfg = load_historical_cfg()
    model = build_midprice(cfg, symbol="MSFT", repo_root=REPO_ROOT)
    assert isinstance(model, HistoricalMidPrice)
    rng = np.random.default_rng(0)
    assert model.reset(rng) == model.path[0]
    assert model.advance_to(5.0) == model.path[5]
    # frozen auction mid is row tau_op - 1
    assert model.advance_to(float(cfg.grid.tau_op)) == model.path[cfg.grid.tau_op - 1]


def test_path_policy_bootstrap_deferred():
    cfg = load_historical_cfg("midprice.historical.path_policy=bootstrap")
    with pytest.raises(NotImplementedError, match="bootstrap"):
        build_midprice(cfg, symbol="MSFT", repo_root=REPO_ROOT)


# --------------------------------------------------------------------------- #
# live download (deselected by default: -m "not network")
# --------------------------------------------------------------------------- #
@pytest.mark.network
def test_live_download(tmp_path):
    pytest.importorskip("yfinance")
    recent = (pd.Timestamp.now(tz="America/New_York") - pd.offsets.BDay(3)).strftime("%Y-%m-%d")
    try:
        out_path, meta = ld.regenerate(
            tickers=list(TICKERS),
            date=recent,
            session_start="14:30",
            session_end="14:35",
            clob_minutes=3,
            auction_minutes=2,
            fill="ffill",
            out=tmp_path / "live.csv",
            cache_dir=str(tmp_path / "cache"),
        )
    except Exception as exc:  # network/data availability is not under test
        pytest.skip(f"live yfinance download unavailable: {exc}")
    assert out_path.exists()
    assert meta["n_rows"] == 6  # tau_cl + 1
    assert meta["yfinance_version"] is not None
    assert_schema(out_path)


def re_escape(s: str) -> str:
    import re

    return re.escape(s)
