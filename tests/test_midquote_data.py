"""True-quote historical-data contract (network-free)."""

from __future__ import annotations

import json
from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from helpers import REPO_ROOT, load_historical_cfg
from lmm.data.load_midquote_data import (
    QuoteDataError,
    aggregate_quote_midpoints,
    download_alpaca_quotes,
    regenerate_range,
)
from lmm.data.historical_artifact import validate_historical_artifact


def _quote(ts: str, bid: float, ask: float, bid_size: int = 2, ask_size: int = 3):
    return {"t": ts, "bp": bid, "ap": ask, "bs": bid_size, "as": ask_size}


def test_alpaca_download_paginates_and_keeps_credentials_out_of_url():
    calls: list[tuple[str, dict[str, str]]] = []

    def requester(url, headers):
        calls.append((url, dict(headers)))
        token = parse_qs(urlparse(url).query).get("page_token")
        if token is None:
            return {
                "quotes": {"MSFT": [_quote("2026-08-03T17:30:00Z", 100, 100.02)]},
                "next_page_token": "page-2",
            }
        return {
            "quotes": {"MSFT": [_quote("2026-08-03T17:30:01Z", 100.01, 100.03)]},
            "next_page_token": None,
        }

    events = download_alpaca_quotes(
        ["MSFT"],
        pd.Timestamp("2026-08-03 13:30", tz="America/New_York"),
        pd.Timestamp("2026-08-03 13:31", tz="America/New_York"),
        api_key_id="test-key",
        api_secret_key="test-secret",
        requester=requester,
    )
    assert len(events["MSFT"]) == 2
    assert len(calls) == 2
    assert parse_qs(urlparse(calls[1][0]).query)["page_token"] == ["page-2"]
    assert "test-key" not in calls[0][0] and "test-secret" not in calls[0][0]
    assert calls[0][1]["APCA-API-KEY-ID"] == "test-key"
    assert calls[0][1]["APCA-API-SECRET-KEY"] == "test-secret"


def test_aggregation_uses_true_midpoint_latest_at_or_before_and_rejects_crossed():
    grid = pd.date_range(
        "2026-08-03 13:30", periods=3, freq="1min", tz="America/New_York", name="Datetime"
    )
    events = {
        "MSFT": [
            _quote("2026-08-03T17:29:59Z", 99.98, 100.02),
            _quote("2026-08-03T17:30:30Z", 101.00, 100.00),  # crossed: rejected
            _quote("2026-08-03T17:30:59Z", 100.00, 100.04),
            _quote("2026-08-03T17:31:30Z", 100.02, 100.06),
            _quote("2026-08-03T17:32:01Z", 999.00, 999.02),  # future: unavailable
        ]
    }
    frame, diagnostics = aggregate_quote_midpoints(
        events, ["MSFT"], grid, max_quote_age_seconds=61
    )
    assert frame["MSFT"].tolist() == pytest.approx([100.0, 100.02, 100.04])
    assert diagnostics["MSFT"]["n_crossed"] == 1
    assert diagnostics["MSFT"]["median_spread_bps"] > 0


def test_aggregation_fails_instead_of_silently_filling_a_stale_quote():
    grid = pd.date_range(
        "2026-08-03 13:30", periods=2, freq="1min", tz="America/New_York"
    )
    events = {"MSFT": [_quote("2026-08-03T17:29:00Z", 100, 100.02)]}
    with pytest.raises(QuoteDataError, match="above the 30s limit"):
        aggregate_quote_midpoints(events, ["MSFT"], grid, max_quote_age_seconds=30)


def test_range_builder_keeps_raw_prices_and_writes_fail_closed_provenance(tmp_path):
    dates = ("2026-08-03", "2026-08-04", "2026-08-05")

    def downloader(tickers, start_ts, end_ts, **kwargs):
        del end_ts, kwargs
        day_offset = dates.index(start_ts.date().isoformat())
        base = 120.0 + day_offset
        rows = []
        for minute in range(3):
            ts = start_ts + pd.Timedelta(minutes=5 + minute) - pd.Timedelta(seconds=1)
            rows.append(_quote(ts.tz_convert("UTC").isoformat(), base + minute, base + minute + 0.02))
        return {ticker: list(rows) for ticker in tickers}

    out = tmp_path / "midquotes.csv"
    raw_dir = tmp_path / "raw"
    path, metadata = regenerate_range(
        tickers=["MSFT"],
        start_date=dates[0],
        end_date=dates[-1],
        train_date_range=[dates[0], dates[0]],
        validation_date_range=[dates[1], dates[1]],
        test_date_range=[dates[2], dates[2]],
        out=out,
        dataset_id="quote_fixture_v1",
        session_start="13:30",
        session_end="13:32",
        clob_minutes=1,
        auction_minutes=1,
        raw_dir=raw_dir,
        cache_dir=None,
        downloader=downloader,
    )
    stored = pd.read_csv(path)
    assert stored["MSFT"].iloc[0] == pytest.approx(120.01)
    assert metadata["source"] == "alpaca_market_data_api"
    assert metadata["price_type"] == "quote_midpoint"
    assert metadata["quote_feed"] == "sip"
    assert metadata["normalize"] == "none"
    assert metadata["time_unit"] == "minutes"
    assert len(list(raw_dir.glob("*.jsonl.gz"))) == 3

    cfg = load_historical_cfg()
    original = cfg.midprice.historical
    assert original is not None
    params = replace(
        original,
        csv_path=path,
        symbols=("MSFT",),
        n_rows=2,
        split_id="quote_fixture_v1",
        train_date_range=(dates[0], dates[0]),
        validation_date_range=(dates[1], dates[1]),
        test_date_range=(dates[2], dates[2]),
        missing_data_treatment="error",
        source="alpaca_market_data_api",
        price_type="quote_midpoint",
        quote_feed="sip",
        artifact_normalization="none",
    )
    manifest = validate_historical_artifact(params, REPO_ROOT, horizon=1)
    assert manifest["csv_sha256"] == json.loads(
        path.with_name(path.name + ".meta.json").read_text()
    )["csv_sha256"]
    with pytest.raises(ValueError, match="historical dataset source mismatch"):
        validate_historical_artifact(replace(params, source="bar_close_proxy"), REPO_ROOT)

    raw_archive = next(raw_dir.glob("*.jsonl.gz"))
    with raw_archive.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="raw historical quote archive digest mismatch"):
        validate_historical_artifact(params, REPO_ROOT, horizon=1)
