"""Session fixtures for the Phase 3 market/env tests (helpers in helpers.py)."""

from __future__ import annotations

import pytest

from helpers import load_historical_cfg, load_synthetic_cfg


@pytest.fixture(scope="session")
def synthetic_cfg():
    return load_synthetic_cfg()


@pytest.fixture(scope="session")
def historical_cfg():
    return load_historical_cfg()


@pytest.fixture(scope="session")
def crn_synthetic_cfg():
    """Synthetic config with the conditional auction draws disabled (see
    helpers.py docstring) — bit-identical streams across an injection pair."""
    return load_synthetic_cfg("auction_flow.p2=0.0", "auction_flow.p4=0.0")
