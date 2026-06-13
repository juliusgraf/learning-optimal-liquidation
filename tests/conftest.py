"""Session fixtures for the Phase 3 market/env tests (helpers in helpers.py)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless figure tests (Phase 7)

from helpers import load_historical_cfg, load_synthetic_cfg

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_run_dir() -> Path:
    """A committed, complete (DQN) run directory for figure/table tests."""
    return FIXTURES / "run_dir"


@pytest.fixture
def fixture_run_dir_ddpg() -> Path:
    """A minimal sibling (DDPG) run dir to exercise the multi-run path."""
    return FIXTURES / "run_dir_ddpg"


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
