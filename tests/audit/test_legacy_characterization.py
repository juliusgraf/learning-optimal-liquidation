"""Characterization tests pinning the LEGACY emulators (Phase 1, deliverable E).

These tests run the legacy `MarketEmulator` from main.py (synthetic /
rough-Heston mid) and data.py (historical mid path) for a few seeded episodes
under fixed deterministic policies, and assert golden summary statistics:
total reward, final inventory, number of decisions, and a SHA-256 hash of the
H_cl trajectory (rounded to 10 decimals). They pin current behavior so later
phases can show exactly which change altered which result and why — they do
NOT assert correctness (several pinned behaviors are known discrepancies; see
audit/AUDIT.md, register items D1-D8 and N1).

Loading notes:
- main.py executes a full training run at module import, so it cannot be
  imported. We exec its source truncated just before the module-level script
  (the line ``SEED = 42``), which yields all class/function definitions
  without running anything (audit-only; legacy files are unmodified).
- data.py is import-safe and is loaded via importlib.

Golden values were generated on 2026-06-11 with Python 3.14.4 / numpy 2.4.4 /
torch 2.11.0 on darwin-arm64. To regenerate after an intentional change of
environment, run:

    pytest tests/audit/test_legacy_characterization.py -m legacy --no-header -q

and copy the values printed by the failing assertions, or re-run the
generation snippet embedded in each GOLDEN dict's construction (the run_*
helpers below are the generators).

Markers: @pytest.mark.legacy (registered in tests/audit/conftest.py).
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import pytest

# Force a non-interactive backend before the legacy sources import pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

# Both legacy modules call torch.set_num_interop_threads(1) at import
# (main.py:20, data.py:18); torch allows that call only once per process, so
# loading the second legacy module would raise RuntimeError. Apply the setting
# once here and make subsequent calls no-ops (thread counts do not affect the
# pinned values).
import torch  # noqa: E402

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.set_num_interop_threads = lambda n: None
torch.set_num_threads(1)

pytestmark = pytest.mark.legacy

REPO_ROOT = Path(__file__).resolve().parents[2]

# Episode seeds used for pinning, plus the auction time of the cancel variant.
SEEDS = (101, 202, 303)
CANCEL_AT = 135


# --------------------------------------------------------------------------
# Legacy module loading
# --------------------------------------------------------------------------

def _load_legacy_main():
    """Exec main.py truncated before its module-level training script."""
    src = (REPO_ROOT / "main.py").read_text()
    cut = src.index("\nSEED = 42")  # everything below is the training script
    mod = types.ModuleType("legacy_main_truncated")
    mod.__file__ = str(REPO_ROOT / "main.py")
    # dont_inherit: do not leak this test module's __future__ flags into the
    # legacy source (deferred annotations would break its dataclasses);
    # sys.modules registration lets dataclasses resolve the defining module.
    sys.modules[mod.__name__] = mod
    code = compile(src[:cut], "main.py[truncated]", "exec", flags=0, dont_inherit=True)
    exec(code, mod.__dict__)
    return mod.__dict__


def _load_legacy_data():
    spec = importlib.util.spec_from_file_location("legacy_data", REPO_ROOT / "data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def legacy_main():
    return _load_legacy_main()


@pytest.fixture(scope="module")
def legacy_data():
    return _load_legacy_data()


@pytest.fixture(scope="module")
def cat_mid_path():
    """Historical mid path exactly as data.py's __main__ builds it (D13)."""
    pd = pytest.importorskip("pandas")
    df = (
        pd.read_csv(REPO_ROOT / "data.csv", parse_dates=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )
    return df["CAT"].to_numpy(dtype=float)[0:120]


# --------------------------------------------------------------------------
# Env construction at the exact legacy instantiation values (ruling D6)
# --------------------------------------------------------------------------

def make_synthetic_env(ns):
    """MarketEmulator with the main.py:1417-1422 instantiation values."""
    return ns["MarketEmulator"](
        tau_op=120, tau_cl=150, T=150, I=100, V=30, L=12, Lc=12, La=12,
        L_max_auction=100, lambda_param=0.5, kappa=0.1, q=1.0, d=0.1,
        gamma=0.95, v_m=2.0, pareto_gamma=2.5, poisson_rate=1.0, alpha=0.01,
        seed=None, V_top_max=15.0, beta_a=2.0, beta_b=5.0, depth_decay=0.5,
        price_band_ticks=10.0, mid_rh_H=0.1, mid_rh_v0=0.02, mid_rh_theta=0.04,
        mid_rh_kappa=0.3, mid_rh_xi=0.3, mid_rh_rho=-0.7, sigma_mid=0.1,
    )


def make_historical_env(mod, mid_path):
    """MarketEmulator with the data.py:1386-1398 instantiation values."""
    return mod.MarketEmulator(
        tau_op=120, tau_cl=150, T=150, I=100, V=30, L=12, Lc=12, La=12,
        L_max_auction=100, lambda_param=0.5, kappa=0.1, q=1.0, d=0.1,
        gamma=0.99,  # legacy passes the RL discount here (AUDIT N2 / Q1)
        v_m=2.0, pareto_gamma=2.5, poisson_rate=60.0, alpha=0.01, seed=None,
        V_top_max=15.0, beta_a=2.0, beta_b=5.0, depth_decay=0.5,
        price_band_ticks=10.0, mid_price_path=mid_path,
    )


# --------------------------------------------------------------------------
# Fixed deterministic policy + episode runner (the golden generator)
# --------------------------------------------------------------------------

def run_episode(env, seed, cancel_at=None):
    """Run one episode under a fixed policy and return summary statistics.

    Policy: CLOB -> (v=5, delta=2) every step; auction -> (K=2.0,
    S = S_mid + 2*alpha, no cancel), except at ``cancel_at`` where the legacy
    cancel-all vector c[tau_op:t] = 1 is submitted (the exact expansion the
    legacy training loop uses, main.py:1758-1763).
    """
    s = env.reset(seed=seed)
    done = False
    total_r = 0.0
    n_steps = 0
    h_traj = [s["X3"]]
    while not done:
        if env.phase == "continuous":
            s, r, done = env.step((5.0, 2.0))
        else:
            t = int(s["time"])
            c_vec = [0] * (env.tau_cl + 1)
            if cancel_at is not None and t == cancel_at:
                c_vec[env.tau_op:t] = [1] * (t - env.tau_op)
            s, r, done = env.step((2.0, s["X10"] + 2 * env.alpha, c_vec))
        total_r += r
        n_steps += 1
        h_traj.append(s["X3"])
    h_hash = hashlib.sha256(
        ",".join(f"{h:.10f}" for h in h_traj).encode()
    ).hexdigest()
    return {
        "total_reward": float(total_r),
        "final_inventory": float(env.inventory),
        "n_steps": n_steps,
        "h_cl_hash": h_hash,
    }


# --------------------------------------------------------------------------
# Golden values (generated 2026-06-11; see module docstring)
# --------------------------------------------------------------------------

GOLDEN_SYNTHETIC = {
    101: {
        "total_reward": 9464.563799794452,
        "final_inventory": 12.130568967475348,
        "n_steps": 106,
        "h_cl_hash": "70c8f54bf8af517b736539ea4b6c7927ae1e10ec6a35e7589d6596a9cf99864d",
    },
    202: {
        "total_reward": 7869.943002944581,
        "final_inventory": 26.45004658559514,
        "n_steps": 115,
        "h_cl_hash": "c5d241985e654ec714823cf341208f3d27c307213ae724838d9874ca66751ed2",
    },
    303: {
        "total_reward": -2595.6562908058404,
        "final_inventory": 55.03602936817605,
        "n_steps": 119,
        "h_cl_hash": "921fd8ad444759f3d9f6d985684979b34d6da341e8a51e0c201c8a2933794851",
    },
}

GOLDEN_SYNTHETIC_CANCEL = {
    101: {
        "total_reward": 9718.759844344373,
        "final_inventory": 11.82753168762634,
        "n_steps": 106,
        "h_cl_hash": "5cee4da515f6bf66401b3479133676983f6909bf514db17eed4f74562e62d347",
    },
}

GOLDEN_HISTORICAL = {
    101: {
        "total_reward": 9325.9119370403,
        "final_inventory": -11.129643302846546,
        "n_steps": 149,
        "h_cl_hash": "199c33c94970f1514df3f171f6c0150e69f304a31702f1bd78fe62abfe573a11",
    },
    202: {
        "total_reward": 10270.778951037511,
        "final_inventory": -16.419149278297027,
        "n_steps": 149,
        "h_cl_hash": "36ff18d3bc4f53ffdb7b03a3c9546c99e03664f79e7fac43b7907220b031e2d4",
    },
    303: {
        "total_reward": 6421.644632207761,
        "final_inventory": 14.684469794030406,
        "n_steps": 149,
        "h_cl_hash": "9bb5b6f11ae192c141093e6f8d956c1f5cc6873f05696e561c515a417e645014",
    },
}

GOLDEN_HISTORICAL_CANCEL = {
    101: {
        "total_reward": 9850.266640160124,
        "final_inventory": -9.545686598820993,
        "n_steps": 149,
        "h_cl_hash": "393c7b5782495f50958a001a2464706ad5b6b53f4d57862eb5287fb97c6c62a9",
    },
}

REL_TOL = 1e-9


def assert_matches_golden(stats, golden):
    assert stats["n_steps"] == golden["n_steps"], stats
    assert math.isclose(
        stats["total_reward"], golden["total_reward"], rel_tol=REL_TOL
    ), stats
    assert math.isclose(
        stats["final_inventory"], golden["final_inventory"],
        rel_tol=REL_TOL, abs_tol=1e-9,
    ), stats
    assert stats["h_cl_hash"] == golden["h_cl_hash"], stats


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_synthetic_fixed_policy(legacy_main, seed):
    env = make_synthetic_env(legacy_main)
    stats = run_episode(env, seed)
    assert_matches_golden(stats, GOLDEN_SYNTHETIC[seed])


def test_synthetic_fixed_policy_with_cancel(legacy_main):
    env = make_synthetic_env(legacy_main)
    stats = run_episode(env, 101, cancel_at=CANCEL_AT)
    assert_matches_golden(stats, GOLDEN_SYNTHETIC_CANCEL[101])
    # The cancel must change the outcome relative to the no-cancel run
    # (pins the legacy same-step cancel effect documented as D1/D4).
    assert stats["h_cl_hash"] != GOLDEN_SYNTHETIC[101]["h_cl_hash"]


@pytest.mark.parametrize("seed", SEEDS)
def test_historical_fixed_policy(legacy_data, cat_mid_path, seed):
    env = make_historical_env(legacy_data, cat_mid_path)
    stats = run_episode(env, seed)
    assert_matches_golden(stats, GOLDEN_HISTORICAL[seed])


def test_historical_fixed_policy_with_cancel(legacy_data, cat_mid_path):
    env = make_historical_env(legacy_data, cat_mid_path)
    stats = run_episode(env, 101, cancel_at=CANCEL_AT)
    assert_matches_golden(stats, GOLDEN_HISTORICAL_CANCEL[101])


def test_historical_decision_count_pins_missing_t_n(legacy_data, cat_mid_path):
    """Pins AUDIT N1: 119 CLOB + 30 auction = 149 decisions (no action at t_n=119)."""
    env = make_historical_env(legacy_data, cat_mid_path)
    s = env.reset(seed=101)
    clob_steps = 0
    auction_steps = 0
    done = False
    while not done:
        if env.phase == "continuous":
            s, _, done = env.step((0.0, 1.0))
            clob_steps += 1
        else:
            s, _, done = env.step((0.0, s["X10"], [0] * (env.tau_cl + 1)))
            auction_steps += 1
    assert clob_steps == 119
    assert auction_steps == 30


def test_historical_mid_frozen_during_auction(legacy_data, cat_mid_path):
    """Pins the frozen-at-tau_op auction mid (paper-consistent; AUDIT A.12)."""
    env = make_historical_env(legacy_data, cat_mid_path)
    s = env.reset(seed=101)
    done = False
    while env.phase == "continuous" and not done:
        s, _, done = env.step((0.0, 1.0))
    mid_at_open = env.mid_price
    while not done:
        s, _, done = env.step((0.0, s["X10"], [0] * (env.tau_cl + 1)))
        assert env.mid_price == mid_at_open
