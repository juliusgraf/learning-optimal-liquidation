"""Smoke-run the Phase 7 launcher scripts (slow integration; CI uses --smoke).

These actually train tiny models, so they are marked ``slow`` (deselect with
``-m 'not slow'``). They write into results/ (gitignored) under namespaced
seeds to avoid clobbering real runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]
SEED = "97"  # namespaced so script smokes never clobber real runs

# (script, setting, algo, symbol-or-None)
SCRIPTS = [
    ("run_synthetic_dqn.sh", "synthetic_rough_heston", "dqn", None),
    ("run_synthetic_ddpg.sh", "synthetic_rough_heston", "ddpg", None),
    ("run_synthetic_td3.sh", "synthetic_rough_heston", "td3", None),
    ("run_synthetic_sac.sh", "synthetic_rough_heston", "sac", None),
    ("run_historical_dqn.sh", "historical_sp500_midquotes", "dqn", "MSFT"),
    ("run_historical_ddpg.sh", "historical_sp500_midquotes", "ddpg", "MSFT"),
    ("run_historical_td3.sh", "historical_sp500_midquotes", "td3", "MSFT"),
    ("run_historical_sac.sh", "historical_sp500_midquotes", "sac", "MSFT"),
]


def _run(script: str, extra: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", f"scripts/{script}", "--smoke", "--seed", SEED, *extra],
        cwd=REPO, capture_output=True, text=True,
    )


@pytest.mark.parametrize("script,setting,algo,symbol", SCRIPTS)
def test_run_script_smoke(script, setting, algo, symbol):
    extra = ["--symbol", symbol] if symbol else []
    res = _run(script, extra)
    assert res.returncode == 0, f"{script} failed:\n{res.stderr[-2000:]}"
    name = f"{algo}_{symbol}_seed{SEED}" if symbol else f"{algo}_seed{SEED}"
    rd = REPO / "results" / "revision_v9" / setting / name
    assert (rd / "checkpoints" / "final.pt").exists()
    assert (rd / "eval" / "records.csv").exists()
    # Traces use the resolved learned-policy label.
    assert (rd / "eval" / "traces" / f"{algo}_ep0.csv").exists()
    assert (rd / "eval" / "policy_difference_as.csv").exists()


def test_acceptance_run_then_make_all_outputs():
    """Acceptance: run_synthetic_dqn.sh --smoke, then make_all_outputs.sh, and
    confirm the full artifact set for that run plus the combined outputs."""
    seed = "96"
    res = subprocess.run(
        ["bash", "scripts/run_synthetic_dqn.sh", "--smoke", "--seed", seed],
        cwd=REPO, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    res = subprocess.run(
        ["bash", "scripts/make_all_outputs.sh"], cwd=REPO, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr[-2000:]

    rd = REPO / "results" / "revision_v9" / "synthetic_rough_heston" / f"dqn_seed{seed}"
    for name in ("training_diagnostics", "episode_anatomy", "eval_distributions"):
        assert (rd / "figures" / f"{name}.pdf").exists()
        assert (rd / "figures" / f"{name}.png").exists()
    for name in ("eval_summary_final", "params_generative"):
        assert (rd / "tables" / f"{name}.tex").exists()
        assert (rd / "tables" / f"{name}.csv").exists()

    combined = REPO / "results" / "revision_v9" / "synthetic_rough_heston" / "_combined"
    assert (combined / "figures" / "algorithm_comparison.png").exists()
    assert (combined / "tables" / "eval_summary_final.csv").exists()
