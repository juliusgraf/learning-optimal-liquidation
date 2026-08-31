"""Cross-seed IQM/bootstrap-CI aggregate tables (multi-seed reporting).

Builds the aggregate tables from RunInfo objects assembled in-memory (one per
(algo, seed[, symbol]) run, each carrying a synthetic eval-records frame), so the
test is fast and needs no on-disk fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lmm.experiments import make_figures as F
from lmm.experiments import tables as T
from lmm.experiments.plotting import RunInfo

_BENCH = {"as": 10000.0, "twap": 6000.0, "initial": -3000.0}


def _records(learned_mean: float, *, n: int = 8) -> pd.DataFrame:
    """A tiny eval-records frame: the learned policy (labelled 'dqn') plus the
    benchmarks, with CRN-shared environment seeds and the revised outcome
    schema (currency and basis-point forms)."""
    rows = []
    for ep in range(n):
        es = 1000 + ep
        value = learned_mean + ep
        rows.append(
            dict(
                policy="dqn",
                episode=ep,
                env_seed=es,
                pnl=value + 50.0,
                risk_adjusted_pnl=value,
                risk_adjusted_pnl_bps=value / 100.0,
                economic_objective=value,
                liquidation_pnl_gross=value + 50.0,
                return_undisc=value - 1_000_000.0,
            )
        )
        for pol, base in _BENCH.items():
            value = base + ep
            rows.append(
                dict(
                    policy=pol,
                    episode=ep,
                    env_seed=es,
                    pnl=value + 50.0,
                    risk_adjusted_pnl=value,
                    risk_adjusted_pnl_bps=value / 100.0,
                    economic_objective=value,
                    liquidation_pnl_gross=value + 50.0,
                    return_undisc=value - 1_000_000.0,
                )
            )
    return pd.DataFrame(rows)


def _run(setting, algo, seed, learned_mean, symbol=None) -> RunInfo:
    records = _records(learned_mean)
    records.loc[records["policy"] == "dqn", "policy"] = algo
    return RunInfo(
        run_dir=Path(f"/x/{algo}_{symbol}_{seed}"),
        cfg=None,  # the multiseed builders never read cfg
        setting=setting,
        algo=algo,
        symbol=symbol,
        records=records,
        metadata={},
        seed=seed,
    )


def test_eval_summary_multiseed_synthetic(tmp_path):
    algos = {"dqn": 12000.0, "ddpg": 20000.0, "td3": 18000.0, "sac": 22000.0}
    runs = []
    for seed, bump in ((42, 0.0), (7, 500.0), (99, -500.0)):
        for a, m in algos.items():
            runs.append(_run("synthetic_rough_heston", a, seed, m + bump))

    table = T.build_eval_summary_multiseed(runs)
    assert any("DDPG" in c for c in table.columns)

    labels = [r.label for r in table.rows]
    assert any(lab.startswith("IQM Risk-adjusted PnL") for lab in labels)

    seeds_row = next(r for r in table.rows if r.label == "Seeds (n)")
    assert all(v == 3 for v in seeds_row.cells)  # 3 seeds per column

    iqm_row = next(r for r in table.rows if r.label.startswith("IQM Risk-adjusted PnL"))
    for cell in iqm_row.cells:
        pt, lo, hi = cell
        assert np.isfinite([pt, lo, hi]).all() and lo <= pt <= hi

    paths = T.write_table(table, tmp_path, "eval_summary_multiseed")
    assert all(p.exists() for p in paths)
    assert "money_ci" not in (tmp_path / "eval_summary_multiseed.csv").read_text()  # rendered, not raw


def test_multiseed_table_uses_paired_seed_level_risk_adjusted_pnl_differences():
    runs = []
    for seed in (1, 2, 3):
        run = _run("synthetic_rough_heston", "dqn", seed, 0.0)
        records = run.records.copy()
        records["risk_adjusted_pnl"] = 0.0
        records.loc[records["policy"] == "dqn", "risk_adjusted_pnl"] = 2.0 + seed
        records.loc[records["policy"] == "as", "risk_adjusted_pnl"] = 0.01
        runs.append(RunInfo(**{**run.__dict__, "records": records}))

    table = T.build_eval_summary_multiseed(runs)
    assert any(row.label.startswith("IQM Risk-adjusted PnL") for row in table.rows)
    header = next(row for row in table.rows if row.fmt == "header")
    assert "Paired Seed-level Risk-adjusted PnL Difference" in header.label
    assert "Currency Units" in header.label
    vs_as = next(row for row in table.rows if row.label == "vs AS")
    assert vs_as.fmt == "money_ci"
    dqn_idx = table.columns.index("DQN")
    expected = T.stats.iqm_ci(np.array([2.99, 3.99, 4.99]), rng=0)
    assert vs_as.cells[dqn_idx] == pytest.approx(expected)
    assert "paired seed-level differences" in table.caption


def test_multiseed_table_rejects_missing_revised_primary_outcome():
    run = _run("synthetic_rough_heston", "dqn", 1, 0.0)
    stale = run.records.drop(columns=["risk_adjusted_pnl"])
    with pytest.raises(KeyError, match="risk_adjusted_pnl"):
        T.build_eval_summary_multiseed(
            [RunInfo(**{**run.__dict__, "records": stale})]
        )


def test_figure_requires_revised_primary_outcome():
    all_metrics = pd.DataFrame(
        {
            "risk_adjusted_pnl": [0.5],
            "economic_objective": [1.0],
            "liquidation_pnl_gross": [2.0],
            "return_undisc": [3.0],
        }
    )
    assert F._primary_col(all_metrics) == "risk_adjusted_pnl"
    assert "Risk-adjusted PnL" in F._outcome_label("risk_adjusted_pnl")
    with pytest.raises(KeyError, match="risk_adjusted_pnl"):
        F._primary_col(all_metrics.drop(columns=["risk_adjusted_pnl"]))


def test_historical_results_multiseed(tmp_path):
    runs = []
    for seed in (42, 7):
        for sym, base in (("MSFT", 15000.0), ("GOOGL", 18000.0)):
            for a, m in (("dqn", 0.0), ("ddpg", 3000.0), ("td3", 2000.0), ("sac", 5000.0)):
                runs.append(_run("historical_sp500", a, seed, base + m, symbol=sym))

    table = T.build_historical_results_multiseed(runs)
    row_labels = [r.label for r in table.rows]
    assert "MSFT" in row_labels and "GOOGL" in row_labels
    assert any("All tickers" in lab for lab in row_labels)
    # the pooled aggregate row carries IQM+CI tuples
    agg = next(r for r in table.rows if r.label.startswith("All tickers"))
    for cell in agg.cells:
        pt, lo, hi = cell
        assert np.isfinite([pt, lo, hi]).all() and lo <= pt <= hi
    paired = next(r for r in table.rows if r.label == "MSFT vs AS")
    assert paired.cells[:2] == [None, None]
    for cell in paired.cells[2:]:
        pt, lo, hi = cell
        assert np.isfinite([pt, lo, hi]).all() and lo <= pt <= hi

    paths = T.write_table(table, tmp_path, "dqn_results_multiseed")
    assert all(p.exists() for p in paths)

    bps = T.build_historical_results_multiseed(
        runs, metric=T.NORMALIZED_PRIMARY_COL
    )
    assert "basis points" in bps.caption
    assert bps.rows[-1].fmt == "num2_ci"
