"""Cross-seed IQM/bootstrap-CI aggregate tables (multi-seed reporting).

Builds the aggregate tables from RunInfo objects assembled in-memory (one per
(algo, seed[, symbol]) run, each carrying a synthetic eval-records frame), so the
test is fast and needs no on-disk fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lmm.experiments import tables as T
from lmm.experiments.plotting import RunInfo

_BENCH = {"as": 10000.0, "twap": 6000.0, "initial": -3000.0}


def _records(learned_mean: float, *, n: int = 8) -> pd.DataFrame:
    """A tiny eval-records frame: the learned policy (labelled 'dqn') plus the
    benchmarks, CRN-shared env seed per episode."""
    rows = []
    for ep in range(n):
        es = 1000 + ep
        rows.append(dict(policy="dqn", episode=ep, env_seed=es, return_undisc=learned_mean + ep))
        for pol, base in _BENCH.items():
            rows.append(dict(policy=pol, episode=ep, env_seed=es, return_undisc=base + ep))
    return pd.DataFrame(rows)


def _run(setting, algo, seed, learned_mean, symbol=None) -> RunInfo:
    return RunInfo(
        run_dir=Path(f"/x/{algo}_{symbol}_{seed}"),
        cfg=None,  # the multiseed builders never read cfg
        setting=setting,
        algo=algo,
        symbol=symbol,
        records=_records(learned_mean),
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
    assert any(lab.startswith("IQM Return") for lab in labels)

    seeds_row = next(r for r in table.rows if r.label == "Seeds (n)")
    assert all(v == 3 for v in seeds_row.cells)  # 3 seeds per column

    iqm_row = next(r for r in table.rows if r.label.startswith("IQM Return"))
    for cell in iqm_row.cells:
        pt, lo, hi = cell
        assert np.isfinite([pt, lo, hi]).all() and lo <= pt <= hi

    paths = T.write_table(table, tmp_path, "eval_summary_multiseed")
    assert all(p.exists() for p in paths)
    assert "money_ci" not in (tmp_path / "eval_summary_multiseed.csv").read_text()  # rendered, not raw


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

    paths = T.write_table(table, tmp_path, "dqn_results_multiseed")
    assert all(p.exists() for p in paths)
