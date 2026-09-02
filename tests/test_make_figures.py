"""make_figures regenerates every figure from a committed fixture run dir.

No env stepping happens here — figures come from saved outputs only.
"""

from __future__ import annotations

import dataclasses

import yaml

from lmm.config import load_config, save_resolved
from lmm.experiments import make_figures

PER_RUN_FIGURES = [
    "training_diagnostics",
    "policy_difference_curve",
    "episode_anatomy",
    "benchmark_anatomy",
    "cancellation_strategy",
    "eval_distributions",
    "algorithm_comparison",
]


def _assert_pdf_png(out_dir, name):
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0, f"empty {p}"


def test_all_figures_from_single_run(fixture_run_dir, tmp_path):
    rc = make_figures.main(["--run-dir", str(fixture_run_dir), "--out", str(tmp_path)])
    assert rc == 0
    for name in PER_RUN_FIGURES:
        _assert_pdf_png(tmp_path, name)


def test_legacy_style_distribution_variant(fixture_run_dir, tmp_path):
    make_figures.main(["--run-dir", str(fixture_run_dir), "--out", str(tmp_path), "--legacy-style"])
    _assert_pdf_png(tmp_path, "eval_distributions_bars")


def test_algorithm_comparison_multi_run(fixture_run_dir, fixture_run_dir_ddpg, tmp_path):
    make_figures.main(
        ["--run-dir", str(fixture_run_dir), str(fixture_run_dir_ddpg), "--out", str(tmp_path)]
    )
    _assert_pdf_png(tmp_path, "algorithm_comparison")


def test_convergence_curves_multiseed(fixture_run_dir, tmp_path):
    # Two complete run dirs of the same setting with distinct seeds exercise the
    # multi-seed convergence figure (aggregated IQM/CI over runs).
    import shutil

    run_dirs = []
    for seed in ("42", "99"):
        rd = tmp_path / f"run_seed{seed}"
        shutil.copytree(fixture_run_dir, rd)
        (rd / "seed.txt").write_text(seed + "\n")
        cfg = load_config(rd / "config_resolved.yaml")
        cfg = dataclasses.replace(
            cfg,
            experiment=dataclasses.replace(
                cfg.experiment,
                master_seed=int(seed),
                seeds=(int(seed),),
            ),
        )
        save_resolved(cfg, rd / "config_resolved.yaml")
        metadata_path = rd / "eval" / "metadata.yaml"
        metadata = yaml.safe_load(metadata_path.read_text())
        metadata["master_seed"] = int(seed)
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
        run_dirs.append(str(rd))
    out = tmp_path / "out"
    rc = make_figures.main(["--multiseed", "--run-dir", *run_dirs, "--out", str(out)])
    assert rc == 0
    _assert_pdf_png(out, "convergence_curves")
    # Cross-seed DQN fixed-policy difference curve.
    _assert_pdf_png(out, "policy_difference_multiseed")
    # Run-level reward decomposition across the two seeds, with companion CSV.
    _assert_pdf_png(out, "reward_decomposition")
    csv_path = out / "reward_decomposition.csv"
    assert csv_path.exists() and csv_path.stat().st_size > 0


def test_cancellation_strategy_marks_cancel_events(fixture_run_dir, tmp_path):
    # The fixture DQN trace never cancels (all c_t = 0); inject two cancel-alls
    # so the figure also exercises the c_t = 1 marking path, then confirm the
    # standalone cancellation figure renders for the same policy/episode that
    # episode_anatomy uses.
    import shutil

    import pandas as pd

    run = tmp_path / "run"
    shutil.copytree(fixture_run_dir, run)
    trace = run / "eval" / "traces" / "dqn_ep0.csv"
    df = pd.read_csv(trace)
    auc_idx = df.index[df["phase"] == "auction"].to_list()
    assert len(auc_idx) >= 2, "fixture must have auction rows"
    df.loc[auc_idx[5], "act_cancel"] = 1
    df.loc[auc_idx[12], "act_cancel"] = 1
    df.to_csv(trace, index=False)

    out = tmp_path / "out"
    make_figures.fig_cancellation_strategy(run, out, policy="dqn", episode=0)
    _assert_pdf_png(out, "cancellation_strategy")


def test_default_out_is_run_figures_dir(fixture_run_dir, tmp_path):
    # Copy the fixture so the default <run>/figures dir lands in tmp.
    import shutil

    run = tmp_path / "run"
    shutil.copytree(fixture_run_dir, run)
    make_figures.main(["--run-dir", str(run)])
    assert (run / "figures" / "training_diagnostics.png").exists()


def test_missing_requested_trace_returns_nonzero(fixture_run_dir, tmp_path):
    (fixture_run_dir / "eval" / "traces" / "dqn_ep0.csv").unlink()
    rc = make_figures.main(
        ["--run-dir", str(fixture_run_dir), "--out", str(tmp_path / "out")]
    )
    assert rc == 1


def test_no_auction_run_omits_cancellation_strategy(fixture_run_dir, tmp_path):
    cfg = load_config(
        "configs/base.yaml",
        "configs/synthetic_rough_heston.yaml",
        "configs/algo/dqn.yaml",
        "configs/treatment/no_auction.yaml",
    )
    save_resolved(cfg, fixture_run_dir / "config_resolved.yaml")
    out = tmp_path / "out"

    rc = make_figures.main(
        ["--run-dir", str(fixture_run_dir), "--out", str(out)]
    )

    assert rc == 0
    assert not (out / "cancellation_strategy.pdf").exists()
    assert not (out / "cancellation_strategy.png").exists()
    _assert_pdf_png(out, "episode_anatomy")
