"""make_figures regenerates every figure from a committed fixture run dir.

No env stepping happens here — figures come from saved outputs only.
"""

from __future__ import annotations

from lmm.experiments import make_figures

PER_RUN_FIGURES = [
    "training_diagnostics",
    "regret_curve",
    "episode_anatomy",
    "benchmark_anatomy",
    "eval_distributions",
    "algorithm_comparison",
    "reward_decomposition",
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
    # The reward decomposition pools both algos + benchmarks for the setting and
    # writes a companion CSV of the plotted IQM/CI numbers alongside the figure.
    _assert_pdf_png(tmp_path, "reward_decomposition")
    csv_path = tmp_path / "reward_decomposition.csv"
    assert csv_path.exists() and csv_path.stat().st_size > 0


def test_convergence_curves_multiseed(fixture_run_dir, tmp_path):
    # Two complete run dirs of the same setting with distinct seeds exercise the
    # multi-seed convergence figure (aggregated IQM/CI over runs).
    import shutil

    run_dirs = []
    for seed in ("42", "99"):
        rd = tmp_path / f"run_seed{seed}"
        shutil.copytree(fixture_run_dir, rd)
        (rd / "seed.txt").write_text(seed + "\n")
        run_dirs.append(str(rd))
    out = tmp_path / "out"
    rc = make_figures.main(["--multiseed", "--run-dir", *run_dirs, "--out", str(out)])
    assert rc == 0
    _assert_pdf_png(out, "convergence_curves")
    # Cross-seed DQN regret curve (synthetic fixture => no ticker suffix).
    _assert_pdf_png(out, "regret_multiseed")


def test_default_out_is_run_figures_dir(fixture_run_dir, tmp_path):
    # Copy the fixture so the default <run>/figures dir lands in tmp.
    import shutil

    run = tmp_path / "run"
    shutil.copytree(fixture_run_dir, run)
    make_figures.main(["--run-dir", str(run)])
    assert (run / "figures" / "training_diagnostics.png").exists()
