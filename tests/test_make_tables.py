"""make_tables regenerates booktabs .tex + .csv from a committed fixture run dir.

Structural .tex lint always runs; an actual pdflatex compile runs only if a
LaTeX toolchain is available (else skipped).
"""

from __future__ import annotations

import shutil
import subprocess

import pandas as pd
import pytest

from lmm.experiments import make_tables

PER_RUN_TABLES = [
    "eval_summary_final",
    "params_generative",
    "params_midprice",
    "hyperparams_dqn",
]


def _make(out_dir, *run_dirs):
    rc = make_tables.main(["--run-dir", *map(str, run_dirs), "--out", str(out_dir)])
    assert rc == 0


def test_tables_exist_and_csv_parses(fixture_run_dir, tmp_path):
    _make(tmp_path, fixture_run_dir)
    for name in PER_RUN_TABLES:
        tex = tmp_path / f"{name}.tex"
        csv = tmp_path / f"{name}.csv"
        assert tex.exists() and tex.stat().st_size > 0
        assert csv.exists() and csv.stat().st_size > 0
        df = pd.read_csv(csv)
        assert df.shape[0] > 0 and df.shape[1] >= 2

    summary = pd.read_csv(tmp_path / "eval_summary_final.csv")
    assert "Metric" in summary.columns
    for col in ("Initial DQN", "AS", "TWAP", "DQN"):
        assert col in summary.columns
    params = pd.read_csv(tmp_path / "params_generative.csv")
    assert list(params.columns) == ["Symbol", "Value", "Comment"]
    assert len(params) == 33
    clock = params.loc[params["Symbol"] == "$[t]$", "Value"]
    assert len(clock) == 1 and clock.iloc[0] in ("minutes", "seconds")


def test_tex_structure(fixture_run_dir, tmp_path):
    _make(tmp_path, fixture_run_dir)
    for name in PER_RUN_TABLES:
        tex = (tmp_path / f"{name}.tex").read_text()
        for token in ("\\toprule", "\\midrule", "\\bottomrule", "\\caption", "\\label"):
            assert token in tex, f"{name}.tex missing {token}"
        assert tex.count("\\begin{table}") == 1
        assert tex.count("\\end{table}") == 1
        assert tex.count("\\begin{tabular}") == tex.count("\\end{tabular}") == 1
        # every body row has the same number of cells as the header
        body = [ln for ln in tex.splitlines() if ln.endswith("\\\\") and "multicolumn" not in ln]
        ncols = body[0].count("&")
        assert all(ln.count("&") == ncols for ln in body)


def test_multi_run_adds_ddpg_column(fixture_run_dir, fixture_run_dir_ddpg, tmp_path):
    _make(tmp_path, fixture_run_dir, fixture_run_dir_ddpg)
    summary = pd.read_csv(tmp_path / "eval_summary_final.csv")
    assert "DDPG" in summary.columns
    assert (tmp_path / "hyperparams_ddpg.tex").exists()


def test_tex_compiles_if_latex_available(fixture_run_dir, tmp_path):
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("no pdflatex available")
    _make(tmp_path, fixture_run_dir)
    for name in PER_RUN_TABLES:
        doc = tmp_path / f"doc_{name}.tex"
        doc.write_text(
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n\\usepackage{amssymb}\n\\usepackage{float}\n"
            "\\begin{document}\n"
            f"\\input{{{(tmp_path / name).as_posix()}.tex}}\n"
            "\\end{document}\n"
        )
        res = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", doc.name],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert res.returncode == 0, f"{name}.tex failed to compile:\n{res.stdout[-1500:]}"
