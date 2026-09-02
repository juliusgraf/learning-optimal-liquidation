"""make_tables regenerates booktabs .tex + .csv from a committed fixture run dir.

Structural .tex lint always runs; an actual pdflatex compile runs only if a
LaTeX toolchain is available (else skipped).
"""

from __future__ import annotations

import shutil
import subprocess

import pandas as pd
import pytest
import yaml

from lmm.experiments import make_tables, policy_differences

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
    for col in ("Initial policy", "AS", "TWAP", "DQN"):
        assert col in summary.columns
    summary_tex = (tmp_path / "eval_summary_final.tex").read_text()
    assert "3 episodes" in summary_tex
    assert "100 episodes" not in summary_tex
    assert "Mean Centered Economic Evaluation Return (Diagnostic)" in set(
        summary["Metric"]
    )
    assert "Shaped Return" not in summary_tex
    params = pd.read_csv(tmp_path / "params_generative.csv")
    assert list(params.columns) == ["Symbol", "Value", "Comment"]
    assert len(params) == 36
    assert params.loc[params["Symbol"] == "$V$", "Value"].item() == "30"
    assert params.loc[params["Symbol"] == "$V_{\\max}$", "Value"].item() == "30"
    clock = params.loc[params["Symbol"] == "$[t]$", "Value"]
    assert len(clock) == 1 and clock.iloc[0] in ("minutes", "seconds")
    assert params.loc[params["Symbol"] == "$B_\\infty$", "Value"].item() == "150"
    assert params.loc[params["Symbol"] == "$B_{\\mathrm{max}}$", "Value"].item() == "10"
    assert "$B_{\\mathrm{loc}}$" not in set(params["Symbol"])


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


def test_generation_failure_returns_nonzero(fixture_run_dir, tmp_path):
    records = fixture_run_dir / "eval" / "records.csv"
    df = pd.read_csv(records).drop(columns=["risk_adjusted_pnl"])
    df.to_csv(records, index=False)

    rc = make_tables.main(
        ["--run-dir", str(fixture_run_dir), "--out", str(tmp_path / "out")]
    )
    assert rc == 1


def test_multiseed_request_requires_two_seeds(fixture_run_dir, tmp_path):
    rc = make_tables.main(
        [
            "--multiseed",
            "--run-dir",
            str(fixture_run_dir),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1


def test_contradictory_saved_seed_is_rejected(fixture_run_dir, tmp_path):
    (fixture_run_dir / "seed.txt").write_text("97\n")
    with pytest.raises(ValueError, match="disagrees with.*experiment.master_seed"):
        make_tables.main(
            ["--run-dir", str(fixture_run_dir), "--out", str(tmp_path / "out")]
        )


def test_metadata_schema_mismatch_is_rejected_by_readers(fixture_run_dir, tmp_path):
    metadata_path = fixture_run_dir / "eval" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    metadata["artifact_schema_version"] = 9
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    with pytest.raises(ValueError, match="active schema 11"):
        make_tables.main(
            ["--run-dir", str(fixture_run_dir), "--out", str(tmp_path / "out")]
        )
    with pytest.raises(SystemExit, match="active schema 11"):
        policy_differences.main(
            ["--run-dir", str(fixture_run_dir), "--benchmark", "as"]
        )


def test_duplicate_run_identity_is_rejected(fixture_run_dir, tmp_path):
    with pytest.raises(ValueError, match="duplicate run identity"):
        make_tables.main(
            [
                "--run-dir",
                str(fixture_run_dir),
                str(fixture_run_dir),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_uniformly_truncated_records_are_rejected(fixture_run_dir, tmp_path):
    records_path = fixture_run_dir / "eval" / "records.csv"
    records = pd.read_csv(records_path)
    records[records["episode"] < records["episode"].max()].to_csv(
        records_path, index=False
    )

    with pytest.raises(ValueError, match="complete metadata-declared matrix"):
        make_tables.main(
            ["--run-dir", str(fixture_run_dir), "--out", str(tmp_path / "out")]
        )
    with pytest.raises(SystemExit, match="complete metadata-declared matrix"):
        policy_differences.main(
            ["--run-dir", str(fixture_run_dir), "--benchmark", "as"]
        )


@pytest.mark.parametrize("corruption", ["policies", "seed_list", "record_seed"])
def test_records_metadata_crn_contract_is_enforced(
    fixture_run_dir, tmp_path, corruption
):
    metadata_path = fixture_run_dir / "eval" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    records_path = fixture_run_dir / "eval" / "records.csv"
    records = pd.read_csv(records_path)
    if corruption == "policies":
        metadata["policies"] = metadata["policies"][:-1]
    elif corruption == "seed_list":
        metadata["evaluation_episode_seeds"] = metadata[
            "evaluation_episode_seeds"
        ][:-1]
    else:
        records.loc[0, "env_seed"] = int(records.loc[0, "env_seed"]) + 1
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    records.to_csv(records_path, index=False)

    with pytest.raises(ValueError):
        make_tables.main(
            ["--run-dir", str(fixture_run_dir), "--out", str(tmp_path / "out")]
        )


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
