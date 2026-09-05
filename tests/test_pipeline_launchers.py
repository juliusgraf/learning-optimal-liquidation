"""Fast contract tests for shell-level provenance forwarding."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _child_calls(tmp_path: Path, script: str, *args: str) -> list[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.txt"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$PIPELINE_CALL_LOG"\n'
    )
    fake_bash.chmod(0o755)
    fake_python = fake_bin / "python3"
    fake_python.write_text(fake_bash.read_text())
    fake_python.chmod(0o755)
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nexit 0\n")
    fake_git.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["PIPELINE_CALL_LOG"] = str(call_log)
    result = subprocess.run(
        ["/bin/bash", f"scripts/{script}", *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return call_log.read_text().splitlines()


def _reproduce_child_calls(tmp_path: Path, *args: str) -> list[str]:
    return _child_calls(tmp_path, "reproduce_all.sh", *args)


def test_reproduce_all_forwards_seed_to_complete_output_generation(tmp_path):
    calls = _reproduce_child_calls(tmp_path, "--smoke", "--seed", "97")
    assert calls[-1].endswith("make_all_outputs.sh --seed 97 --require-complete")


def test_symbol_scoped_reproduction_does_not_claim_complete_matrix(tmp_path):
    calls = _reproduce_child_calls(tmp_path, "--seed=7", "--symbol", "MSFT")
    assert calls[-1].endswith("make_all_outputs.sh --seed 7")
    assert "--require-complete" not in calls[-1]


def test_multiseed_output_request_requires_two_distinct_seeds():
    result = subprocess.run(
        ["/bin/bash", "scripts/make_multiseed_outputs.sh", "--seeds", "42"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "at least two distinct" in result.stderr


def test_run_multiseed_requires_complete_publication_matrix(tmp_path):
    calls = _child_calls(
        tmp_path, "run_multiseed.sh", "--seeds", "9001 9002", "--smoke"
    )
    assert calls[-1].endswith(
        "--seeds 9001 9002 --jobs 1 --threads-per-job 2 --smoke"
    )


def test_run_multiseed_forwards_symbol_to_aggregate_validation(tmp_path):
    calls = _child_calls(
        tmp_path,
        "run_multiseed.sh",
        "--seeds=9001 9002",
        "--symbol",
        "MSFT",
        "--smoke",
    )
    assert calls[-1].endswith(
        "--seeds 9001 9002 --jobs 1 --threads-per-job 2 --smoke --symbol MSFT"
    )


def test_run_multiseed_default_is_full_five_seed_treatment_publication(tmp_path):
    calls = _child_calls(tmp_path, "run_multiseed.sh")
    assert len(calls) == 1
    assert "lmm.experiments.run_matrix" in calls[0]
    assert calls[0].endswith("--seeds 42 7 99 123 2024 --jobs 1 --threads-per-job 2")


def test_run_multiseed_parallel_workers_leave_shared_outputs_serial(tmp_path):
    calls = _child_calls(
        tmp_path,
        "run_multiseed.sh",
        "--seeds",
        "9001 9002",
        "--jobs=2",
        "--threads-per-job=1",
        "--smoke",
    )
    assert len(calls) == 1
    assert "lmm.experiments.run_matrix" in calls[0]
    assert calls[0].endswith("--seeds 9001 9002 --jobs 2 --threads-per-job 1 --smoke")


def test_multiseed_smoke_defaults_to_disjoint_nonpublication_seeds(tmp_path):
    calls = _child_calls(tmp_path, "run_multiseed.sh", "--smoke")
    assert calls[-1].endswith(
        "--seeds 9001 9002 --jobs 1 --threads-per-job 2 --smoke"
    )


def test_multiseed_smoke_rejects_canonical_seed_names():
    result = subprocess.run(
        [
            "/bin/bash",
            "scripts/run_multiseed.sh",
            "--smoke",
            "--seeds",
            "42 7",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "smoke runs must use noncanonical seeds" in result.stderr


def test_reproduce_all_can_defer_shared_output_generation(tmp_path):
    calls = _reproduce_child_calls(tmp_path, "--seed", "42", "--skip-output-generation")
    assert len(calls) == 8
    assert not any("make_all_outputs.sh" in call for call in calls)


def test_treatment_launcher_reuses_shaped_headline_instead_of_retraining_alias():
    script = (REPO / "scripts" / "run_synthetic_treatments.sh").read_text()
    arms = script.split("ARMS=(", 1)[1].split(")", 1)[0]
    assert "ablation_h_on_shaping_on" not in arms
    assert "ablation_h_on_shaping_off" in arms
    assert "canonical synthetic headline" in script
    assert (REPO / "configs/treatment/ablation_h_on_shaping_off.yaml").exists()


def test_full_publication_rejects_noncanonical_seed_subset():
    result = subprocess.run(
        [
            "/bin/bash",
            "scripts/run_multiseed.sh",
            "--seeds",
            "42 7",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "exactly the canonical seeds" in result.stderr


def test_publication_aggregation_rejects_symbol_scoping():
    result = subprocess.run(
        [
            "/bin/bash",
            "scripts/make_multiseed_outputs.sh",
            "--publication",
            "--symbol",
            "MSFT",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "complete five-ticker" in result.stderr


def test_full_publication_launcher_rejects_dirty_worktree(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = status ]; then echo ' M uncommitted.py'; else echo deadbeef; fi\n"
    )
    fake_git.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["/bin/bash", "scripts/run_multiseed.sh"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "clean git worktree" in result.stderr


def test_make_all_outputs_always_rebuilds_paired_differences():
    script = (REPO / "scripts" / "make_all_outputs.sh").read_text().splitlines()
    commands = [line.strip() for line in script if "policy_differences" in line]
    assert len(commands) == 2
    assert all(line.startswith("python3 -m") for line in commands)
    assert all("||" not in line and "[[" not in line for line in commands)


def test_make_all_outputs_complete_mode_requires_both_publication_settings():
    script = (REPO / "scripts" / "make_all_outputs.sh").read_text()
    assert "complete_settings=\"\"" in script
    assert "for required_setting in synthetic_rough_heston historical_sp500_midquotes" in script
    assert "missing complete publication setting" in script


def test_per_run_pipeline_writes_manifest_after_both_difference_artifacts():
    script = (REPO / "scripts" / "_common.sh").read_text()
    as_difference = script.index(
        "policy_differences --run-dir \"$run_dir\" --benchmark as"
    )
    twap_difference = script.index(
        "policy_differences --run-dir \"$run_dir\" --benchmark twap"
    )
    manifest = script.index("--write-completion-manifest \"$run_dir\"")
    assert as_difference < twap_difference < manifest
    assert "pipeline_complete.txt" not in script
