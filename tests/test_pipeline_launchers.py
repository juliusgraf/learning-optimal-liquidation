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
        tmp_path, "run_multiseed.sh", "--seeds", "42 7", "--smoke"
    )
    assert calls[-1].endswith(
        "make_multiseed_outputs.sh --seeds 42 7 --require-complete"
    )


def test_run_multiseed_forwards_symbol_to_aggregate_validation(tmp_path):
    calls = _child_calls(
        tmp_path,
        "run_multiseed.sh",
        "--seeds=42 7",
        "--symbol",
        "MSFT",
    )
    assert calls[-1].endswith(
        "make_multiseed_outputs.sh --seeds 42 7 --require-complete --symbol MSFT"
    )


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
