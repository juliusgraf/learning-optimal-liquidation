"""Selective v20 replacement: exact scope, safe staging, and preserved provenance."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from lmm.config import load_config, price_generator_identity
from lmm.experiments import replace_synthetic as R
from lmm.experiments import retained_history as H
from lmm.experiments.run_matrix import Job

REPO = Path(__file__).resolve().parents[1]


def put(path, text="old contents\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def saved_campaign(tmp_path):
    repo = tmp_path / "repo"
    main = repo / "results" / H.ROOT_NAMES[0]
    historical = main / H.HISTORICAL / "dqn_MSFT_seed42"
    put(historical / "config_resolved.yaml")
    put(historical / "git_sha.txt", "old-source-dirty\n")
    put(historical / "logs/run.log")
    put(historical / "pipeline_complete.json", "{}")
    synthetic = main / "synthetic_rough_heston/dqn_seed42"
    put(synthetic / "config_resolved.yaml")
    put(synthetic / "pipeline_complete.json", "{}")
    put(synthetic / "git_sha.txt", "old-source\n")
    put(main / "_forecasts" / H.HISTORICAL / "forecast_overlay.yaml", "{}")
    put(main / "_forecasts/synthetic_rough_heston/forecast_overlay.yaml", "{}")
    put(main / "_publication/manifest.json", "{}")
    put(main / "_provenance/source_attestation.json", "{}")
    put(main / "_orchestration/synthetic__dqn_seed42.log")
    put(main / "_orchestration/MSFT__dqn_seed42.log")
    for name in H.ROOT_NAMES[1:]:
        (repo / "results" / name).mkdir(parents=True)
    original = {}
    for path in (historical, synthetic):
        original[path.relative_to(repo).as_posix()] = dict(
            config_sha256=H.digest(path / "config_resolved.yaml"),
            completion_sha256=H.digest(path / "pipeline_complete.json"),
            git_revision=(path / "git_sha.txt").read_text().strip(),
        )
    baseline = dict(
        schema="v20-synthetic-replacement-baseline-v1",
        original_runs=original,
        historical_runs={
            historical.relative_to(repo).as_posix(): dict(
                **original[historical.relative_to(repo).as_posix()],
                tree_sha256=H.tree_digest(historical),
            )
        },
        historical_forecast_tree_sha256=H.tree_digest(
            main / "_forecasts" / H.HISTORICAL
        ),
        prior_report_sha256=H.digest(main / "_publication/manifest.json"),
        prior_report_source_attestation=dict(
            sha256=H.digest(main / "_provenance/source_attestation.json"),
            disclosure="Original recorded source disclosure.",
        ),
    )
    put(repo / H.BASELINE, json.dumps(baseline))
    put(repo / H.MESH_SOURCE, (REPO / H.MESH_SOURCE).read_text())
    return repo, main, historical, synthetic, baseline


def test_exact_320_replacements_and_three_original_roots():
    jobs = R.replacement_jobs()
    assert len(jobs) == len({j.name for j in jobs}) == 320
    assert [sum(j.root_name == n for j in jobs) for n in H.ROOT_NAMES] == [240, 40, 40]
    assert all(j.job.block not in ("MSFT", "JPM", "PG", "GOOGL", "CAT") for j in jobs)
    baseline = R.load_baseline(REPO)
    assert len(baseline["original_runs"]) == 520
    assert len(baseline["historical_runs"]) == 200


def test_shell_dry_run_is_read_only_and_explicit():
    root = REPO / "results/revision_v20"
    before = (root / H.STATE).exists()
    result = subprocess.run(
        [
            "bash",
            "scripts/run_multiseed.sh",
            "--replace-synthetic",
            "--jobs",
            "10",
            "--threads-per-job",
            "1",
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["synthetic_runs"] == 320 and plan["retained_historical_runs"] == 200
    assert plan["max_step_minutes"] == 0.25
    assert plan["workers"] == 10 and plan["threads_per_worker"] == 1
    assert (root / H.STATE).exists() == before
    assert not any("v21" in x for x in plan["roots"])


@pytest.mark.parametrize("flag", ["--smoke", "--legacy-clearing", "--symbol=MSFT"])
def test_shell_rejects_incompatible_replacement_modes(flag):
    result = subprocess.run(
        ["bash", "scripts/run_multiseed.sh", "--replace-synthetic", "--dry-run", flag],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2


def test_archive_preserves_history_and_is_restartable(saved_campaign):
    repo, main, historical, synthetic, baseline = saved_campaign
    history_hash = H.tree_digest(historical)
    log_hash = H.digest(main / "_orchestration/MSFT__dqn_seed42.log")
    state = R.initialize_state(repo, baseline, "new-source")
    assert synthetic.exists()  # preparation itself has not moved anything
    # Simulate interruption after rename but before recording the completed move.
    first = state["moves"][0]
    destination = repo / first["destination"]
    destination.parent.mkdir(parents=True)
    (repo / first["source"]).rename(destination)
    R.complete_archive(repo, state)
    assert not synthetic.exists()
    assert (main / H.ARCHIVE / "synthetic_rough_heston/dqn_seed42").exists()
    assert H.tree_digest(historical) == history_hash
    assert H.digest(main / "_orchestration/MSFT__dqn_seed42.log") == log_hash
    R.complete_archive(repo, state)
    retained = H.RetainedHistory.load(repo, main)
    retained.revalidate()
    assert retained.validate_run(historical) == "old-source-dirty"
    assert H.digest(main / H.MESH) == H.digest(repo / H.MESH_SOURCE)
    with pytest.raises(ValueError, match="another source/config"):
        R.initialize_state(repo, baseline, "different-source")


def test_retained_history_rejects_any_changed_log_or_forecast(saved_campaign):
    repo, main, historical, _, baseline = saved_campaign
    state = R.initialize_state(repo, baseline, "new-source")
    R.complete_archive(repo, state)
    retained = H.RetainedHistory.load(repo, main)
    retained.revalidate()
    put(historical / "logs/run.log", "tampered log")
    with pytest.raises(ValueError, match="historical artifacts changed"):
        retained.revalidate()
    put(historical / "logs/run.log")
    put(main / "_forecasts" / H.HISTORICAL / "forecast_overlay.yaml", "tampered fit")
    with pytest.raises(ValueError, match="forecast artifacts changed"):
        retained.revalidate()


def test_changed_original_campaign_fails_before_archive(saved_campaign):
    repo, main, historical, synthetic, baseline = saved_campaign
    put(synthetic / "config_resolved.yaml", "changed")
    with pytest.raises(ValueError, match="changed since preparation"):
        R.initialize_state(repo, baseline, "new-source")
    assert synthetic.exists() and historical.exists()
    assert not (main / H.STATE).exists()


def test_restart_journal_cannot_archive_historical_runs(saved_campaign):
    repo, main, historical, _, baseline = saved_campaign
    state = R.initialize_state(repo, baseline, "new-source")
    state["moves"].append(
        dict(
            source=historical.relative_to(repo).as_posix(),
            destination=(main / H.ARCHIVE / historical.relative_to(main))
            .relative_to(repo)
            .as_posix(),
            done=False,
        )
    )
    with pytest.raises(ValueError, match="outside synthetic scope"):
        R.complete_archive(repo, state)
    assert historical.exists()


def test_partial_attempt_archived_but_completed_attempt_reused(saved_campaign):
    repo, main, _, synthetic, baseline = saved_campaign
    state = R.initialize_state(repo, baseline, "new-source")
    R.complete_archive(repo, state)
    job = R.ReplacementJob(H.ROOT_NAMES[0], Job(42, "dqn", "synthetic"))
    put(synthetic / "metrics.csv", "partial new training")
    R.archive_partial_attempts(repo, [job])
    assert not synthetic.exists()
    assert (
        len(
            list(
                (main / H.ARCHIVE / "interrupted_attempts").glob(
                    "*/synthetic_rough_heston/dqn_seed42/metrics.csv"
                )
            )
        )
        == 1
    )
    put(synthetic / "pipeline_complete.json", "{}")
    R.archive_partial_attempts(repo, [job])
    assert synthetic.exists()


@pytest.mark.parametrize(
    "block",
    ["synthetic", *R.ARMS, "mechanism_economic_cashflow", "mechanism_economic_dense_h"],
)
def test_worker_applies_mesh_before_fitted_forecast(tmp_path, block):
    cfg = load_config(
        REPO / "configs/base.yaml",
        REPO / "configs/synthetic_rough_heston.yaml",
        REPO / H.MESH_SOURCE,
    )
    put(tmp_path / H.MESH, (REPO / H.MESH_SOURCE).read_text())
    overlay = tmp_path / "_forecasts/synthetic_rough_heston/forecast_overlay.yaml"
    put(
        overlay,
        yaml.safe_dump(
            dict(
                algo1=dict(
                    clob_forecast_weights=[0.1, 0.2, 0.3, 0.4],
                    clob_forecast_mechanism="max_volume_v2",
                    clob_forecast_price_generator=price_generator_identity(cfg),
                )
            )
        ),
    )
    fake = put(
        tmp_path / "record-python",
        f"#!{sys.executable}\nimport json,os,sys\n"
        'with open(os.environ["CALL_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n',
    )
    fake.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        LMM_PYTHON=str(fake),
        LMM_RESULTS_ROOT=str(tmp_path),
        LMM_CLEARING_MECHANISM="max_volume_v2",
        LMM_FORECAST_ROOT=str(tmp_path / "_forecasts"),
        CALL_LOG=str(log),
    )
    subprocess.run(
        ["bash", "scripts/_run_matrix_job.sh", "42", "dqn", block, "0"],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
    )
    command = json.loads(log.read_text().splitlines()[0])
    configs = [command[i + 1] for i, v in enumerate(command) if v == "--config"]
    overrides = [command[i + 1] for i, v in enumerate(command) if v == "-o"]
    actual = load_config(*configs, overrides=overrides)
    assert actual.midprice.rough_heston.rough_heston_max_step_minutes == 0.25
    assert actual.algo1.clob_forecast_weights == (0.1, 0.2, 0.3, 0.4)
    assert actual.algo1.clob_forecast_price_generator == price_generator_identity(
        actual
    )
    assert configs.index(str(tmp_path / H.MESH)) < configs.index(str(overlay))
    assert actual.experiment.results_root == tmp_path
    assert actual.experiment.episodes == 800 and actual.rl.test_size == 100


def test_real_refined_forecast_smoke_round_trip(tmp_path, monkeypatch):
    from lmm.experiments import clearing_campaign as C

    original_check_output = subprocess.check_output

    def check_output(command, **kwargs):
        # The real bounded fit needs a source identity, but this test does not
        # test Git itself. Source archives have no .git directory; keep the
        # production gate intact and provide identity only for this fixture.
        if command == ['git', 'rev-parse', 'HEAD'] and kwargs.get('cwd') == REPO:
            return 'a' * 40 + '\n'
        return original_check_output(command, **kwargs)

    monkeypatch.setattr(C.subprocess, 'check_output', check_output)
    put(tmp_path / H.MESH, (REPO / H.MESH_SOURCE).read_text())
    C.prepare_forecast(REPO, tmp_path, "synthetic_rough_heston", smoke=True)
    overlay = C.validate_forecast_fit(
        REPO, tmp_path, "synthetic_rough_heston", smoke=True
    )
    cfg = load_config(
        REPO / "configs/base.yaml",
        REPO / "configs/synthetic_rough_heston.yaml",
        REPO / "configs/clearing/max_volume_v2.yaml",
        tmp_path / H.MESH,
        overlay,
    )
    assert cfg.algo1.clob_forecast_price_generator == price_generator_identity(cfg)
    assert cfg.midprice.rough_heston.rough_heston_max_step_minutes == 0.25
    assert C.refinement_configs(tmp_path, H.HISTORICAL) == []


@pytest.mark.parametrize("training_code", [0, 1])
def test_orchestration_fits_then_trains_then_reports_only_on_success(
    saved_campaign, monkeypatch, training_code
):
    from lmm.experiments import publication
    from lmm.experiments import cashflow_comparison
    from types import SimpleNamespace

    repo, main, historical, _, baseline = saved_campaign
    for name in ("base.yaml", "synthetic_rough_heston.yaml"):
        put(repo / "configs" / name, (REPO / "configs" / name).read_text())
    monkeypatch.setattr(R, "load_baseline", lambda _: baseline)
    job = R.ReplacementJob(H.ROOT_NAMES[0], Job(42, "dqn", "synthetic"))
    monkeypatch.setattr(R, "replacement_jobs", lambda: [job])
    monkeypatch.setattr(publication, "_clean_head_sha", lambda: "new-source")
    monkeypatch.setattr(R, "validate_forecast_fit", lambda *a: None)
    events = []
    history = H.tree_digest(historical)

    def queue(jobs, **kwargs):
        is_forecast = isinstance(jobs[0], R.ForecastJob)
        events.append("forecast" if is_forecast else "training")
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["workers"] == (1 if is_forecast else 10)
        assert kwargs["env"]["LMM_FORECAST_ROOT"] == str(main / "_forecasts")
        assert (main / H.ARCHIVE / "synthetic_rough_heston/dqn_seed42").exists()
        assert H.digest(main / H.MESH) == H.digest(repo / H.MESH_SOURCE)
        return 0 if is_forecast else training_code

    def report(command, **kwargs):
        events.append(command[2])
        assert kwargs["check"]
        if command[2] == "lmm.experiments.cashflow_comparison":
            # Exercise its actual CLI parser, without reading/training policies.
            assert cashflow_comparison.main([*command[3:], "--dry-run"]) == 0
        else:
            assert command[2] == "lmm.experiments.make_report"
            assert "--publication" in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(R, "run_queue", queue)
    monkeypatch.setattr(R.subprocess, "run", report)
    args = SimpleNamespace(dry_run=False, jobs=10, threads_per_job=1)
    assert R.execute(repo, args) == training_code
    expected = ["forecast", "training"]
    if not training_code:
        expected += [
            "lmm.experiments.make_report",
            "lmm.experiments.cashflow_comparison",
            "lmm.experiments.cashflow_comparison",
        ]
        assert json.loads((main / H.STATE).read_text())["status"] == "complete"
    assert events == expected
    assert H.tree_digest(historical) == history
