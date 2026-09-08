"""Replace only the 320 synthetic runs of the saved 520-run v20 campaign.

Preparation is read-only. Archiving, fitting and training occur only when the
user launches the non-dry command from a clean checkout. Historical files are
never rewritten; their prior published identities are retained explicitly.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from lmm.config import VOLUME_MAX_CLEARING, load_config
from lmm.experiments.clearing_campaign import ForecastJob, validate_forecast_fit
from lmm.experiments.protocol import PUBLICATION_SEEDS
from lmm.experiments.retained_history import (
    ARCHIVE,
    BASELINE,
    MESH,
    MESH_SOURCE,
    ROOT_NAMES,
    STATE,
    RetainedHistory,
    digest,
)
from lmm.experiments.run_matrix import (
    ALGOS,
    ARMS,
    Job,
    root_lock,
    run_queue,
    worker_environment,
)


@dataclass(frozen=True)
class ReplacementJob:
    root_name: str
    job: Job

    @property
    def name(self):
        return f"{self.root_name}__{self.job.name}"

    def command(self, repo):
        return [
            "env",
            f"LMM_RESULTS_ROOT={repo / 'results' / self.root_name}",
            *self.job.command(repo, False),
        ]

    def run_path(self, repo):
        suffix = "" if self.job.block == "synthetic" else "__" + self.job.block
        return (
            repo
            / "results"
            / self.root_name
            / ("synthetic_rough_heston" + suffix)
            / f"{self.job.algo}{suffix}_seed{self.job.seed}"
        )


def replacement_jobs():
    blocks = [(ROOT_NAMES[0], block) for block in ("synthetic", *ARMS)]
    blocks += [
        (ROOT_NAMES[1], "mechanism_economic_cashflow"),
        (ROOT_NAMES[2], "mechanism_economic_dense_h"),
    ]
    return [
        ReplacementJob(root, Job(seed, algo, block))
        for root, block in blocks
        for seed in PUBLICATION_SEEDS
        for algo in ALGOS
    ]


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_baseline(repo):
    baseline = json.loads((repo / BASELINE).read_text())
    expected_synthetic = {
        j.run_path(repo).relative_to(repo).as_posix() for j in replacement_jobs()
    }
    saved = set(baseline["original_runs"]) - set(baseline["historical_runs"])
    if (
        baseline.get("schema") != "v20-synthetic-replacement-baseline-v1"
        or len(baseline["historical_runs"]) != 200
        or len(saved) != 320
        or saved != expected_synthetic
    ):
        raise ValueError(
            "saved v20 inventory does not match the complete 320/200 replacement plan"
        )
    return baseline


def archive_sources(repo):
    """Only synthetic trees, obsolete combined reports, and their old queue logs."""
    paths = {j.run_path(repo).parent for j in replacement_jobs()}
    main = repo / "results" / ROOT_NAMES[0]
    paths.add(main / "_forecasts/synthetic_rough_heston")
    for name in (
        "_publication",
        "_development",
        "_comparison_with_v19",
        "_provenance/source_attestation.json",
    ):
        paths.add(main / name)
    for root_name in ROOT_NAMES:
        root = repo / "results" / root_name
        paths.add(root / "_comparison")
        for relative in (
            "_orchestration/status.json",
            "_orchestration/launch_plan.json",
            "_orchestration/forecasts/status.json",
            "_orchestration/forecasts/forecast_synthetic_rough_heston.log",
        ):
            paths.add(root / relative)
        for path in (root / "_orchestration").glob("*.log"):
            if any(
                path.name.startswith(block + "__")
                for block in (
                    "synthetic",
                    *ARMS,
                    "mechanism_economic_cashflow",
                    "mechanism_economic_dense_h",
                )
            ):
                paths.add(path)
    return sorted(p for p in paths if p.exists())


def preflight_original(repo, baseline):
    """Check the saved inventories before moving anything; no fitting or writes."""
    observed = {
        p.parent.relative_to(repo).as_posix()
        for name in ROOT_NAMES
        for p in (repo / "results" / name).glob("*/*/config_resolved.yaml")
        if not p.parent.parent.name.startswith("_")
    }
    if observed != set(baseline["original_runs"]):
        raise ValueError("original v20 run inventory changed since preparation")
    for name, expected in baseline["original_runs"].items():
        path = repo / name
        if (
            path.is_symlink()
            or path.parent.is_symlink()
            or digest(path / "pipeline_complete.json") != expected["completion_sha256"]
            or digest(path / "config_resolved.yaml") != expected["config_sha256"]
        ):
            raise ValueError(f"original campaign changed since preparation: {path}")
    main = repo / "results" / ROOT_NAMES[0]
    if digest(main / "_publication/manifest.json") != baseline["prior_report_sha256"]:
        raise ValueError("original v20 publication report changed since preparation")
    previous = baseline["prior_report_source_attestation"]
    if digest(main / "_provenance/source_attestation.json") != previous["sha256"]:
        raise ValueError("original source attestation changed since preparation")
    # Revalidate ALL historical bytes (including logs) independently of the
    # completion manifest, then retain those original source identities.
    retained = RetainedHistory(repo, main, baseline, {})
    retained.revalidate()


def initialize_state(repo, baseline, revision):
    main = repo / "results" / ROOT_NAMES[0]
    path = main / STATE
    if path.exists():
        state = json.loads(path.read_text())
        if (
            state.get("launch_git_sha") != revision
            or state.get("baseline_sha256") != digest(repo / BASELINE)
            or state.get("mesh_sha256") != digest(repo / MESH_SOURCE)
        ):
            raise ValueError(
                "replacement already belongs to another source/config revision"
            )
        validate_moves(repo, state)
        return state
    if any((repo / "results" / name / ARCHIVE).exists() for name in ROOT_NAMES):
        raise ValueError(
            "archive exists without replacement journal; inspect before proceeding"
        )
    preflight_original(repo, baseline)
    moves = []
    for source in archive_sources(repo):
        if source.is_symlink():
            raise ValueError(f"refusing to archive symlink {source}")
        root = next(
            repo / "results" / name
            for name in ROOT_NAMES
            if source.is_relative_to(repo / "results" / name)
        )
        moves.append(
            dict(
                source=source.relative_to(repo).as_posix(),
                destination=(root / ARCHIVE / source.relative_to(root))
                .relative_to(repo)
                .as_posix(),
                done=False,
            )
        )
    state = dict(
        schema="v20-synthetic-replacement-v1",
        status="archiving",
        launch_git_sha=revision,
        baseline_sha256=digest(repo / BASELINE),
        mesh_sha256=digest(repo / MESH_SOURCE),
        max_step_minutes=0.25,
        started_utc=datetime.now(timezone.utc).isoformat(),
        synthetic_runs=320,
        retained_historical_runs=200,
        moves=moves,
    )
    atomic_json(path, state)
    return state


def validate_moves(repo, state):
    """Do not let a corrupted restart journal expand the authorized write scope."""
    if state.get("schema") != "v20-synthetic-replacement-v1":
        raise ValueError("unknown replacement journal schema")
    settings = {j.run_path(repo).parent for j in replacement_jobs()}
    allowed_metadata = {
        "_comparison",
        "_orchestration/status.json",
        "_orchestration/launch_plan.json",
        "_orchestration/forecasts/status.json",
        "_orchestration/forecasts/forecast_synthetic_rough_heston.log",
    }
    main_metadata = {
        "_publication",
        "_development",
        "_comparison_with_v19",
        "_forecasts/synthetic_rough_heston",
        "_provenance/source_attestation.json",
    }
    seen = set()
    for move in state["moves"]:
        source = repo / move["source"]
        roots = [
            repo / "results" / name
            for name in ROOT_NAMES
            if source.is_relative_to(repo / "results" / name)
        ]
        if len(roots) != 1 or ".." in source.parts or source in seen:
            raise ValueError("invalid archive source in restart journal")
        seen.add(source)
        root = roots[0]
        relative = source.relative_to(root).as_posix()
        synthetic_log = (
            source.parent == root / "_orchestration"
            and source.suffix == ".log"
            and any(
                source.name.startswith(block + "__")
                for block in (
                    "synthetic",
                    *ARMS,
                    "mechanism_economic_cashflow",
                    "mechanism_economic_dense_h",
                )
            )
        )
        allowed = (
            source in settings
            or relative in allowed_metadata
            or synthetic_log
            or (root.name == ROOT_NAMES[0] and relative in main_metadata)
        )
        destination = root / ARCHIVE / source.relative_to(root)
        if (
            not allowed
            or repo / move["destination"] != destination
            or source.is_symlink()
            or destination.is_symlink()
        ):
            raise ValueError(
                "restart journal attempts to archive outside synthetic scope"
            )
        if move["done"] and not destination.exists():
            raise ValueError("completed archive entry is missing")


def complete_archive(repo, state):
    """Recover even if interrupted between rename and journal update."""
    main = repo / "results" / ROOT_NAMES[0]
    validate_moves(repo, state)
    for move in state["moves"]:
        if move["done"]:
            continue
        source, destination = repo / move["source"], repo / move["destination"]
        if destination.exists():
            if source.exists():
                raise ValueError(f"ambiguous interrupted archive: {source}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        move["done"] = True
        atomic_json(main / STATE, state)
    for name in ROOT_NAMES:
        path = repo / "results" / name / MESH
        if path.exists() and digest(path) != state["mesh_sha256"]:
            raise ValueError(f"campaign refinement overlay changed: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / MESH_SOURCE, path)
    state["status"] = "prepared"
    atomic_json(main / STATE, state)


def archive_partial_attempts(repo, jobs):
    """An interrupted fresh attempt is never silently treated as a resumable model."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for job in jobs:
        path = job.run_path(repo)
        if path.exists() and not (path / "pipeline_complete.json").exists():
            root = repo / "results" / job.root_name
            destination = (
                root / ARCHIVE / "interrupted_attempts" / stamp / path.relative_to(root)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)


def plan(repo, jobs, args):
    return dict(
        synthetic_runs=len(jobs),
        retained_historical_runs=200,
        max_step_minutes=0.25,
        accuracy_status="user-selected candidate; not established accuracy threshold",
        workers=args.jobs,
        threads_per_worker=args.threads_per_job,
        roots=[str(repo / "results" / name) for name in ROOT_NAMES],
        archive=str(ARCHIVE),
        baseline=str(repo / BASELINE),
        forecast="one fresh synthetic training-only fit; retain historical fit",
        normalization="refit training-only normalizer and inventory reference in every new run",
        commands=[j.command(repo) for j in jobs],
        reports=[
            "revision_v20/_publication",
            "revision_v20_cashflow/_comparison",
            "revision_v20_economic_dense_h/_comparison",
        ],
        manuscript="existing paper remains a legacy-generator snapshot; no automatic rewrite of outcome-dependent prose",
    )


def execute(repo, args):
    repo = Path(repo).resolve()
    baseline = load_baseline(repo)
    mesh_cfg = load_config(
        repo / "configs/base.yaml",
        repo / "configs/synthetic_rough_heston.yaml",
        repo / MESH_SOURCE,
    )
    if mesh_cfg.midprice.rough_heston.rough_heston_max_step_minutes != 0.25:
        raise ValueError("v20 replacement requires the user-selected 0.25-minute mesh")
    jobs = replacement_jobs()
    launch_plan = plan(repo, jobs, args)
    if args.dry_run:
        print(json.dumps(launch_plan, indent=2))
        return 0
    from lmm.experiments.publication import _clean_head_sha

    revision = _clean_head_sha()
    main = repo / "results" / ROOT_NAMES[0]
    env = worker_environment(args.threads_per_job, main)
    env.update(
        LMM_CLEARING_MECHANISM=VOLUME_MAX_CLEARING,
        LMM_FORECAST_ROOT=str(main / "_forecasts"),
        LMM_PYTHON=sys.executable,
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    with ExitStack() as locks:
        for name in ROOT_NAMES:
            locks.enter_context(root_lock(repo / "results" / name / "_orchestration"))
        state = initialize_state(repo, baseline, revision)
        complete_archive(repo, state)
        retained = RetainedHistory.load(repo, main)
        retained.revalidate()
        log_dir = main / "_orchestration/synthetic_replacement"
        atomic_json(log_dir / "launch_plan.json", launch_plan)
        state["status"] = "fitting"
        atomic_json(main / STATE, state)
        fit = ForecastJob("synthetic_rough_heston")
        # Recover a crashed, incomplete fit, retaining its diagnostics.
        directory = main / "_forecasts/synthetic_rough_heston"
        if directory.exists() and not (directory / "completion.json").exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            destination = main / ARCHIVE / "interrupted_forecasts" / stamp
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(destination)
        code = run_queue(
            [fit],
            workers=1,
            env=env,
            log_dir=log_dir / "forecasts",
            cwd=repo,
            command=lambda j: j.command(repo, main, False),
        )
        if code:
            return code
        validate_forecast_fit(repo, main, fit.setting)
        archive_partial_attempts(repo, jobs)
        state["status"] = "training"
        atomic_json(main / STATE, state)
        code = run_queue(
            jobs,
            workers=args.jobs,
            env=env,
            log_dir=log_dir,
            cwd=repo,
            command=lambda j: j.command(repo),
        )
        if code:
            return code
        # Verify every fresh run's source/config/complete pipeline before a
        # combined report can become the active v20 result.
        state["status"] = "reporting"
        atomic_json(main / STATE, state)
        commands = [
            [
                sys.executable,
                "-m",
                "lmm.experiments.make_report",
                "--root",
                str(main),
                "--seeds",
                *map(str, PUBLICATION_SEEDS),
                "--publication",
            ]
        ]
        for name, conditioned in zip(ROOT_NAMES[1:], (False, True)):
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "lmm.experiments.cashflow_comparison",
                    "--root",
                    str(repo / "results" / name),
                    "--headline-root",
                    str(main),
                    "--report-only",
                    *(["--conditioned"] if conditioned else []),
                ]
            )
        for command in commands:
            subprocess.run(command, cwd=repo, env=env, check=True)
        retained.revalidate()
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(main / STATE, state)
        print(
            f"Replaced 320 synthetic runs; retained all 200 historical runs. Report: {main / '_publication/index.html'}"
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.threads_per_job < 1:
        parser.error("jobs and threads-per-job must be positive")
    try:
        return execute(Path(__file__).resolve().parents[3], args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Interrupted; active workers stopped. Relaunch the same command to retain completed runs and archive incomplete attempts.\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
