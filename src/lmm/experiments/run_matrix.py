"""Run independent experiments through a bounded, continuously refilled queue.

Bounded orchestration: learning, seeds and per-run completion
validation remain in the existing launchers. One root lock includes final
report generation. A failed job stops further dispatch and drains active jobs.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from lmm.experiments.protocol import PUBLICATION_SEEDS, RESULTS_ROOT, REVISED_RESULTS_ROOT
from lmm.config import LEGACY_CLEARING, VOLUME_MAX_CLEARING
from lmm.experiments.clearing_campaign import (
    ForecastJob, SETTINGS, validate_campaign_root, validate_forecast_fit,
)

ALGOS = ("dqn", "ddpg", "td3", "sac")
TICKERS = ("MSFT", "JPM", "PG", "GOOGL", "CAT")
ARMS = ("mechanism_fixed_anchor", "mechanism_economic_dense", "mechanism_economic_sparse",
        "mechanism_h_feature_off", "no_auction")
CANONICAL_SEEDS = set(PUBLICATION_SEEDS)


@dataclass(frozen=True)
class Job:
    seed: int
    algo: str
    block: str

    @property
    def name(self):
        return f"{self.block}__{self.algo}_seed{self.seed}"

    def command(self, repo, smoke):
        return ["bash", str(repo / "scripts/_run_matrix_job.sh"),
                str(self.seed), self.algo, self.block, str(int(smoke))]


def build_jobs(seeds, symbol=None):
    if len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if symbol is not None and symbol not in TICKERS:
        raise ValueError("unknown historical symbol")
    return [Job(seed, algo, block)
            for block in ("synthetic", *(TICKERS if symbol is None else (symbol,)), *ARMS)
            for seed in seeds for algo in ALGOS]


def worker_environment(threads, root):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "LMM_TORCH_INTRAOP_THREADS"):
        env[name] = str(threads)
    env["LMM_TORCH_INTEROP_THREADS"] = "1"
    env["LMM_RESULTS_ROOT"] = str(root)
    return env


@contextmanager
def root_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "active.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another matrix launcher is active for this results root") from None
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        yield
        # Closing releases the kernel lock even after interruption; the file
        # remaining on disk is not interpreted as a stale running process.


def run_queue(jobs, *, workers, env, log_dir, command, cwd, poll_seconds=.2):
    if workers < 1 or len({j.name for j in jobs}) != len(jobs):
        raise ValueError("positive workers and unique job identities required")
    pending, active, completed, failed = deque(jobs), {}, [], []
    started = time.monotonic()
    log_dir.mkdir(parents=True, exist_ok=True)

    def status(state):
        value = dict(state=state, total=len(jobs), completed=completed, failed=failed,
                     active=[j.name for j, _ in active.values()], pending=[j.name for j in pending],
                     elapsed_seconds=round(time.monotonic() - started, 3), workers=workers)
        temporary = log_dir / "status.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, log_dir / "status.json")

    try:
        while pending or active:
            while pending and len(active) < workers and not failed:
                job = pending.popleft()
                log = (log_dir / f"{job.name}.log").open("a")
                log.write(f"\n=== dispatch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                log.flush()
                try:
                    process = subprocess.Popen(command(job), cwd=cwd, env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[process] = (job, log)
                print(f"START {job.name} ({len(active)}/{workers} active)", flush=True)
            status("draining_after_failure" if failed else "running")
            if not active:
                break
            time.sleep(poll_seconds)
            for process, (job, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[process]
                if code:
                    failed.append(dict(job=job.name, exit_code=code))
                    print(f"FAILED {job.name}: exit {code}; see {log.name}. Stopping dispatch.", flush=True)
                else:
                    completed.append(job.name)
                    print(f"DONE {job.name} ({len(completed)}/{len(jobs)})", flush=True)
        status("failed" if failed else "complete")
        return 1 if failed else 0
    except BaseException:
        # Every worker owns a process group, including its training/evaluation
        # children. Stop the group so Ctrl-C cannot leave orphan writers.
        for process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for process, (_, log) in active.items():
            try:
                process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
        status("interrupted")
        raise


def report_command(repo, seeds, smoke, symbol):
    result = ["bash", str(repo / "scripts/make_multiseed_outputs.sh"),
              "--seeds", " ".join(map(str, seeds)), "--require-complete"]
    if symbol:
        result += ["--symbol", symbol]
    if not smoke:
        result += ["--publication"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=2)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--legacy-clearing", action="store_true", help="explicitly reproduce the retained v19 mechanism")
    parser.add_argument("--symbol", choices=TICKERS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    if args.root is None:
        args.root = Path(RESULTS_ROOT if args.legacy_clearing else REVISED_RESULTS_ROOT)
    mechanism = LEGACY_CLEARING if args.legacy_clearing else VOLUME_MAX_CLEARING
    if args.jobs < 1 or args.threads_per_job < 1:
        parser.error("jobs and threads-per-job must be positive")
    if len(args.seeds) < 2:
        parser.error("at least two master seeds required")
    if args.smoke and CANONICAL_SEEDS.intersection(args.seeds):
        parser.error("smoke runs must use noncanonical seeds")
    if not args.smoke and (set(args.seeds) != CANONICAL_SEEDS or args.symbol):
        parser.error("publication requires all ten canonical seeds and five tickers")
    try:
        jobs = build_jobs(args.seeds, args.symbol)
        root = (args.root if args.root.is_absolute() else repo / args.root).resolve()
        forecasts = [] if args.legacy_clearing else [ForecastJob(setting) for setting in SETTINGS]
        if args.dry_run:
            print(json.dumps(dict(jobs=len(jobs), workers=args.jobs, threads_per_worker=args.threads_per_job,
                                  clearing_mechanism=mechanism, root=str(root),
                                  forecast_commands=[j.command(repo, root, args.smoke) for j in forecasts],
                                  reference_calibration="fresh training-only normalization/inventory reference in every training job",
                                  forecast_directory=str(root/'_forecasts') if forecasts else None,
                                  commands=[j.command(repo, args.smoke) for j in jobs]), indent=2))
            return 0
        if not args.smoke:
            dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, text=True)
            if dirty.strip():
                parser.error("full publication mode requires a clean git worktree")
        validate_campaign_root(root, mechanism)
        env = worker_environment(args.threads_per_job, root)
        env['LMM_CLEARING_MECHANISM'] = mechanism
        env['LMM_PYTHON'] = sys.executable
        if forecasts:
            env['LMM_FORECAST_ROOT'] = str(root/'_forecasts')
        else:
            env.pop('LMM_FORECAST_ROOT', None)
        env.setdefault("MPLCONFIGDIR", str(repo / ".cache/matplotlib"))
        with root_lock(root / "_orchestration"):
            if forecasts:
                print('Preparing two revised forecasts: fit on training paths; validation is diagnostic only.', flush=True)
                code = run_queue(forecasts, workers=min(args.jobs, len(forecasts)), env=env,
                                 log_dir=root/'_orchestration/forecasts', cwd=repo,
                                 command=lambda j: j.command(repo, root, args.smoke))
                if code:
                    return code
                for setting in SETTINGS:
                    validate_forecast_fit(repo, root, setting, smoke=args.smoke)
            (root/'_orchestration/launch_plan.json').write_text(json.dumps(dict(
                clearing_mechanism=mechanism, root=str(root), jobs=len(jobs),
                workers=args.jobs, threads_per_worker=args.threads_per_job,
                forecast_commands=[j.command(repo, root, args.smoke) for j in forecasts],
                commands=[j.command(repo, args.smoke) for j in jobs],
            ), indent=2)+'\n')
            code = run_queue(jobs, workers=args.jobs, env=env, log_dir=root / "_orchestration",
                             command=lambda j: j.command(repo, args.smoke), cwd=repo)
            if code:
                return code
            print("All runs completed; generating the research report once.", flush=True)
            return subprocess.call(report_command(repo, args.seeds, args.smoke, args.symbol), cwd=repo, env=env)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; active workers stopped. Partial runs retain their diagnostics.\n")


if __name__ == "__main__":
    raise SystemExit(main())
