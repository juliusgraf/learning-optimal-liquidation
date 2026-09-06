"""Queue resource limits, refill, failure isolation and publication matrix."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from lmm.experiments import run_matrix as M


def test_matrix_contains_exactly_440_unique_experiments():
    jobs = M.build_jobs(M.PUBLICATION_SEEDS)
    assert len(jobs) == len({j.name for j in jobs}) == 440
    assert sum(j.block == "synthetic" for j in jobs) == 40
    assert sum(j.block in M.TICKERS for j in jobs) == 200
    assert sum(j.block in M.ARMS for j in jobs) == 200
    assert all("shaping_on_shaping_on" not in j.block for j in jobs)
    assert len(M.build_jobs([9001, 9002], "MSFT")) == 56


def _worker(tmp_path):
    path = tmp_path / "worker.py"
    path.write_text('''import json,sys,time,os
from pathlib import Path
name, delay, code = sys.argv[1:]
p = Path(name+'.json')
start = time.monotonic()
time.sleep(float(delay))
p.write_text(json.dumps(dict(start=start,end=time.monotonic(),threads=os.environ.get('OMP_NUM_THREADS'))))
sys.exit(int(code))
''')
    return path


def test_queue_refills_immediately_and_exceeds_five_seed_workers(tmp_path):
    worker = _worker(tmp_path)
    jobs = M.build_jobs([9001, 9002], "MSFT")[:9]
    # Eight active runs although only two master seeds exist. One deliberately
    # slow run must not hold up dispatch after the other slots become free.
    code = M.run_queue(jobs, workers=8, env=M.worker_environment(1, tmp_path),
                       log_dir=tmp_path / "logs", cwd=tmp_path, poll_seconds=.01,
                       command=lambda j: [sys.executable, str(worker), j.name,
                                          "1.2" if j == jobs[0] else ".1", "0"])
    assert code == 0
    data = [json.loads((tmp_path / (j.name+'.json')).read_text()) for j in jobs]
    assert data[8]['start'] < data[0]['end']
    events = sorted([(v['start'],1) for v in data]+[(v['end'],-1) for v in data])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert 5 < peak <= 8
    assert all(v['threads'] == '1' for v in data)
    state = json.loads((tmp_path/'logs/status.json').read_text())
    assert state['state'] == 'complete' and len(state['completed']) == 9


def test_failed_job_stops_dispatch_and_drains_active_jobs(tmp_path):
    worker = _worker(tmp_path)
    jobs = M.build_jobs([9001, 9002], "MSFT")[:4]
    code = M.run_queue(jobs, workers=2, env=os.environ.copy(), log_dir=tmp_path/'logs', cwd=tmp_path,
                       poll_seconds=.01, command=lambda j: [sys.executable,str(worker),j.name,
                                        '.03' if j == jobs[0] else '.3','2' if j == jobs[0] else '0'])
    assert code == 1
    state = json.loads((tmp_path/'logs/status.json').read_text())
    assert state['state'] == 'failed'
    assert state['completed'] == [jobs[1].name]
    assert state['pending'] == [j.name for j in jobs[2:]]
    assert not (tmp_path/(jobs[2].name+'.json')).exists()


def test_results_root_lock_is_exclusive_and_reusable(tmp_path):
    with M.root_lock(tmp_path):
        with pytest.raises(ValueError, match='another matrix launcher'):
            with M.root_lock(tmp_path):
                pass
    with M.root_lock(tmp_path):
        pass


def test_interruption_terminates_active_workers(tmp_path, monkeypatch):
    worker = _worker(tmp_path)
    original = M.subprocess.Popen
    original_sleep = M.time.sleep
    processes = []

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    def interrupt(_):
        # Interrupt the queue once; subprocess.wait must retain its real sleep
        # while reaping the terminated children.
        monkeypatch.setattr(M.time, 'sleep', original_sleep)
        raise KeyboardInterrupt

    monkeypatch.setattr(M.subprocess, 'Popen', start)
    monkeypatch.setattr(M.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        M.run_queue(M.build_jobs([9001, 9002])[:2], workers=2, env=os.environ.copy(),
                    log_dir=tmp_path/'logs', cwd=tmp_path,
                    command=lambda j: [sys.executable, str(worker), j.name, '30', '0'])
    assert len(processes) == 2 and all(p.poll() is not None for p in processes)
    assert json.loads((tmp_path/'logs/status.json').read_text())['state'] == 'interrupted'


def test_thread_limits_cover_native_libraries_and_torch(tmp_path):
    env = M.worker_environment(1, tmp_path)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS',
                'LMM_TORCH_INTRAOP_THREADS','LMM_TORCH_INTEROP_THREADS']:
        assert env[key] == '1'


def test_direct_entry_point_preserves_publication_guard(tmp_path):
    with pytest.raises(SystemExit) as e:
        M.main(['--seeds','42','7','--root',str(tmp_path)])
    assert e.value.code == 2
    assert not list(tmp_path.iterdir())


def test_report_runs_once_after_success_and_not_after_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(M, 'run_queue', lambda *a, **k: 0)
    monkeypatch.setattr(M.subprocess, 'call', lambda command, **k: calls.append(command) or 0)
    args = ['--seeds','9001','9002','--smoke','--symbol','MSFT','--root',str(tmp_path),'--jobs','10','--threads-per-job','1']
    assert M.main(args) == 0
    assert len(calls) == 1
    assert calls[0][-3:] == ['--require-complete','--symbol','MSFT']
    monkeypatch.setattr(M, 'run_queue', lambda *a, **k: 1)
    assert M.main(args) == 1
    assert len(calls) == 1


def test_dry_run_has_no_training_or_filesystem_side_effects(tmp_path, capsys):
    assert M.main(['--seeds', *map(str, M.PUBLICATION_SEEDS), '--jobs','10',
                   '--threads-per-job','1','--dry-run','--root',str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['jobs'] == 440 and plan['workers'] == 10
    assert len(plan['commands']) == 440
    assert not list(tmp_path.iterdir())
