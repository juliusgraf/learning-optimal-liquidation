"""Explicit, audited post-run inclusion amendment; never train or rewrite checkpoints.

All canonical runs use their best mature validation policy, even if it failed
to improve on initialization. Existing evaluations are reused only after full
checkpoint-content equivalence. Original configs, gate decisions, Git SHAs and
completion manifests remain unchanged. Missing evaluations alone are computed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pandas as pd
import yaml

from lmm.config import load_config
from lmm.experiments import publication as P

SCHEMA = "lmm-best-mature-reporting-amendment-v1"
RULE = "best-mature-regardless-of-initial-improvement"
RELATIVE_PATH = "_reporting_protocol/best_mature_v1.json"
NOTE = (
    "Post-run reporting amendment: every algorithm, market, treatment and seed "
    "is included using its best mature economic-validation checkpoint, regardless "
    "of improvement over initialization. This changes the original inclusion "
    "gate after observing a non-improving run; it was not preregistered. Training, "
    "validation seeds, maturity thresholds, stopping histories and test seeds "
    "are unchanged. Existing evaluations are reused only where checkpoint "
    "contents are equivalent. Non-improvement is retained as a result."
)
REPO = Path(__file__).resolve().parents[3]
# These files implement the audited reporting amendment. Every other executable
# source/config must equal the saved training revision, including train.py,
# agents, simulator, dependencies and policy-difference calculations.
REPORTING_FILES = {
    "src/lmm/experiments/mature_reporting.py",
    "src/lmm/experiments/publication.py",
    "src/lmm/experiments/plotting.py",
    "src/lmm/experiments/evaluate.py",
    "src/lmm/experiments/make_report.py",
    "src/lmm/experiments/make_figures.py",
}


def source_inventory():
    paths = {p for folder in ("src", "configs", "scripts")
             for p in (REPO / folder).rglob("*")
             if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".sh", ".json", ".toml"}}
    paths.update(REPO / p for p in ("pyproject.toml",))
    return {p.relative_to(REPO).as_posix(): P._sha256_file(p) for p in sorted(paths)}


def audit_source_revision(sha):
    current = source_inventory()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", sha, "--", "src", "configs", "scripts", "pyproject.toml"],
        cwd=REPO, text=True,
    ).splitlines()
    original = {n for n in names if Path(n).suffix in {".py", ".yaml", ".yml", ".sh", ".json", ".toml"}}
    for name in (set(current) | original) - REPORTING_FILES:
        if name not in current or name not in original:
            raise ValueError(f"execution source inventory changed: {name}")
        data = subprocess.check_output(["git", "show", f"{sha}:{name}"], cwd=REPO)
        if hashlib.sha256(data).hexdigest() != current[name]:
            raise ValueError(f"execution source differs from training revision: {name}")
    return current


def expected_runs(root):
    from lmm.experiments.run_matrix import build_jobs, TICKERS
    from lmm.experiments.protocol import V17_SEEDS
    runs = []
    # This recovery protocol is deliberately restricted to the archived v17
    # matrix. V18 retains non-improvers prospectively and needs no amendment.
    for job in build_jobs(V17_SEEDS):
        symbol = job.block if job.block in TICKERS else None
        if symbol:
            setting, name = "historical_sp500_midquotes", f"{job.algo}_{symbol}_seed{job.seed}"
        else:
            suffix = "" if job.block == "synthetic" else f"__{job.block}"
            setting, name = f"synthetic_rough_heston{suffix}", f"{job.algo}{suffix}_seed{job.seed}"
        rd = root / setting / name
        cfg = load_config(rd / "config_resolved.yaml")
        if cfg.experiment.master_seed != job.seed or int((rd / "seed.txt").read_text()) != job.seed:
            raise ValueError(f"{rd}: seed provenance mismatch")
        run = SimpleNamespace(run_dir=rd, cfg=cfg, algo=job.algo, seed=job.seed,
                              setting=setting, symbol=symbol)
        P._validate_canonical_publication_config(run)
        runs.append(run)
    observed = {p.parent.resolve() for p in root.glob("*/*/config_resolved.yaml")
                if not p.parent.parent.name.startswith("_")}
    if observed != {r.run_dir.resolve() for r in runs}:
        raise ValueError("amendment requires exactly the canonical 220-run matrix")
    return runs


def validate_mature_selection(rd, cfg):
    selection = yaml.safe_load((rd / "checkpoints/best_mature_selection.yaml").read_text())
    initial = yaml.safe_load((rd / "checkpoints/initial_validation.yaml").read_text())
    if (selection.get("metric") != cfg.rl.checkpoint_metric or selection.get("mode") != "max"
            or selection.get("seed_stream") != "env_eval"
            or selection.get("validation_seeds") != initial.get("validation_seeds")
            or len(selection.get("validation_seeds", [])) != cfg.rl.validation_size):
        raise ValueError(f"{rd}: invalid mature validation provenance")
    required = {"clob": cfg.rl.checkpoint_min_clob_updates,
                "auction": cfg.rl.checkpoint_min_auction_updates if cfg.experiment.auction_enabled else 0}
    eligibility = selection.get("eligibility", {})
    if (eligibility.get("eligible") is not True or eligibility.get("required_updates") != required
            or any(eligibility.get("observed_updates", {}).get(k, -1) < v for k, v in required.items())):
        raise ValueError(f"{rd}: mature checkpoint fails configured update thresholds")
    metrics = pd.read_csv(rd / "metrics.csv")
    eligible = metrics.loc[(metrics.eval_checkpoint_eligible == 1) & metrics.eval_checkpoint_score.notna()]
    chosen = eligible.loc[eligible.episode == selection["episode"]]
    if (len(chosen) != 1 or not math.isfinite(selection["value"])
            or not math.isclose(float(chosen.eval_checkpoint_score.iloc[0]), selection["value"], abs_tol=1e-10)
            or not math.isclose(float(eligible.eval_checkpoint_score.max()), selection["value"], abs_tol=1e-10)):
        raise ValueError(f"{rd}: selected checkpoint is not the best observed mature validation")
    if not math.isclose(selection["economic_safety"]["initial_validation_score"], initial["value"], abs_tol=1e-10):
        raise ValueError(f"{rd}: initial-validation provenance mismatch")
    return selection


def _equivalent(a, b):
    """Exact payload equality, ignoring ZIP entry timestamps only (SB3 saves)."""
    import numpy as np
    import torch
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equivalent(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, bytes) and a != b and zipfile.is_zipfile(io.BytesIO(a)) and zipfile.is_zipfile(io.BytesIO(b)):
        with zipfile.ZipFile(io.BytesIO(a)) as x, zipfile.ZipFile(io.BytesIO(b)) as y:
            return set(x.namelist()) == set(y.namelist()) and all(x.read(n) == y.read(n) for n in x.namelist())
    return a == b


def equivalent_checkpoints(a, b):
    # Fast path avoids deserializing 219 models. PyTorch names the archive root
    # after the destination file; that name is not part of the model state.
    def members(path):
        with zipfile.ZipFile(path) as z:
            return {n.split("/", 1)[1]: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    if members(a) == members(b):
        return True
    import torch
    return _equivalent(torch.load(a, weights_only=False, map_location="cpu"),
                       torch.load(b, weights_only=False, map_location="cpu"))


def training_inventory(rd):
    return {p.relative_to(rd).as_posix(): P._sha256_file(p) for p in sorted(rd.rglob("*"))
            if p.is_file() and p.relative_to(rd).parts[0] not in {"eval", "figures", "tables", "logs"}
            and p.name != P.COMPLETION_MANIFEST_NAME}


def load_protocol(runs, *, verify_source=True):
    roots = {r.run_dir.resolve().parents[1] for r in runs}
    present = [root for root in roots if (root / RELATIVE_PATH).is_file()]
    if not present:
        return None
    if len(roots) != 1:
        raise ValueError("cannot mix amended and unrelated reporting roots")
    root = present[0]
    protocol = json.loads((root / RELATIVE_PATH).read_text())
    if protocol.get("schema") != SCHEMA or protocol.get("rule") != RULE:
        raise ValueError("unsupported reporting amendment")
    if verify_source and protocol.get("source_files") != source_inventory():
        raise ValueError("source files changed since reporting audit; review and explicitly re-audit the amendment")
    if len(protocol.get("runs", {})) != 220:
        raise ValueError("reporting amendment must cover every one of the 220 runs")
    protocol["_root"] = str(root)
    return protocol


def validate_protocol_run(protocol, run):
    rd = run.run_dir
    key = rd.resolve().relative_to(Path(protocol["_root"])).as_posix()
    entry = protocol["runs"][key]
    selection = validate_mature_selection(rd, run.cfg)
    if entry["selected_episode"] != selection["episode"] or entry["validation_score"] != selection["value"]:
        raise ValueError("selection differs from reporting audit")
    name = entry["evaluated_checkpoint"]
    if name == "best":
        if P._sha256_file(rd / P.COMPLETION_MANIFEST_NAME) != entry["original_completion_sha256"]:
            raise ValueError("original completion manifest changed since amendment")
    elif name == "best_mature":
        if P._completion_manifest_payload(rd, checkpoint=name) != entry["completion"]:
            raise ValueError("amended evaluation or original training artifacts changed")
        if (run.metadata.get("checkpoint_selection_rule") != RULE
                or run.metadata.get("checkpoint_selection") != selection):
            raise ValueError("mature evaluation selection provenance mismatch")
    else:
        raise ValueError("invalid amended checkpoint")
    return name


def adopt(root, *, refresh_reporting_source=False):
    from lmm.experiments.run_matrix import root_lock, worker_environment
    root = Path(root)
    with root_lock(root / "_orchestration"):
        destination = root / RELATIVE_PATH
        runs = expected_runs(root)
        if destination.exists():
            protocol = load_protocol(runs, verify_source=not refresh_reporting_source)
            source = audit_source_revision(protocol["training_git_sha"])
            if refresh_reporting_source:
                changed = {n for n in source.keys() | protocol["source_files"].keys()
                           if source.get(n) != protocol["source_files"].get(n)}
                if changed - (REPORTING_FILES - {"src/lmm/experiments/evaluate.py"}):
                    raise ValueError("source refresh may change reporting only; evaluation source must remain identical")
            from lmm.experiments.plotting import collect_runs
            for run in collect_runs([r.run_dir for r in runs]):
                validate_protocol_run(protocol, run)
                if (run.run_dir / P.COMPLETION_MANIFEST_NAME).exists():
                    P.validate_completion_manifest(run.run_dir)
            if refresh_reporting_source and source != protocol["source_files"]:
                protocol.setdefault("source_audit_history", []).append(protocol["source_files"])
                protocol["source_files"] = source
                protocol.pop("_root")
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, destination)
            print(f"Existing amendment verified: {destination}", flush=True)
            return
        shas = {(r.run_dir / "git_sha.txt").read_text().strip() for r in runs}
        if len(shas) != 1:
            raise ValueError("training revisions differ across the matrix")
        sha = shas.pop()
        source = audit_source_revision(sha)
        entries, missing = {}, []
        for i, run in enumerate(runs, 1):
            rd = run.run_dir
            selected = validate_mature_selection(rd, run.cfg)
            entry = {"selected_episode": selected["episode"], "validation_score": selected["value"],
                     "initial_validation_score": selected["economic_safety"]["initial_validation_score"],
                     "improved_over_initial": selected["value"] > selected["economic_safety"]["initial_validation_score"]}
            if (rd / P.COMPLETION_MANIFEST_NAME).exists():
                P.validate_completion_manifest(rd)
                best = yaml.safe_load((rd / "checkpoints/best_selection.yaml").read_text())
                if ((best["episode"], best["value"]) != (selected["episode"], selected["value"])
                        or not equivalent_checkpoints(rd / "checkpoints/best.pt", rd / "checkpoints/best_mature.pt")):
                    raise ValueError(f"{rd}: existing evaluation does not represent the best mature policy")
                entry.update(evaluated_checkpoint="best", checkpoint_equivalence="exact-state-and-nested-archive-payload",
                             original_completion_sha256=P._sha256_file(rd / P.COMPLETION_MANIFEST_NAME))
            else:
                failure_path = rd / "checkpoints/selection_failure.yaml"
                failure = yaml.safe_load(failure_path.read_text()) if failure_path.exists() else {}
                metrics = pd.read_csv(rd / "metrics.csv")
                if (not str(failure.get("reason", "")).startswith("no mature checkpoint passed the configured initial-policy improvement gate:")
                        or selected["economic_safety"]["reportable"] is not False
                        or (rd / "checkpoints/best.pt").exists()
                        or len(metrics) != run.cfg.experiment.episodes
                        or not (rd / "checkpoints/final.pt").exists()):
                    raise ValueError(f"{rd}: incomplete run is not a completed non-improving training run")
                entry["evaluated_checkpoint"] = "best_mature"
                missing.append((run, training_inventory(rd)))
            entries[rd.relative_to(root).as_posix()] = entry
            if i % 20 == 0:
                print(f"Audited {i}/{len(runs)} saved training runs", flush=True)
        print(f"Reusing {len(runs)-len(missing)} evaluations; evaluating {len(missing)} saved mature policies. No training.", flush=True)
        log_dir = root / "_reporting_protocol"
        log_dir.mkdir(exist_ok=True)
        for run, before in missing:
            rd = run.run_dir
            eval_dir = rd / "eval"
            # Resume only an evaluation created by this exact audited procedure.
            receipt = log_dir / f"{rd.name}.evaluation_started.json"
            expected_receipt = {"source_files": source, "training_files": before, "rule": RULE}
            if eval_dir.exists() and any(eval_dir.rglob("*")):
                if not receipt.is_file() or json.loads(receipt.read_text()) != expected_receipt:
                    raise ValueError(f"{rd}: refusing to overwrite an unaudited existing evaluation")
            receipt.write_text(json.dumps(expected_receipt, indent=2) + "\n")
            env = worker_environment(1, root)
            commands = [[sys.executable, "-m", "lmm.experiments.evaluate", "--run-dir", str(rd),
                         "--checkpoint", "best_mature", "--trace-episodes", "1"]]
            if run.symbol:
                commands[0] += ["--symbol", run.symbol]
            commands += [[sys.executable, "-m", "lmm.experiments.policy_differences", "--run-dir", str(rd),
                          "--benchmark", b] for b in ("as", "twap")]
            with (log_dir / f"{rd.name}.log").open("a") as log:
                for command in commands:
                    subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if training_inventory(rd) != before:
                raise ValueError(f"{rd}: evaluation modified a training artifact")
            entries[rd.relative_to(root).as_posix()]["completion"] = P._completion_manifest_payload(rd, checkpoint="best_mature")
        if source_inventory() != source:
            raise ValueError("source changed during reporting audit")
        protocol = {"schema": SCHEMA, "rule": RULE, "disclosure": NOTE, "training_git_sha": sha,
                    "source_files": source, "runs": entries}
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
        print(f"Wrote reporting amendment: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/revision_v17"))
    parser.add_argument("--refresh-reporting-source", action="store_true",
                        help="re-audit existing artifacts after a reporting-only code change; never change evaluation code or rerun evaluation")
    args = parser.parse_args(argv)
    adopt(args.root, refresh_reporting_source=args.refresh_reporting_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
