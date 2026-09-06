"""Publication-run validation and paired synthetic-treatment contrasts.

The treatment report is deliberately built from saved evaluation artifacts.  A
contrast pairs two learned policies from the *same algorithm and master seed*,
then verifies the exact episode/environment-seed common-random-number (CRN)
alignment before reducing the episode differences to one mean per master seed.
Those seed-level means are the independent units consumed by the table builder.

The headline ``synthetic_rough_heston`` runs are the
``H_on__shaping_on__auction_on`` arm. Keeping that mapping here avoids a
second set of numerically identical training runs under an
``ablation_h_on_shaping_on`` directory name. The shaping-off specification
is an explicit treatment; every policy is evaluated on economic performance.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from lmm.config import ExperimentConfig, load_config, to_dict
from lmm.experiments.protocol import PUBLICATION_SEEDS
from lmm.experiments.plotting import (
    ALGO_ORDER,
    HISTORICAL_SETTING,
    RunInfo,
    collect_runs,
)

__all__ = [
    "COMPLETION_MANIFEST_NAME",
    "COMPLETION_MANIFEST_SCHEMA",
    "HEADLINE_TREATMENT",
    "PUBLICATION_SEEDS",
    "TREATMENT_SPECS",
    "TREATMENT_CONTRASTS",
    "write_completion_manifest",
    "validate_completion_manifest",
    "validate_publication_runs",
    "validate_completed_run",
    "validate_treatment_run_configs",
    "build_treatment_contrast_records",
]


HEADLINE_TREATMENT = "headline"
COMPLETION_MANIFEST_NAME = "pipeline_complete.json"
COMPLETION_MANIFEST_SCHEMA = "lmm-pipeline-completion-v3"
DISK_SAFE_CHECKPOINT_INTERVAL_EPISODES = 10_000_000
_COMPLETION_REQUIRED_FILES = (
    "config_resolved.yaml",
    "seed.txt",
    "git_sha.txt",
    "runtime_versions.json",
    "split_seeds.yaml",
    "metrics.csv",
    "realized_grids.jsonl",
    "h_forecasts_train.csv",
    "feature_normalizer.yaml",
    "checkpoints/initial.pt",
    "checkpoints/initial_validation.yaml",
    "checkpoints/best_mature.pt",
    "checkpoints/best_mature_selection.yaml",
    "checkpoints/best.pt",
    "checkpoints/best_selection.yaml",
    "checkpoints/final.pt",
    "eval/records.csv",
    "eval/metadata.yaml",
    "eval/policy_difference_as.csv",
    "eval/policy_difference_twap.csv",
)

# Stable insertion order becomes the expected matrix/discovery order in the
# shell pipeline.  ``setting`` is the resolved experiment.name/run-directory
# name; the remaining values are the treatment-defining resolved fields.
TREATMENT_SPECS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            HEADLINE_TREATMENT,
            {
                "setting": "synthetic_rough_heston",
                "ablation_label": "H_on__shaping_on__auction_on",
                "h_cl": True,
                "auction_anchor": "indicative",
                "shaping": True,
                "clob_shaping": True,
                "auction_shaping": True,
                "clawback_shaping": True,
                "auction": True,
                "cancellation": True,
            },
        ),
        (
            "h_off_shaping_off",
            {
                "setting": "synthetic_rough_heston__ablation_h_off_shaping_off",
                "ablation_label": "H_off__shaping_off__auction_on",
                "h_cl": False,
                "auction_anchor": "frozen_mid",
                "shaping": False,
                "clob_shaping": False,
                "auction_shaping": False,
                "clawback_shaping": True,
                "auction": True,
                "cancellation": True,
            },
        ),
        (
            "h_on_shaping_off",
            {
                "setting": "synthetic_rough_heston__ablation_h_on_shaping_off",
                "ablation_label": "H_on__shaping_off__auction_on",
                "h_cl": True,
                "auction_anchor": "indicative",
                "shaping": False,
                "clob_shaping": False,
                "auction_shaping": False,
                "clawback_shaping": True,
                "auction": True,
                "cancellation": True,
            },
        ),
        (
            "h_off_shaping_on",
            {
                "setting": "synthetic_rough_heston__ablation_h_off_shaping_on",
                "ablation_label": "H_off__shaping_on__auction_on",
                "h_cl": False,
                "auction_anchor": "frozen_mid",
                "shaping": True,
                "clob_shaping": True,
                "auction_shaping": True,
                "clawback_shaping": True,
                "auction": True,
                "cancellation": True,
            },
        ),
        (
            "no_auction",
            {
                "setting": "synthetic_rough_heston__no_auction",
                "ablation_label": "H_off__shaping_off__auction_off",
                "h_cl": False,
                "auction_anchor": "frozen_mid",
                "shaping": False,
                "clob_shaping": False,
                "auction_shaping": False,
                "clawback_shaping": False,
                "auction": False,
                "cancellation": False,
            },
        ),
        (
            "no_cancellation",
            {
                "setting": "synthetic_rough_heston__no_cancellation",
                "ablation_label": "H_on__shaping_on__auction_on__cancel_off",
                "h_cl": True,
                "auction_anchor": "indicative",
                "shaping": True,
                "clob_shaping": True,
                "auction_shaping": True,
                "clawback_shaping": True,
                "auction": True,
                "cancellation": False,
            },
        ),
    ]
)

# (machine key, publication row label, positive arm, negative arm).  Every
# reported number has the sign ``positive arm - negative arm``.
TREATMENT_CONTRASTS = (
    (
        "h_anchor_shaping_off",
        "H/anchor effect, shaping off (on - off)",
        "h_on_shaping_off",
        "h_off_shaping_off",
    ),
    (
        "h_anchor_shaping_on",
        "H/anchor effect, shaping on (on - off)",
        HEADLINE_TREATMENT,
        "h_off_shaping_on",
    ),
    (
        "shaping_h_off",
        "Shaping effect, H/anchor off (on - off)",
        "h_off_shaping_on",
        "h_off_shaping_off",
    ),
    (
        "shaping_h_on",
        "Shaping effect, H/anchor on (on - off)",
        HEADLINE_TREATMENT,
        "h_on_shaping_off",
    ),
    (
        "auction_aware_vs_no_auction",
        "Full auction-aware treatment vs no-auction comparator",
        HEADLINE_TREATMENT,
        "no_auction",
    ),
    (
        "auction_access_h_off_shaping_off",
        "Auction access, H/anchor and shaping off (on - off)",
        "h_off_shaping_off",
        "no_auction",
    ),
    (
        "cancellation",
        "Cancellation effect (on - off)",
        HEADLINE_TREATMENT,
        "no_cancellation",
    ),
)

_SETTING_TO_TREATMENT = {
    spec["setting"]: treatment for treatment, spec in TREATMENT_SPECS.items()
}
_TREATMENT_OVERLAYS = {
    HEADLINE_TREATMENT: None,
    "h_off_shaping_off": "ablation_h_off_shaping_off.yaml",
    "h_off_shaping_on": "ablation_h_off_shaping_on.yaml",
    "h_on_shaping_off": "ablation_h_on_shaping_off.yaml",
    "no_auction": "no_auction.yaml",
    "no_cancellation": "no_cancellation.yaml",
}
_PUBLICATION_EPISODES = 800
_PUBLICATION_EVAL_EPISODES = 100


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_manifest_payload(run_dir: Path, *, checkpoint: str = "best") -> dict[str, Any]:
    if checkpoint not in {"best", "best_mature"}:
        raise ValueError("completion checkpoint must be best or best_mature")
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise ValueError(
            f"{run_dir}: cannot build completion manifest; missing bound file "
            "'config_resolved.yaml'"
        )
    try:
        cfg = load_config(config_path)
        learned_policy = cfg.algo.name
    except Exception as exc:
        raise ValueError(
            f"{run_dir}: cannot build completion manifest from its resolved config"
        ) from exc

    required = (*(p for p in _COMPLETION_REQUIRED_FILES if checkpoint == "best" or p not in {
        "checkpoints/best.pt", "checkpoints/best_selection.yaml"
    }), *(
        f"eval/traces/{policy}_ep0.csv"
        for policy in (learned_policy, "initial", "as", "twap")
    ))
    if cfg.midprice.historical is not None:
        required = (*required, "historical_data_manifest.json")

    # Bind every training/provenance/checkpoint artifact that existed when the
    # pipeline completed. Generated reports and the timestamped human-readable
    # log are intentionally outside this inventory; the full evaluation tree is
    # bound separately below. This catches missing or later-added training
    # diagnostics without hard-coding only today's known filenames.
    training_files = (
        path.relative_to(run_dir).as_posix()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(run_dir).parts[0]
        not in {"eval", "figures", "tables", "logs"}
        and path.name != COMPLETION_MANIFEST_NAME
        and not path.name.startswith(f".{COMPLETION_MANIFEST_NAME}.tmp-")
    )
    # Evaluation produces several diagnostics in addition to the records and
    # trace files consumed by the standard reports. Bind the complete eval/
    # inventory so a completed run cannot silently mix pre- and post-rerun
    # evaluation artifacts. Generated figures/tables live outside eval/.
    eval_files = (
        path.relative_to(run_dir).as_posix()
        for path in sorted((run_dir / "eval").rglob("*"))
        if path.is_file()
    )
    bound_files = tuple(dict.fromkeys((*required, *training_files, *eval_files)))
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative in bound_files:
        path = run_dir / relative
        if not path.is_file():
            missing.append(relative)
        else:
            files[relative] = _sha256_file(path)
    if missing:
        raise ValueError(
            f"{run_dir}: cannot build completion manifest; missing bound files "
            f"{missing!r}"
        )
    return {
        "schema": COMPLETION_MANIFEST_SCHEMA,
        "hash_algorithm": "sha256",
        "inventory_policy": "complete-training-provenance-plus-complete-eval-tree",
        "files": files,
    }


def write_completion_manifest(run_dir: str | Path) -> Path:
    """Atomically bind every publication-critical per-run artifact by SHA-256."""

    run_dir = Path(run_dir)
    payload = _completion_manifest_payload(run_dir)
    destination = run_dir / COMPLETION_MANIFEST_NAME
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_completion_manifest(run_dir: str | Path) -> None:
    """Fail if a completed run changed after its atomic manifest was written."""

    run_dir = Path(run_dir)
    path = run_dir / COMPLETION_MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"{path} is missing")
    try:
        observed = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    expected = _completion_manifest_payload(run_dir)
    if observed == expected:
        return

    observed_files = observed.get("files", {}) if isinstance(observed, dict) else {}
    if not isinstance(observed_files, dict):
        observed_files = {}
    changed = [
        relative
        for relative, digest in expected["files"].items()
        if observed_files.get(relative) != digest
    ]
    extra = sorted(set(observed_files).difference(expected["files"]))
    details: list[str] = []
    if not isinstance(observed, dict) or observed.get("schema") != expected["schema"]:
        details.append("schema")
    if not isinstance(observed, dict) or observed.get("hash_algorithm") != "sha256":
        details.append("hash_algorithm")
    if (
        not isinstance(observed, dict)
        or observed.get("inventory_policy") != expected["inventory_policy"]
    ):
        details.append("inventory_policy")
    if changed:
        details.append(f"changed/missing files={changed!r}")
    if extra:
        details.append(f"unexpected files={extra!r}")
    raise ValueError(
        f"{path} does not match the current bound artifacts"
        + (f" ({'; '.join(details)})" if details else "")
    )


def _canonical_publication_config(run: "RunInfo") -> ExperimentConfig:
    """Resolve the exact current config stack for one publication identity."""

    if run.algo not in ALGO_ORDER:
        raise ValueError(f"unsupported publication algorithm {run.algo!r}")
    repo_root = Path(__file__).resolve().parents[3]
    config_root = repo_root / "configs"
    paths = [config_root / "base.yaml"]
    treatment = _SETTING_TO_TREATMENT.get(run.setting)
    if run.setting == HISTORICAL_SETTING:
        paths.append(config_root / "historical_sp500_midquotes.yaml")
    elif treatment is not None:
        paths.append(config_root / "synthetic_rough_heston.yaml")
    else:
        raise ValueError(f"unsupported publication setting {run.setting!r}")
    paths.append(config_root / "algo" / f"{run.algo}.yaml")
    if treatment is not None:
        overlay = _TREATMENT_OVERLAYS[treatment]
        if overlay is not None:
            paths.append(config_root / "treatment" / overlay)
    expected = load_config(
        *paths,
        overrides=[f"experiment.name={run.setting}"],
    )
    return dataclasses.replace(
        expected,
        experiment=dataclasses.replace(
            expected.experiment,
            master_seed=run.seed,
            seeds=(run.seed,),
        ),
    )


def _config_differences(expected: Any, observed: Any, prefix: str = "") -> list[str]:
    """Return concise dotted paths whose exact resolved values differ."""

    if isinstance(expected, dict) and isinstance(observed, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{path} (unexpected)")
            elif key not in observed:
                differences.append(f"{path} (missing)")
            else:
                differences.extend(
                    _config_differences(expected[key], observed[key], path)
                )
        return differences
    return [] if expected == observed else [prefix or "<root>"]


def _validate_canonical_publication_config(run: "RunInfo") -> None:
    if not isinstance(run.cfg, ExperimentConfig):
        raise ValueError("resolved publication config is not an ExperimentConfig")
    expected = to_dict(_canonical_publication_config(run))
    observed = to_dict(run.cfg)

    # The artifact root is operational rather than scientific.  Permit an
    # isolated/HPC root only when the supplied run directory is located exactly
    # at <saved results_root>/<setting>/<run name>; otherwise a stale or
    # misleading resolved path could be normalized away here.
    expected_results_root = expected["experiment"]["results_root"]
    observed_results_root = observed["experiment"].get("results_root")
    if observed_results_root != expected_results_root:
        repo_root = Path(__file__).resolve().parents[3]
        configured_root = Path(str(observed_results_root))
        if not configured_root.is_absolute():
            configured_root = repo_root / configured_root
        expected_parent = (configured_root / run.setting).resolve()
        actual_parent = Path(run.run_dir).resolve().parent
        if actual_parent != expected_parent:
            raise ValueError(
                "resolved experiment.results_root does not contain the run at "
                "<results_root>/<setting>/<run_name>: "
                f"saved root={observed_results_root!r}, run={str(run.run_dir)!r}"
            )
        expected["experiment"]["results_root"] = observed_results_root

    # The full launcher suppresses enormous replay-bearing resume checkpoints.
    # This operational cadence does not change learning, selection, or reported
    # outputs; it is the sole accepted deviation from the checked-in config stack.
    expected_interval = expected["algo"]["hyperparams"][
        "checkpoint_interval_episodes"
    ]
    observed_interval = observed["algo"]["hyperparams"].get(
        "checkpoint_interval_episodes"
    )
    if observed_interval == DISK_SAFE_CHECKPOINT_INTERVAL_EPISODES:
        observed["algo"]["hyperparams"][
            "checkpoint_interval_episodes"
        ] = expected_interval

    differences = _config_differences(expected, observed)
    if differences:
        preview = differences[:12]
        suffix = " ..." if len(differences) > len(preview) else ""
        raise ValueError(
            "resolved config differs from the current canonical publication "
            f"stack at {preview!r}{suffix}; accepted operational overrides are "
            "a structurally matching experiment.results_root and "
            "algo.hyperparams.checkpoint_interval_episodes=10000000"
        )


def _clean_head_sha() -> str:
    """Return HEAD only when the current repository state is fully attributable."""

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot resolve current Git HEAD for publication") from exc
    if not sha or dirty:
        raise ValueError(
            "publication validation requires a clean Git worktree and a resolvable HEAD"
        )
    return sha


def validate_publication_runs(
    runs: Iterable["RunInfo"], *, expected_git_sha: str | None = None
) -> None:
    """Reject development/smoke artifacts from publication aggregation.

    This checks the resolved maximum training budget and held-out evaluation
    size rather than guessing from directory names.  A mature ``best.pt`` is
    also required, and the saved seed tuple must identify exactly the executed
    master seed.
    """

    runs = list(runs)
    errors: list[str] = []
    from lmm.experiments.mature_reporting import load_protocol, validate_protocol_run
    protocol = load_protocol(runs)
    if protocol is not None:
        protocol_sha = protocol["training_git_sha"]
        if expected_git_sha is not None and expected_git_sha != protocol_sha:
            errors.append("requested training revision disagrees with reporting amendment")
        expected_git_sha = protocol_sha
    elif expected_git_sha is None:
        try:
            expected_git_sha = _clean_head_sha()
        except ValueError as exc:
            errors.append(str(exc))
    observed_master_seeds = {run.seed for run in runs}
    expected_master_seeds = set(PUBLICATION_SEEDS)
    if observed_master_seeds != expected_master_seeds:
        errors.append(
            "master-seed set is not the canonical ten-seed publication set: "
            f"expected {list(PUBLICATION_SEEDS)!r}, got "
            f"{sorted(observed_master_seeds)!r}"
        )

    # Publication aggregation is intentionally closed over the two canonical
    # settings. Cross-treatment input is the one supported mixed-setting case;
    # its exact six-arm matrix is checked by validate_treatment_run_configs.
    settings = {run.setting for run in runs}
    treatment_settings = {spec["setting"] for spec in TREATMENT_SPECS.values()}
    if settings == {"synthetic_rough_heston"}:
        expected_identities = {
            (algo, None, seed)
            for algo in ALGO_ORDER
            for seed in PUBLICATION_SEEDS
        }
        observed_identities = {
            (run.algo, run.symbol, run.seed) for run in runs
        }
        if observed_identities != expected_identities:
            missing = sorted(expected_identities - observed_identities, key=repr)
            extra = sorted(observed_identities - expected_identities, key=repr)
            errors.append(
                "canonical synthetic publication matrix must contain exactly "
                "four algorithms x ten seeds "
                f"(missing={missing!r}, extra={extra!r})"
            )
    elif settings == {HISTORICAL_SETTING}:
        configured_symbol_sets = {
            tuple(run.cfg.midprice.historical.symbols)
            for run in runs
            if run.cfg.midprice.historical is not None
        }
        if len(configured_symbol_sets) != 1:
            errors.append(
                "historical publication runs disagree on the configured symbol set"
            )
            configured_symbols: tuple[str, ...] = ()
        else:
            configured_symbols = next(iter(configured_symbol_sets))
        if len(configured_symbols) != 5 or len(set(configured_symbols)) != 5:
            errors.append(
                "historical publication config must contain exactly five distinct symbols; "
                f"got {configured_symbols!r}"
            )
        expected_identities = {
            (algo, symbol, seed)
            for algo in ALGO_ORDER
            for symbol in configured_symbols
            for seed in PUBLICATION_SEEDS
        }
        observed_identities = {
            (run.algo, run.symbol, run.seed) for run in runs
        }
        if observed_identities != expected_identities:
            missing = sorted(expected_identities - observed_identities, key=repr)
            extra = sorted(observed_identities - expected_identities, key=repr)
            errors.append(
                "canonical historical publication matrix must contain exactly "
                "four algorithms x five configured symbols x ten seeds "
                f"(missing={missing[:8]!r}, extra={extra[:8]!r})"
            )
    elif settings != treatment_settings:
        errors.append(
            "unsupported or partial publication setting set: "
            f"{sorted(settings)!r}"
        )
    for run in runs:
        prefix = str(run.run_dir)
        cfg = run.cfg
        try:
            _validate_canonical_publication_config(run)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")
        git_sha_path = Path(run.run_dir) / "git_sha.txt"
        saved_git_sha = (
            git_sha_path.read_text().strip() if git_sha_path.is_file() else None
        )
        if expected_git_sha is None or saved_git_sha != expected_git_sha:
            errors.append(
                f"{prefix}: git_sha.txt={saved_git_sha!r}, expected current clean "
                f"HEAD {expected_git_sha!r}"
            )
        if cfg.experiment.episodes != _PUBLICATION_EPISODES:
            errors.append(
                f"{prefix}: experiment.episodes={cfg.experiment.episodes}, "
                f"expected publication budget {_PUBLICATION_EPISODES}"
            )
        if cfg.rl.test_size != _PUBLICATION_EVAL_EPISODES:
            errors.append(
                f"{prefix}: rl.test_size={cfg.rl.test_size}, expected "
                f"{_PUBLICATION_EVAL_EPISODES}"
            )
        try:
            evaluated = int(run.metadata.get("n_episodes"))
        except (TypeError, ValueError):
            evaluated = -1
        if evaluated != _PUBLICATION_EVAL_EPISODES:
            errors.append(
                f"{prefix}: eval metadata n_episodes={run.metadata.get('n_episodes')!r}, "
                f"expected {_PUBLICATION_EVAL_EPISODES}"
            )
        if tuple(cfg.experiment.seeds) != (run.seed,):
            errors.append(
                f"{prefix}: resolved experiment.seeds={tuple(cfg.experiment.seeds)!r}, "
                f"expected ({run.seed},)"
            )
        checkpoint_name = "best"
        if protocol is not None:
            try:
                checkpoint_name = validate_protocol_run(protocol, run)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{prefix}: {exc}")
        if not (Path(run.run_dir) / "checkpoints" / f"{checkpoint_name}.pt").is_file():
            errors.append(f"{prefix}: missing mature reportable checkpoints/best.pt")
        expected_checkpoint = (Path(run.run_dir) / "checkpoints" / f"{checkpoint_name}.pt").resolve()
        recorded_checkpoint = run.metadata.get("checkpoint")
        if recorded_checkpoint is None:
            errors.append(f"{prefix}: eval metadata is missing checkpoint provenance")
        else:
            try:
                observed_checkpoint = Path(str(recorded_checkpoint)).resolve()
            except (TypeError, ValueError, OSError):
                observed_checkpoint = None
            if observed_checkpoint != expected_checkpoint:
                errors.append(
                    f"{prefix}: eval metadata checkpoint={recorded_checkpoint!r}, "
                    f"expected {str(Path(run.run_dir) / 'checkpoints' / 'best.pt')!r}"
                )
        if checkpoint_name == "best" and run.metadata.get("early_stopping") is not True:
            errors.append(
                f"{prefix}: eval metadata early_stopping must be true "
                "(publication evaluation must use best.pt)"
            )
        try:
            if checkpoint_name == "best":
                validate_completion_manifest(run.run_dir)
        except ValueError as exc:
            errors.append(str(exc))
        for benchmark in ("as", "twap"):
            difference = (
                Path(run.run_dir)
                / "eval"
                / f"policy_difference_{benchmark}.csv"
            )
            if not difference.is_file() or difference.stat().st_size == 0:
                errors.append(
                    f"{prefix}: missing paired-difference artifact {difference.name}"
                )
        if run.metadata.get("ablation_label") != cfg.experiment.ablation_label:
            errors.append(
                f"{prefix}: eval metadata ablation_label does not match resolved config"
            )
    if errors:
        raise ValueError("publication artifact validation failed:\n  " + "\n  ".join(errors))


def validate_completed_run(
    run: RunInfo,
    expected_cfg: ExperimentConfig,
    *,
    expected_symbol: str | None,
    expected_git_sha: str | None = None,
) -> None:
    """Fail closed unless a launcher may safely skip one completed run.

    This is stricter than existence checks: the resolved config must be exactly
    what the current command would write, evaluation must use ``best.pt`` on
    the configured held-out size, paired-difference outputs must exist, and an
    completion manifest written only after the whole per-run pipeline must
    cryptographically match every publication-critical artifact.
    """

    errors: list[str] = []
    prefix = str(run.run_dir)
    if expected_git_sha is None:
        try:
            expected_git_sha = _clean_head_sha()
        except ValueError as exc:
            errors.append(str(exc))
    git_sha_path = run.run_dir / "git_sha.txt"
    saved_git_sha = (
        git_sha_path.read_text().strip() if git_sha_path.is_file() else None
    )
    if expected_git_sha is None or saved_git_sha != expected_git_sha:
        errors.append(
            f"git_sha.txt={saved_git_sha!r}, expected current clean HEAD "
            f"{expected_git_sha!r}"
        )
    if to_dict(run.cfg) != to_dict(expected_cfg):
        errors.append("saved resolved config differs from the current launcher config")
    if run.seed != expected_cfg.experiment.master_seed:
        errors.append(
            f"saved master seed {run.seed} differs from expected "
            f"{expected_cfg.experiment.master_seed}"
        )
    if run.symbol != expected_symbol:
        errors.append(
            f"saved symbol {run.symbol!r} differs from expected {expected_symbol!r}"
        )
    try:
        n_episodes = int(run.metadata.get("n_episodes"))
    except (TypeError, ValueError):
        n_episodes = -1
    if n_episodes != expected_cfg.rl.test_size:
        errors.append(
            f"evaluation has {n_episodes} episodes; expected "
            f"{expected_cfg.rl.test_size}"
        )
    expected_checkpoint = (run.run_dir / "checkpoints" / "best.pt").resolve()
    recorded_checkpoint = run.metadata.get("checkpoint")
    observed_checkpoint = (
        Path(str(recorded_checkpoint)).resolve()
        if recorded_checkpoint is not None
        else None
    )
    if observed_checkpoint != expected_checkpoint:
        errors.append("evaluation checkpoint provenance is not this run's best.pt")
    if run.metadata.get("early_stopping") is not True:
        errors.append("evaluation metadata early_stopping is not true")
    if not expected_checkpoint.is_file():
        errors.append("checkpoints/best.pt is missing")
    for benchmark in ("as", "twap"):
        difference = run.run_dir / "eval" / f"policy_difference_{benchmark}.csv"
        if not difference.is_file() or difference.stat().st_size == 0:
            errors.append(f"missing paired-difference artifact {difference.name}")
    try:
        validate_completion_manifest(run.run_dir)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError(
            f"cannot skip incomplete or drifted run {prefix}:\n  "
            + "\n  ".join(errors)
            + "\nMove the partial run directory aside (preserving it for diagnosis) "
            "and rerun; automatic resume is intentionally disabled."
        )


def _treatment_defining_values(run: "RunInfo") -> dict[str, Any]:
    cfg = run.cfg
    return {
        "setting": cfg.experiment.name,
        "ablation_label": cfg.experiment.ablation_label,
        "h_cl": bool(cfg.rl.h_cl_feature_enabled),
        "auction_anchor": cfg.actions.auction_anchor,
        "shaping": bool(cfg.reward.shaping_enabled),
        "clob_shaping": bool(cfg.reward.effective_clob_shaping),
        "auction_shaping": bool(cfg.reward.effective_auction_shaping),
        "clawback_shaping": bool(cfg.reward.clawback_shaping),
        "auction": bool(cfg.experiment.auction_enabled),
        "cancellation": bool(
            cfg.experiment.auction_enabled
            and cfg.actions.auction_cancel_mode == "enabled"
        ),
    }


def _matched_config_payload(run: "RunInfo") -> dict[str, Any]:
    """Resolved config with only the intended treatment coordinates removed."""

    payload = dataclasses.asdict(run.cfg)
    permitted = (
        ("experiment", "name"),
        ("experiment", "ablation_label"),
        ("experiment", "auction_enabled"),
        ("rl", "h_cl_feature_enabled"),
        ("reward", "shaping_enabled"),
        ("reward", "clob_shaping_enabled"),
        ("reward", "auction_shaping_enabled"),
        ("reward", "clawback_shaping"),
        ("actions", "auction_anchor"),
        ("actions", "auction_cancel_mode"),
    )
    for section, field in permitted:
        payload[section].pop(field, None)
    return payload


def validate_treatment_run_configs(runs: Iterable["RunInfo"]) -> None:
    """Require the exact six-arm matched synthetic treatment specification.

    Besides checking the labels/switches, compare every non-treatment resolved
    field to the canonical headline run for the same algorithm/master seed.
    This catches a mislabeled directory or an accidental unmatched calibration.
    """

    runs = list(runs)
    errors: list[str] = []
    by_key: dict[tuple[str, str, int], "RunInfo"] = {}
    for run in runs:
        treatment = _SETTING_TO_TREATMENT.get(run.setting)
        if treatment is None:
            errors.append(f"{run.run_dir}: unexpected treatment setting {run.setting!r}")
            continue
        key = (treatment, run.algo, run.seed)
        if key in by_key:
            errors.append(f"duplicate treatment/algo/seed artifact: {key!r}")
            continue
        by_key[key] = run
        if run.symbol is not None:
            errors.append(f"{run.run_dir}: synthetic treatment run has symbol={run.symbol!r}")
        if run.cfg.experiment.setting != "synthetic_rough_heston":
            errors.append(
                f"{run.run_dir}: experiment.setting must remain 'synthetic_rough_heston'"
            )
        expected = TREATMENT_SPECS[treatment]
        observed = _treatment_defining_values(run)
        if observed != expected:
            errors.append(
                f"{run.run_dir}: treatment-defining config mismatch; "
                f"expected {expected!r}, got {observed!r}"
            )
        if run.metadata.get("ablation_label") != expected["ablation_label"]:
            errors.append(
                f"{run.run_dir}: eval metadata carries the wrong treatment label"
            )

    all_seeds = sorted({run.seed for run in runs})
    required = {
        (treatment, algo, seed)
        for treatment in TREATMENT_SPECS
        for algo in ALGO_ORDER
        for seed in all_seeds
    }
    missing = sorted(required.difference(by_key), key=repr)
    extra_algos = sorted({run.algo for run in runs}.difference(ALGO_ORDER))
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        suffix = " ..." if len(missing) > 8 else ""
        errors.append(f"incomplete treatment matrix; missing {preview}{suffix}")
    if extra_algos:
        errors.append(f"unexpected treatment algorithms: {extra_algos!r}")
    if len(all_seeds) < 2:
        errors.append("treatment contrasts require at least two master seeds")

    # All non-treatment fields must match the reused headline run within each
    # algorithm/master-seed block.
    for algo in ALGO_ORDER:
        for seed in all_seeds:
            headline = by_key.get((HEADLINE_TREATMENT, algo, seed))
            if headline is None:
                continue
            reference = _matched_config_payload(headline)
            for treatment in TREATMENT_SPECS:
                candidate = by_key.get((treatment, algo, seed))
                if candidate is not None and _matched_config_payload(candidate) != reference:
                    errors.append(
                        f"{candidate.run_dir}: non-treatment resolved config differs "
                        f"from headline for algorithm={algo}, seed={seed}"
                    )
    if errors:
        raise ValueError("treatment matrix validation failed:\n  " + "\n  ".join(errors))


def _learned_frame(run: "RunInfo", metric: str) -> pd.DataFrame:
    if run.records is None:
        raise ValueError(f"{run.run_dir}: missing evaluation records")
    required = {"policy", "episode", "env_seed", metric}
    missing = required.difference(run.records.columns)
    if missing:
        raise ValueError(f"{run.run_dir}: evaluation records lack {sorted(missing)!r}")
    frame = run.records.loc[
        run.records["policy"] == run.algo,
        ["episode", "env_seed", metric],
    ].copy()
    if frame.empty:
        raise ValueError(f"{run.run_dir}: no learned-policy rows labelled {run.algo!r}")
    if frame[["episode", "env_seed", metric]].isna().any().any():
        raise ValueError(f"{run.run_dir}: learned-policy records contain missing values")
    frame["episode"] = frame["episode"].astype(int)
    frame["env_seed"] = frame["env_seed"].astype(int)
    if frame["episode"].duplicated().any():
        raise ValueError(f"{run.run_dir}: duplicate learned-policy episode rows")
    return frame.sort_values("episode").reset_index(drop=True)


def _paired_episode_differences(
    positive: "RunInfo", negative: "RunInfo", metric: str
) -> tuple[np.ndarray, tuple[int, ...]]:
    a = _learned_frame(positive, metric)
    b = _learned_frame(negative, metric)
    a_episodes = tuple(int(v) for v in a["episode"])
    b_episodes = tuple(int(v) for v in b["episode"])
    if a_episodes != b_episodes:
        raise ValueError(
            "cross-treatment episode sets differ for "
            f"algorithm={positive.algo}, seed={positive.seed}: "
            f"{positive.setting} vs {negative.setting}"
        )
    a_seeds = tuple(int(v) for v in a["env_seed"])
    b_seeds = tuple(int(v) for v in b["env_seed"])
    if a_seeds != b_seeds:
        raise ValueError(
            "cross-treatment CRN violation: evaluation env_seed differs for "
            f"algorithm={positive.algo}, master seed={positive.seed}: "
            f"{positive.setting} vs {negative.setting}"
        )
    return a[metric].to_numpy(float) - b[metric].to_numpy(float), a_seeds


def build_treatment_contrast_records(
    runs: Iterable["RunInfo"], *, metric: str = "risk_adjusted_pnl"
) -> pd.DataFrame:
    """Return one paired mean difference per contrast/algorithm/master seed.

    Matrix/config validation is intentionally separate so fast unit tests can
    exercise the numerical pairing with lightweight ``RunInfo`` objects.  This
    function itself still requires the exact six-arm/four-algorithm balanced
    matrix and identical seed coverage.
    """

    runs = list(runs)
    matrix: dict[tuple[str, str, int], "RunInfo"] = {}
    for run in runs:
        treatment = _SETTING_TO_TREATMENT.get(run.setting)
        if treatment is None:
            raise ValueError(f"unexpected treatment setting {run.setting!r}")
        key = (treatment, run.algo, run.seed)
        if key in matrix:
            raise ValueError(f"duplicate treatment/algo/seed artifact: {key!r}")
        matrix[key] = run
    seed_sets: dict[tuple[str, str], set[int]] = {}
    for treatment in TREATMENT_SPECS:
        for algo in ALGO_ORDER:
            seed_sets[(treatment, algo)] = {
                seed
                for (arm, candidate_algo, seed) in matrix
                if arm == treatment and candidate_algo == algo
            }
    unique_seed_sets = {tuple(sorted(seeds)) for seeds in seed_sets.values()}
    if len(unique_seed_sets) != 1 or not unique_seed_sets:
        raise ValueError(
            "treatment matrix must have identical master-seed coverage for every arm/algorithm"
        )
    seeds = next(iter(unique_seed_sets))
    if len(seeds) < 2:
        raise ValueError("treatment contrasts require at least two master seeds")
    if any(not values for values in seed_sets.values()):
        raise ValueError("treatment matrix is missing an arm/algorithm block")

    rows: list[dict[str, Any]] = []
    for contrast_key, label, positive_arm, negative_arm in TREATMENT_CONTRASTS:
        for algo in ALGO_ORDER:
            for seed in seeds:
                positive = matrix[(positive_arm, algo, seed)]
                negative = matrix[(negative_arm, algo, seed)]
                differences, env_seeds = _paired_episode_differences(
                    positive, negative, metric
                )
                seed_digest = hashlib.sha256(
                    ",".join(map(str, env_seeds)).encode("ascii")
                ).hexdigest()
                rows.append(
                    {
                        "contrast_key": contrast_key,
                        "contrast": label,
                        "positive_treatment": positive_arm,
                        "negative_treatment": negative_arm,
                        "algorithm": algo,
                        "master_seed": seed,
                        "n_episodes": int(differences.size),
                        "metric": metric,
                        "mean_difference": float(np.mean(differences)),
                        "evaluation_seed_sha256": seed_digest,
                        "positive_run_dir": str(positive.run_dir),
                        "negative_run_dir": str(negative.run_dir),
                    }
                )
    return pd.DataFrame(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmm-publication",
        description="Write or validate a cryptographically complete run.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-completed-run", metavar="RUN_DIR")
    mode.add_argument("--write-completion-manifest", metavar="RUN_DIR")
    parser.add_argument("--config", action="append", metavar="YAML")
    parser.add_argument("--override", "-o", action="append", default=[])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--symbol", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.write_completion_manifest is not None:
        if args.config or args.override or args.seed is not None or args.symbol is not None:
            parser.error(
                "--write-completion-manifest accepts only its run directory"
            )
        try:
            path = write_completion_manifest(args.write_completion_manifest)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"wrote completion manifest: {path}")
        return 0

    if not args.config or args.seed is None:
        parser.error("--validate-completed-run requires --config and --seed")
    expected = load_config(*args.config, overrides=args.override)
    expected = dataclasses.replace(
        expected,
        experiment=dataclasses.replace(
            expected.experiment,
            master_seed=args.seed,
            seeds=(args.seed,),
        ),
    )
    try:
        runs = collect_runs([Path(args.validate_completed_run)])
        if len(runs) != 1:
            raise ValueError("completed-run validation resolved no unique run")
        validate_completed_run(
            runs[0], expected, expected_symbol=args.symbol
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"validated completed run: {args.validate_completed_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
