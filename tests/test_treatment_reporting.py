"""Paired, common-random-number reporting across synthetic treatments."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from lmm.config import load_config, save_resolved
from lmm.experiments import publication
from lmm.experiments import tables as T
from lmm.experiments.plotting import ALGO_ORDER, RunInfo


_ARM_LEVEL = {
    "headline": 100.0,
    "h_off_shaping_off": 10.0,
    "h_on_shaping_off": 30.0,
    "h_off_shaping_on": 50.0,
    "no_auction": 20.0,
    "no_cancellation": 80.0,
}
_REPO = Path(__file__).resolve().parents[1]


def _canonical_run_config(setting: str, algo: str, seed: int):
    paths = [_REPO / "configs/base.yaml"]
    if setting == "synthetic_rough_heston":
        paths.append(_REPO / "configs/synthetic_rough_heston.yaml")
    elif setting == "historical_sp500_midquotes":
        paths.append(_REPO / "configs/historical_sp500_midquotes.yaml")
    else:
        raise AssertionError(setting)
    paths.append(_REPO / "configs/algo" / f"{algo}.yaml")
    cfg = load_config(*paths)
    return dataclasses.replace(
        cfg,
        experiment=dataclasses.replace(
            cfg.experiment,
            master_seed=seed,
            seeds=(seed,),
        ),
    )


def _materialize_completion(run_dir: Path, cfg, metadata: dict) -> None:
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "eval" / "traces").mkdir(parents=True)
    save_resolved(cfg, run_dir / "config_resolved.yaml")
    (run_dir / "seed.txt").write_text(f"{cfg.experiment.master_seed}\n")
    (run_dir / "git_sha.txt").write_text("test-head\n")
    (run_dir / "runtime_versions.json").write_text('{"python": "test"}\n')
    if cfg.midprice.historical is not None:
        (run_dir / "historical_data_manifest.json").write_text(
            '{"schema": "test"}\n'
        )
    (run_dir / "split_seeds.yaml").write_text(
        yaml.safe_dump({"master_seed": cfg.experiment.master_seed})
    )
    (run_dir / "metrics.csv").write_text("episode,return_undisc\n0,1.0\n")
    (run_dir / "realized_grids.jsonl").write_text('{"episode":0}\n')
    (run_dir / "h_forecasts_train.csv").write_text("episode,h_cl\n0,100.0\n")
    (run_dir / "feature_normalizer.yaml").write_text("fitted: true\n")
    (run_dir / "checkpoints" / "initial.pt").write_bytes(b"initial")
    (run_dir / "checkpoints" / "initial_validation.yaml").write_text(
        "metric: risk_adjusted_pnl\n"
    )
    (run_dir / "checkpoints" / "best_mature.pt").write_bytes(b"best-mature")
    (run_dir / "checkpoints" / "best_mature_selection.yaml").write_text(
        "metric: risk_adjusted_pnl\n"
    )
    (run_dir / "checkpoints" / "best.pt").write_bytes(
        f"best:{cfg.algo.name}:{cfg.experiment.master_seed}".encode()
    )
    (run_dir / "checkpoints" / "final.pt").write_bytes(b"final")
    (run_dir / "checkpoints" / "best_selection.yaml").write_text(
        yaml.safe_dump(
            {
                "metric": cfg.rl.checkpoint_metric,
                "episode": 100,
                "eligibility": {"eligible": True},
                "economic_safety": {"reportable": True},
            }
        )
    )
    (run_dir / "eval" / "records.csv").write_text(
        "policy,episode,env_seed,risk_adjusted_pnl\n"
        f"{cfg.algo.name},0,123,1.0\n"
    )
    (run_dir / "eval" / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False)
    )
    for benchmark in ("as", "twap"):
        (run_dir / "eval" / f"policy_difference_{benchmark}.csv").write_text(
            "episode,policy_minus_benchmark\n0,1\n"
        )
    for policy in (cfg.algo.name, "initial", "as", "twap"):
        (run_dir / "eval" / "traces" / f"{policy}_ep0.csv").write_text(
            "t,phase\n0,clob\n"
        )
    publication.write_completion_manifest(run_dir)


def _records(algo: str, value: float, *, corrupt_crn: bool = False) -> pd.DataFrame:
    rows = []
    for episode in range(4):
        env_seed = 10_000 + episode + (1 if corrupt_crn and episode == 0 else 0)
        rows.append(
            {
                "policy": algo,
                "episode": episode,
                "env_seed": env_seed,
                "risk_adjusted_pnl": value + episode,
            }
        )
    return pd.DataFrame(rows)


def _matrix() -> list[RunInfo]:
    runs = []
    for treatment, spec in publication.TREATMENT_SPECS.items():
        for algo_index, algo in enumerate(ALGO_ORDER):
            for seed in (7, 42, 99, 123, 2024):
                value = _ARM_LEVEL[treatment] + 1_000.0 * algo_index + seed
                runs.append(
                    RunInfo(
                        run_dir=Path(f"/{treatment}/{algo}/{seed}"),
                        cfg=None,
                        setting=spec["setting"],
                        algo=algo,
                        symbol=None,
                        records=_records(algo, value),
                        metadata={},
                        seed=seed,
                    )
                )
    return runs


def test_cross_treatment_table_reports_explicit_paired_contrasts():
    table, provenance = T.build_synthetic_treatment_contrasts(_matrix(), n_boot=100)

    assert table.columns == ["DQN", "DDPG", "TD3", "SAC"]
    assert len(table.rows) == 6
    assert provenance.shape[0] == 6 * 4 * 5
    assert set(provenance["n_episodes"]) == {4}
    assert provenance["evaluation_seed_sha256"].str.len().eq(64).all()

    expected = {
        "H/anchor effect, shaping off (on - off)": 20.0,
        "H/anchor effect, shaping on (on - off)": 50.0,
        "Shaping effect, H/anchor off (on - off)": 40.0,
        "Shaping effect, H/anchor on (on - off)": 70.0,
        "Full auction-aware treatment vs no-auction comparator": 80.0,
        "Cancellation effect (on - off)": 20.0,
    }
    for row in table.rows:
        assert row.label in expected
        for point, lo, hi in row.cells:
            assert point == pytest.approx(expected[row.label])
            assert lo == pytest.approx(expected[row.label])
            assert hi == pytest.approx(expected[row.label])
    assert "first-named condition minus second-named condition" in table.caption
    assert "headline" in table.caption


def test_cross_treatment_pairing_rejects_env_seed_mismatch():
    runs = _matrix()
    target = next(
        run
        for run in runs
        if run.setting.endswith("ablation_h_off_shaping_on")
        and run.algo == "dqn"
        and run.seed == 42
    )
    target.records = _records("dqn", 50.0 + 42, corrupt_crn=True)
    with pytest.raises(ValueError, match="cross-treatment CRN violation"):
        publication.build_treatment_contrast_records(runs)


def test_cross_treatment_pairing_requires_balanced_six_arm_matrix():
    runs = _matrix()
    runs.pop()
    with pytest.raises(ValueError, match="identical master-seed coverage"):
        publication.build_treatment_contrast_records(runs)


@pytest.fixture(scope="module")
def configured_treatment_matrix() -> list[RunInfo]:
    runs = []
    for treatment, spec in publication.TREATMENT_SPECS.items():
        arm = None if treatment == "headline" else spec["setting"].split("__", 1)[1]
        for algo in ALGO_ORDER:
            paths = [
                _REPO / "configs/base.yaml",
                _REPO / "configs/synthetic_rough_heston.yaml",
                _REPO / "configs/algo" / f"{algo}.yaml",
            ]
            if arm is not None:
                paths.append(_REPO / "configs/treatment" / f"{arm}.yaml")
            base_cfg = load_config(
                *paths, overrides=[f"experiment.name={spec['setting']}"]
            )
            for seed in publication.PUBLICATION_SEEDS:
                cfg = dataclasses.replace(
                    base_cfg,
                    experiment=dataclasses.replace(
                        base_cfg.experiment,
                        master_seed=seed,
                        seeds=(seed,),
                    ),
                )
                runs.append(
                    RunInfo(
                        run_dir=Path(f"/{treatment}/{algo}/{seed}"),
                        cfg=cfg,
                        setting=cfg.experiment.name,
                        algo=algo,
                        symbol=None,
                        records=None,
                        metadata={"ablation_label": cfg.experiment.ablation_label},
                        seed=seed,
                    )
                )
    return runs


@pytest.mark.parametrize(
    ("setting_suffix", "reward_change"),
    [
        ("synthetic_rough_heston", {"clob_shaping_enabled": False}),
        (
            "synthetic_rough_heston__ablation_h_off_shaping_off",
            {"clawback_shaping": False},
        ),
    ],
)
def test_treatment_config_validation_rejects_phase_or_clawback_drift(
    configured_treatment_matrix, setting_suffix, reward_change
):
    runs = list(configured_treatment_matrix)
    index = next(
        i
        for i, run in enumerate(runs)
        if run.setting == setting_suffix and run.algo == "dqn" and run.seed == 42
    )
    original = runs[index]
    cfg = dataclasses.replace(
        original.cfg,
        reward=dataclasses.replace(original.cfg.reward, **reward_change),
    )
    runs[index] = dataclasses.replace(original, cfg=cfg)
    with pytest.raises(ValueError, match="treatment-defining config mismatch"):
        publication.validate_treatment_run_configs(runs)


def test_current_treatment_overlays_form_exact_matched_matrix(
    configured_treatment_matrix,
):
    publication.validate_treatment_run_configs(configured_treatment_matrix)
    for run in configured_treatment_matrix:
        publication._validate_canonical_publication_config(run)


def test_publication_validation_rejects_smoke_budget(tmp_path):
    run_dir = tmp_path / "smoke"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").touch()
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            episodes=4,
            seeds=(42,),
            ablation_label="H_on__shaping_on__auction_on",
        ),
        rl=SimpleNamespace(test_size=3),
    )
    run = SimpleNamespace(
        run_dir=run_dir,
        cfg=cfg,
        setting="synthetic_rough_heston",
        algo="dqn",
        symbol=None,
        seed=42,
        metadata={
            "n_episodes": 3,
            "ablation_label": cfg.experiment.ablation_label,
            "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
            "early_stopping": True,
        },
    )
    with pytest.raises(ValueError, match="publication artifact validation failed"):
        publication.validate_publication_runs([run], expected_git_sha="test-head")


def _publication_runs(tmp_path):
    runs = []
    for algo in ALGO_ORDER:
        for seed in publication.PUBLICATION_SEEDS:
            run_dir = tmp_path / f"{algo}_seed{seed}"
            cfg = _canonical_run_config("synthetic_rough_heston", algo, seed)
            label = cfg.experiment.ablation_label
            metadata = {
                "n_episodes": 100,
                "ablation_label": label,
                "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
                "early_stopping": True,
            }
            _materialize_completion(run_dir, cfg, metadata)
            runs.append(
                RunInfo(
                    run_dir=run_dir,
                    cfg=cfg,
                    setting="synthetic_rough_heston",
                    algo=algo,
                    symbol=None,
                    records=None,
                    seed=seed,
                    metadata=metadata,
                )
            )
    return runs


def test_publication_validation_requires_best_checkpoint_evaluation(tmp_path):
    runs = _publication_runs(tmp_path)
    publication.validate_publication_runs(runs, expected_git_sha="test-head")

    runs[0].metadata["checkpoint"] = str(
        runs[0].run_dir / "checkpoints" / "final.pt"
    )
    runs[0].metadata["early_stopping"] = False
    with pytest.raises(ValueError, match="publication evaluation must use best.pt"):
        publication.validate_publication_runs(runs, expected_git_sha="test-head")


def test_completed_run_skip_requires_bound_manifest_and_derived_outputs(tmp_path):
    run = _publication_runs(tmp_path)[0]
    publication.validate_completed_run(
        run, run.cfg, expected_symbol=None, expected_git_sha="test-head"
    )

    (run.run_dir / "eval" / "records.csv").write_text(
        "policy,episode,env_seed,risk_adjusted_pnl\ndqn,0,123,999.0\n"
    )
    with pytest.raises(ValueError, match="Move the partial run directory aside"):
        publication.validate_completed_run(
            run, run.cfg, expected_symbol=None, expected_git_sha="test-head"
        )


def test_completion_manifest_binds_every_required_publication_artifact(tmp_path):
    run = _publication_runs(tmp_path)[0]
    manifest = json.loads(
        (run.run_dir / publication.COMPLETION_MANIFEST_NAME).read_text()
    )
    assert manifest["schema"] == publication.COMPLETION_MANIFEST_SCHEMA
    assert manifest["hash_algorithm"] == "sha256"
    assert manifest["inventory_policy"] == (
        "complete-training-provenance-plus-complete-eval-tree"
    )
    assert set(manifest["files"]) == {
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
        f"eval/traces/{run.algo}_ep0.csv",
        "eval/traces/initial_ep0.csv",
        "eval/traces/as_ep0.csv",
        "eval/traces/twap_ep0.csv",
    }
    assert all(len(digest) == 64 for digest in manifest["files"].values())
    assert not list(run.run_dir.glob(".pipeline_complete.json.tmp-*"))

    extra_evaluation = run.run_dir / "eval" / "new_diagnostic.csv"
    extra_evaluation.write_text("value\n1\n")
    with pytest.raises(ValueError, match="changed/missing files"):
        publication.validate_completion_manifest(run.run_dir)
    extra_evaluation.unlink()
    publication.validate_completion_manifest(run.run_dir)

    # Training diagnostics are part of the same atomic inventory, not just the
    # files explicitly consumed by current table builders.
    (run.run_dir / "h_forecasts_train.csv").unlink()
    with pytest.raises(ValueError, match="missing bound files"):
        publication.validate_completion_manifest(run.run_dir)
    (run.run_dir / "h_forecasts_train.csv").write_text(
        "episode,h_cl\n0,100.0\n"
    )
    publication.validate_completion_manifest(run.run_dir)

    (run.run_dir / "checkpoints" / "best_selection.yaml").write_text(
        "metric: tampered\n"
    )
    with pytest.raises(ValueError, match="changed/missing files"):
        publication.validate_completion_manifest(run.run_dir)


def test_publication_validation_rejects_partial_algorithm_matrix(tmp_path):
    runs = _publication_runs(tmp_path)
    runs.pop()
    with pytest.raises(ValueError, match="four algorithms x five seeds"):
        publication.validate_publication_runs(runs, expected_git_sha="test-head")


def test_publication_validation_rejects_noncanonical_resolved_config(tmp_path):
    runs = _publication_runs(tmp_path)
    original = runs[0]
    drifted = dataclasses.replace(
        original.cfg,
        grid=dataclasses.replace(original.cfg.grid, S0=101.0),
    )
    save_resolved(drifted, original.run_dir / "config_resolved.yaml")
    publication.write_completion_manifest(original.run_dir)
    runs[0] = dataclasses.replace(original, cfg=drifted)

    with pytest.raises(ValueError, match="current canonical publication stack"):
        publication.validate_publication_runs(runs, expected_git_sha="test-head")


def test_publication_validation_accepts_disk_safe_checkpoint_override(tmp_path):
    runs = _publication_runs(tmp_path)
    original = runs[0]
    hyperparams = dict(original.cfg.algo.hyperparams)
    hyperparams["checkpoint_interval_episodes"] = (
        publication.DISK_SAFE_CHECKPOINT_INTERVAL_EPISODES
    )
    disk_safe = dataclasses.replace(
        original.cfg,
        algo=dataclasses.replace(original.cfg.algo, hyperparams=hyperparams),
    )
    save_resolved(disk_safe, original.run_dir / "config_resolved.yaml")
    publication.write_completion_manifest(original.run_dir)
    runs[0] = dataclasses.replace(original, cfg=disk_safe)

    publication.validate_publication_runs(runs, expected_git_sha="test-head")


def test_publication_validation_accepts_structurally_matching_results_root(tmp_path):
    runs = _publication_runs(tmp_path)
    original = runs[0]
    isolated_root = tmp_path / "isolated-results"
    relocated_run_dir = isolated_root / original.setting / original.run_dir.name
    relocated_run_dir.parent.mkdir(parents=True)
    original.run_dir.rename(relocated_run_dir)
    isolated = dataclasses.replace(
        original.cfg,
        experiment=dataclasses.replace(
            original.cfg.experiment, results_root=isolated_root
        ),
    )
    metadata = dict(original.metadata)
    metadata["checkpoint"] = str(relocated_run_dir / "checkpoints" / "best.pt")
    save_resolved(isolated, relocated_run_dir / "config_resolved.yaml")
    (relocated_run_dir / "eval" / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False)
    )
    publication.write_completion_manifest(relocated_run_dir)
    runs[0] = dataclasses.replace(
        original, run_dir=relocated_run_dir, cfg=isolated, metadata=metadata
    )

    publication.validate_publication_runs(runs, expected_git_sha="test-head")


def test_publication_validation_rejects_misleading_results_root(tmp_path):
    runs = _publication_runs(tmp_path)
    original = runs[0]
    misleading = dataclasses.replace(
        original.cfg,
        experiment=dataclasses.replace(
            original.cfg.experiment, results_root=tmp_path / "somewhere-else"
        ),
    )
    save_resolved(misleading, original.run_dir / "config_resolved.yaml")
    publication.write_completion_manifest(original.run_dir)
    runs[0] = dataclasses.replace(original, cfg=misleading)

    with pytest.raises(ValueError, match="does not contain the run"):
        publication.validate_publication_runs(runs, expected_git_sha="test-head")


def test_publication_validation_rejects_partial_historical_ticker_matrix(tmp_path):
    symbols = ("MSFT", "JPM", "PG", "GOOGL", "CAT")
    runs = []
    for algo in ALGO_ORDER:
        for symbol in symbols:
            for seed in publication.PUBLICATION_SEEDS:
                run_dir = tmp_path / f"{algo}_{symbol}_seed{seed}"
                cfg = _canonical_run_config(
                    "historical_sp500_midquotes", algo, seed
                )
                label = cfg.experiment.ablation_label
                metadata = {
                    "n_episodes": 100,
                    "ablation_label": label,
                    "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
                    "early_stopping": True,
                }
                _materialize_completion(run_dir, cfg, metadata)
                runs.append(
                    RunInfo(
                        run_dir=run_dir,
                        cfg=cfg,
                        setting="historical_sp500_midquotes",
                        algo=algo,
                        symbol=symbol,
                        records=None,
                        seed=seed,
                        metadata=metadata,
                    )
                )
    publication.validate_publication_runs(runs, expected_git_sha="test-head")
    runs.pop()
    with pytest.raises(ValueError, match="four algorithms x five configured symbols"):
        publication.validate_publication_runs(runs, expected_git_sha="test-head")
