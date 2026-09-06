"""Negative results stay included without bypassing maturity or provenance."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch
import yaml

from lmm.config import load_config
from lmm.experiments import mature_reporting as M, publication as P
from lmm.experiments import plotting


@pytest.fixture
def mature(tmp_path):
    rd = tmp_path / "synthetic_rough_heston__no_auction/dqn__no_auction_seed123"
    (rd / "checkpoints").mkdir(parents=True)
    cfg = load_config(M.REPO / "configs/base.yaml", M.REPO / "configs/synthetic_rough_heston.yaml",
                      M.REPO / "configs/algo/dqn.yaml", M.REPO / "configs/treatment/no_auction.yaml",
                      overrides=['rl.checkpoint_min_clob_updates=5000',
                                 'rl.checkpoint_require_initial_improvement=true'])
    selected = dict(metric=cfg.rl.checkpoint_metric, mode="max", seed_stream="env_eval",
                    value=12., episode=499, validation_seeds=list(range(cfg.rl.validation_size)),
                    eligibility=dict(eligible=True, required_updates={"clob": 5000, "auction": 0},
                                     observed_updates={"clob": 35000, "auction": 0}),
                    economic_safety=dict(reportable=False, initial_validation_score=13.))
    (rd / "checkpoints/best_mature_selection.yaml").write_text(yaml.safe_dump(selected))
    (rd / "checkpoints/initial_validation.yaml").write_text(
        yaml.safe_dump(dict(value=13., validation_seeds=selected["validation_seeds"])))
    (rd / "metrics.csv").write_text(
        "episode,eval_checkpoint_eligible,eval_checkpoint_score\n99,0,90\n499,1,12\n799,1,11\n")
    return rd, cfg, selected


def test_non_improvement_is_valid_but_immaturity_is_not(mature):
    rd, cfg, selected = mature
    assert M.validate_mature_selection(rd, cfg)["economic_safety"]["reportable"] is False
    selected["eligibility"]["observed_updates"]["clob"] = 4
    (rd / "checkpoints/best_mature_selection.yaml").write_text(yaml.safe_dump(selected))
    with pytest.raises(ValueError, match="thresholds"):
        M.validate_mature_selection(rd, cfg)


def test_cannot_choose_ineligible_argmax_or_infer_a_later_checkpoint(mature):
    rd, cfg, selected = mature
    selected.update(episode=799, value=11.)
    (rd / "checkpoints/best_mature_selection.yaml").write_text(yaml.safe_dump(selected))
    with pytest.raises(ValueError, match="best observed mature"):
        M.validate_mature_selection(rd, cfg)


def test_checkpoint_equality_ignores_only_archive_timestamps(tmp_path):
    def archive(year, content):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr(zipfile.ZipInfo("policy.pth", (year, 1, 1, 0, 0, 0)), content)
        return stream.getvalue()
    a, b = tmp_path / "best.pt", tmp_path / "best_mature.pt"
    torch.save({"model": archive(2025, b"same"), "weights": torch.tensor([1.])}, a)
    torch.save({"model": archive(2026, b"same"), "weights": torch.tensor([1.])}, b)
    assert M.equivalent_checkpoints(a, b)
    torch.save({"model": archive(2026, b"different"), "weights": torch.tensor([1.])}, b)
    assert not M.equivalent_checkpoints(a, b)
    torch.save({"model": archive(2026, b"same"), "weights": torch.tensor([2.])}, b)
    assert not M.equivalent_checkpoints(a, b)


def test_protocol_requires_full_matrix_and_unchanged_source(tmp_path, monkeypatch):
    run = SimpleNamespace(run_dir=tmp_path / "setting/run")
    path = tmp_path / M.RELATIVE_PATH
    path.parent.mkdir()
    monkeypatch.setattr(M, "source_inventory", lambda: {"train.py": "original"})
    protocol = dict(schema=M.SCHEMA, rule=M.RULE, source_files={"train.py": "original"}, runs={})
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="220"):
        M.load_protocol([run])
    protocol["runs"] = {str(i): {} for i in range(220)}
    path.write_text(json.dumps(protocol))
    assert M.load_protocol([run])["rule"] == M.RULE
    monkeypatch.setattr(M, "source_inventory", lambda: {"train.py": "changed"})
    with pytest.raises(ValueError, match="source files changed"):
        M.load_protocol([run])


def test_amended_completion_binds_original_failure_and_eval(mature, monkeypatch):
    rd, cfg, selection = mature
    from lmm.config import save_resolved
    save_resolved(cfg, rd / "config_resolved.yaml")
    # Materialize required file inventory without creating a fictional best.pt.
    for name in P._COMPLETION_REQUIRED_FILES:
        if name in {"checkpoints/best.pt", "checkpoints/best_selection.yaml"}:
            continue
        p = rd / name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture")
    for policy in ("dqn", "initial", "as", "twap"):
        p = rd / f"eval/traces/{policy}_ep0.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("t,phase\n0,clob\n")
    failure = rd / "checkpoints/selection_failure.yaml"
    failure.write_text("reason: non-improvement\n")
    entry = dict(selected_episode=499, validation_score=12., evaluated_checkpoint="best_mature",
                 completion=P._completion_manifest_payload(rd, checkpoint="best_mature"))
    root = rd.parents[1]
    protocol = {"_root": str(root), "runs": {rd.relative_to(root).as_posix(): entry}}
    run = SimpleNamespace(run_dir=rd, cfg=cfg, metadata={"checkpoint_selection_rule": M.RULE,
                                                      "checkpoint_selection": selection})
    assert M.validate_protocol_run(protocol, run) == "best_mature"
    with pytest.raises(ValueError, match="missing bound files"):
        P._completion_manifest_payload(rd)
    failure.write_text("reason: silently changed\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        M.validate_protocol_run(protocol, run)


def test_missing_evaluation_is_not_mislabeled_as_old_contract(mature):
    rd, cfg, _ = mature
    from lmm.config import save_resolved
    save_resolved(cfg, rd / "config_resolved.yaml")
    (rd / "checkpoints/selection_failure.yaml").write_text("reason: non-improvement\n")
    with pytest.raises(ValueError, match="missing eval/metadata.yaml.*checkpoint selection failed"):
        plotting.collect_runs([rd])
