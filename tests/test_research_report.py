"""Scientific reporting contracts: estimands, pairing, accounting and output scope."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lmm.agents.base import ENVIRONMENT_CONTRACT
from lmm.config import load_config, save_resolved
from lmm.experiments import make_report as R, publication
from lmm.experiments.plotting import RunInfo

REPO = Path(__file__).resolve().parents[1]


def example_run(tmp_path, algo="dqn", seed=7, *, setting=R.SYNTHETIC, symbol=None, materialize=False):
    base_setting = R.P.HISTORICAL_SETTING if symbol else R.SYNTHETIC
    paths = [REPO / "configs/base.yaml", REPO / f"configs/{base_setting}.yaml", REPO / f"configs/algo/{algo}.yaml"]
    if "__" in setting:
        paths.append(REPO / f"configs/treatment/{setting.split('__', 1)[1]}.yaml")
    cfg = load_config(*paths)
    cfg = dataclasses.replace(cfg, experiment=dataclasses.replace(cfg.experiment, name=setting, master_seed=seed, seeds=(seed,)))
    rd = tmp_path / setting / f"{algo}{'_' + symbol if symbol else ''}_seed{seed}"
    rows = []
    for policy in (algo, "initial", "as", "twap"):
        for ep in range(3):
            # Include a signed over-sale episode and a costly auction episode.
            open_i = [5., 20., 2.][ep]
            z = [3., 22., -2.][ep] if cfg.experiment.auction_enabled else 0.
            final = open_i - z
            clearing = 100 + [.2, -.1, .3][ep]
            fee = .02 if cfg.actions.auction_cancel_mode == "enabled" else 0.
            pnl = (4 if policy in R.ALGOS else 2) + .5 * ep + .002 * seed
            objective = pnl - cfg.reward.lambda_inv * final ** 2
            rows.append(dict(policy=policy, episode=ep, env_seed=10000 + seed + ep,
                             initial_mid=100., initial_inventory=100., clob_exec_qty=100-open_i,
                             auction_exec_qty=z, I_final=final, S_cl=clearing,
                             residual_liquidation_price=100., cancel_cost=fee,
                             inventory_penalty=cfg.reward.lambda_inv * final**2,
                             pnl=pnl, pnl_bps=pnl, risk_adjusted_pnl=objective, risk_adjusted_pnl_bps=objective))
    meta = dict(master_seed=seed, symbol=symbol, environment_contract=ENVIRONMENT_CONTRACT,
                ablation_label=cfg.experiment.ablation_label,
                artifact_schema_version=cfg.experiment.artifact_schema_version,
                n_episodes=3, policies=[algo, "initial", "as", "twap"],
                learned_policy_label=algo, evaluation_episode_seeds=[10000+seed+ep for ep in range(3)])
    data = pd.DataFrame(rows)
    if materialize:
        (rd / "eval").mkdir(parents=True)
        (rd / "checkpoints").mkdir()
        save_resolved(cfg, rd / "config_resolved.yaml")
        (rd / "seed.txt").write_text(str(seed))
        (rd / "git_sha.txt").write_text("fixture-not-trained")
        (rd / "eval/metadata.yaml").write_text(yaml.safe_dump(meta))
        data.to_csv(rd / "eval/records.csv", index=False)
        # Full figure layout without running 800 training episodes.
        pd.DataFrame({"episode": [99, 199, 299, 399, 499, 599, 699, 799],
                      "eval_risk_adjusted_pnl_mean": np.linspace(1, 7, 8) + seed*.001}).to_csv(rd / "metrics.csv", index=False)
        (rd / "checkpoints/initial_validation.yaml").write_text("metric: risk_adjusted_pnl\nvalue: -1.0\nvalidation_seeds: [1, 2, 3]\n")
        (rd / "checkpoints/best.pt").write_bytes(b"fixture")
        (rd / "checkpoints/best_selection.yaml").write_text(yaml.safe_dump({
            "metric": "risk_adjusted_pnl", "episode": 799,
            "eligibility": {"eligible": True}, "economic_safety": {"reportable": True}}))
    return RunInfo(rd, cfg, setting, algo, symbol, data, meta, seed)


def materialize_matrix(tmp_path, seeds=(9501, 9502)):
    for seed in seeds:
        for algo in R.ALGOS:
            for spec in publication.TREATMENT_SPECS.values():
                example_run(tmp_path, algo, seed, setting=spec["setting"], materialize=True)
            for symbol in ["MSFT", "JPM", "PG", "GOOGL", "CAT"]:
                example_run(tmp_path, algo, seed, setting=R.P.HISTORICAL_SETTING, symbol=symbol, materialize=True)


def test_auction_contribution_matches_independent_no_order_settlement(tmp_path):
    run = example_run(tmp_path)
    raw = R.frame(run, "dqn")
    data = R.economic_rows(run, "dqn")
    opened = raw.initial_inventory - raw.clob_exec_qty
    # Recover the non-auction cash term from the actual objective, then settle
    # the same inventory at the frozen mark with no fees and no auction fills.
    common_cash = (raw.risk_adjusted_pnl - raw.S_cl*raw.auction_exec_qty
                   - raw.residual_liquidation_price*raw.I_final
                   + raw.cancel_cost + raw.inventory_penalty)
    noop = common_cash + raw.residual_liquidation_price*opened - run.cfg.reward.lambda_inv*opened**2
    np.testing.assert_allclose(data.auction_value_bps, raw.risk_adjusted_pnl-noop)
    np.testing.assert_allclose(data.auction_value_bps, data.auction_cash_bps + data.auction_risk_relief_bps)
    assert (raw.I_final < 0).any()
    assert (data.auction_value_bps < 0).any()  # never clip adverse outcomes


def test_accounting_corruption_is_rejected(tmp_path):
    run = example_run(tmp_path)
    run.records.loc[0, "I_final"] += 1
    with pytest.raises(ValueError, match="conservation"):
        R.economic_rows(run, "dqn")


def test_reference_policies_are_not_counted_four_times(tmp_path):
    runs = [example_run(tmp_path, a) for a in R.ALGOS]
    outcomes = R.seed_outcomes(runs)
    assert len(outcomes) == 6
    np.testing.assert_allclose(outcomes.loc[outcomes.policy.isin(R.ALGOS), "gap_dqn_bps"], 0)
    np.testing.assert_allclose(outcomes.loc[outcomes.policy.isin(R.ALGOS), "gap_as_bps"], 2)
    runs[1].records.loc[runs[1].records.policy == "as", "pnl"] += 1
    runs[1].records.loc[runs[1].records.policy == "as", "risk_adjusted_pnl"] += 1
    runs[1].records.drop(columns=["pnl_bps", "risk_adjusted_pnl_bps"], inplace=True)
    with pytest.raises(ValueError, match="reference policy differs"):
        R.seed_outcomes(runs)


def test_pairing_requires_env_seed_not_just_episode(tmp_path):
    runs = [example_run(tmp_path, a) for a in R.ALGOS]
    runs[1].records.loc[runs[1].records.policy == runs[1].algo, "env_seed"] += 1
    with pytest.raises(ValueError, match="CRN violation"):
        R.seed_outcomes(runs)


def test_means_retain_bad_seeds_and_single_seed_has_no_interval():
    assert R.mean_interval([-100., 1., 2., 3., 4.])[0] == -18.
    assert np.isnan(R.mean_interval([1.])[1])
    assert R.mean_interval([1., 4., -2.]) == R.mean_interval([-2., 1., 4.])


def test_scope_discloses_training_pool_and_reused_historical_holdout(tmp_path):
    run = example_run(tmp_path, symbol="MSFT", setting=R.P.HISTORICAL_SETTING)
    scope = R.experiment_scope([run])
    assert "training dates only" in scope
    assert "reused holdout" in scope
    assert "Additional training seeds do not add independent historical dates" in scope
    assert R.experiment_scope([example_run(tmp_path)]) == ""


@pytest.mark.parametrize("has_actor", [False, True])
def test_training_row_covers_current_schema_including_actor_rates(has_actor):
    from lmm.experiments.train import METRICS_COLUMNS, _metrics_row
    from lmm.rl.loops import EpisodeResult
    result = EpisodeResult(env_seed=7)
    if has_actor:
        result.diagnostics = {"actor_learning_rate_clob": 0.0001,
                              "actor_learning_rate_auction": 0.00005}
    row = _metrics_row(0, result, .1, {"clob": 1, "auction": 1},
                       {"clob": 1, "auction": 1}, None, None, None, None,
                       None, None, 0.0)
    assert len(row) == len(METRICS_COLUMNS)
    saved = dict(zip(METRICS_COLUMNS, row))
    for phase, expected in [("clob", .0001), ("auction", .00005)]:
        value = saved[f"actor_learning_rate_{phase}"]
        assert float(value) == expected if has_actor else value == ""


def test_historical_macro_averages_tickers_inside_seed_blocks():
    data = pd.DataFrame({"market": ["CAT", "MSFT", "CAT", "MSFT"], "policy": ["dqn"]*4,
                         "seed": [7, 7, 42, 42], "objective_bps": [1., 3., 10., 30.]})
    macro = R.auction_groups(data)
    assert len(macro) == 2
    np.testing.assert_allclose(macro.objective_bps, [2., 20.])


def test_learning_aggregate_never_forward_fills_stopped_seeds():
    data = pd.DataFrame({"market": ["Synthetic"]*5, "policy": ["dqn"]*5,
                         "seed": [7, 7, 42, 42, 42], "episodes_completed": [0, 100, 0, 100, 200],
                         "objective_bps": [1., 2., 3., 4., 100.]})
    out = R.learning_summary(data)
    assert list(out.episodes_completed) == [0, 100]
    assert list(out["mean"]) == [2., 3.]


def test_credit_learning_requires_matched_paths_and_observed_budgets(tmp_path):
    dense = example_run(tmp_path, setting=R.SYNTHETIC+'__mechanism_economic_dense', materialize=True)
    sparse = example_run(tmp_path, setting=R.SYNTHETIC+'__mechanism_economic_sparse', materialize=True)
    metrics_path = sparse.run_dir/'metrics.csv'
    metrics = pd.read_csv(metrics_path)
    # Leave an internal missing evaluation, while retaining the selected one.
    metrics = metrics[metrics.episode != 199]
    metrics['eval_risk_adjusted_pnl_mean'] -= 2
    metrics.to_csv(metrics_path, index=False)
    data = R.credit_learning_rows([dense, sparse])
    assert 200 not in set(data.episodes_completed)
    np.testing.assert_allclose(data.loc[data.episodes_completed > 0, 'objective_bps'], 2.)
    meta = sparse.run_dir/'checkpoints/initial_validation.yaml'
    payload = yaml.safe_load(meta.read_text()); payload['validation_seeds'] = [1, 2, 4]
    meta.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match='validation CRN mismatch'):
        R.credit_learning_rows([dense, sparse])


def test_treatment_figure_includes_pure_auction_access(tmp_path, monkeypatch):
    data = pd.DataFrame([dict(contrast_key=key, algorithm=algo, mean_difference=1.)
                         for key, *_ in publication.TREATMENT_CONTRASTS for algo in R.ALGOS])
    figures = []
    monkeypatch.setattr(R, '_finish', lambda fig, *args: figures.append(fig))
    R.treatment_figure(data, tmp_path)
    visible = [a for a in figures[0].axes if a.get_visible()]
    assert len(visible) == len(publication.TREATMENT_CONTRASTS)
    assert any('Auction access on' in a.get_title() for a in visible)
    R.P.plt.close(figures[0])


@pytest.mark.parametrize("amended", [False, True])
def test_complete_curated_bundle_and_read_only_inputs(tmp_path, monkeypatch, amended):
    materialize_matrix(tmp_path)
    if amended:
        from lmm.experiments import mature_reporting as M
        protocol = {"disclosure": M.NOTE, "runs": {"synthetic/dqn__no_auction_seed123": {
            "selected_episode": 499, "validation_score": 12., "initial_validation_score": 13.,
            "improved_over_initial": False, "evaluated_checkpoint": "best_mature"}}}
        monkeypatch.setattr(M, "load_protocol", lambda runs: protocol)
        path = tmp_path / M.RELATIVE_PATH
        path.parent.mkdir()
        path.write_text(json.dumps(protocol))
    inputs = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    out = R.generate(tmp_path, [9501, 9502], complete=True, treatments=True)
    assert len(list((out / "figures").glob("*.pdf"))) == 5
    assert len(list((out / "figures").glob("*.png"))) == 5
    assert len(list((out / "tables").glob("*.tex"))) == 3
    assert len(pd.read_csv(out / "audit/economic_by_seed.csv")) == 6*6*2
    assert len(pd.read_csv(out / "audit/treatments_by_seed.csv")) == 5*4*2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["publication"] is False
    assert len(manifest["runs"]) == 88
    for p, digest in inputs.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert "DEVELOPMENT ONLY" in (out / "index.html").read_text()
    if amended:
        assert "Post-run reporting amendment" in (out / "index.html").read_text()
        assert "initial 13.000000" in (out / "audit/reporting_amendment.tex").read_text()
        assert len(pd.read_csv(out / "audit/checkpoint_selection.csv")) == 1
    assert "longtable" in (out / "tables/economic_performance.tex").read_text()
    assert r"95\%" in (out / "tables/economic_performance.tex").read_text()
    if shutil.which("pdflatex"):
        document = (r"\documentclass[11pt]{article}\usepackage[margin=1in]{geometry}"
                    r"\usepackage[T1]{fontenc}\usepackage{booktabs,longtable}\begin{document}" + "\n")
        for name in ("economic_performance", "auction_mechanism", "treatments"):
            document += "\\input{" + str(out / "tables" / f"{name}.tex") + "}\\clearpage\n"
        if amended:
            document += "\\input{" + str(out / "audit/reporting_amendment.tex") + "}\n"
        (tmp_path / "verify.tex").write_text(document + r"\end{document}")
        compiled = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "verify.tex"],
                                  cwd=tmp_path, capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stdout
        assert "Overfull" not in compiled.stdout


def test_publication_rejects_development_seeds_before_writing(tmp_path):
    with pytest.raises(ValueError, match="canonical seeds"):
        R.generate(tmp_path, [9501, 9502], publication_mode=True)
    assert not (tmp_path / "_publication").exists()


def test_incomplete_block_is_not_silently_omitted(tmp_path):
    for algo in R.ALGOS[:-1]:
        example_run(tmp_path, algo, 7, materialize=True)
    with pytest.raises(ValueError, match="unbalanced"):
        R.generate(tmp_path, [7])
    assert not (tmp_path / "_single_seed7").exists()
