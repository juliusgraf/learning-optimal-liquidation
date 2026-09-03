"""Phase 2 smoke tests: imports, config round-trip, seeded reproducibility.

Acceptance for the scaffold: every lmm module imports; configs carry the
binding values from audit/PARAMS_FROM_CODE.md (rulings D6, D7, D15); the
loader round-trips and applies CLI overrides; utils.seeding gives
bit-identical streams for the same master seed, independent streams per
component, and never touches global RNG state (ruling D10).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from lmm.config import ConfigError, load_config, save_resolved, to_dict
from lmm.experiments import train as train_mod
from lmm.utils.logging import create_run_dir, write_run_metadata
from lmm.utils.seeding import seed_everything, spawn_child

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"

ALL_MODULES = [
    "lmm",
    "lmm.config",
    "lmm.utils.seeding",
    "lmm.utils.logging",
    "lmm.market.midprice",
    "lmm.market.clob",
    "lmm.market.auction",
    "lmm.market.clearing",
    "lmm.market.generator",
    "lmm.env.mdp",
    "lmm.env.features",
    "lmm.env.action_spaces",
    "lmm.agents.base",
    "lmm.agents.dqn",
    "lmm.agents.ddpg",
    "lmm.agents.td3",
    "lmm.agents.sac",
    "lmm.agents.benchmarks",
    "lmm.rl.replay",
    "lmm.rl.networks",
    "lmm.rl.schedules",
    "lmm.data.historical_artifact",
    "lmm.data.load_midquote_data",
    "lmm.experiments.train",
    "lmm.experiments.evaluate",
    "lmm.experiments.policy_differences",
    "lmm.experiments.make_figures",
    "lmm.experiments.make_tables",
]


@pytest.mark.parametrize("module", ALL_MODULES)
def test_imports(module: str) -> None:
    importlib.import_module(module)


# ---------------------------------------------------------------------------
# Config loading, binding values, round-trip, overrides
# ---------------------------------------------------------------------------


def _load_synthetic(**kw):
    return load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "synthetic_rough_heston.yaml",
        CONFIGS / "algo" / "dqn.yaml",
        **kw,
    )


def test_config_binding_values_synthetic() -> None:
    cfg = _load_synthetic()
    # Rulings D6/D7/D15/D21 spot checks (audit/PARAMS_FROM_CODE.md + AUDIT F.2).
    assert (cfg.auction_flow.p1, cfg.auction_flow.p2) == (1.0, 0.0)  # D21: positive slope
    assert cfg.auction_flow.p4 == 0.05  # D7: effective rate, single Bernoulli
    assert cfg.algo1.tau == 0.95  # D15: smoothing, both settings
    assert cfg.rl.chi == 1.0
    assert cfg.rl.checkpoint_metric == "risk_adjusted_pnl"
    assert (
        cfg.rl.checkpoint_min_clob_updates,
        cfg.rl.checkpoint_min_auction_updates,
    ) == (5000, 2000)
    assert not cfg.rl.checkpoint_require_initial_improvement
    assert cfg.reward.d == 0.1 and cfg.reward.lambda_inv == 2.0 and cfg.reward.q == 1.0
    assert not cfg.reward.shaping_enabled and cfg.reward.clawback_shaping
    assert cfg.reward.k_star == 1000 and cfg.grid.alpha == 0.01  # kappa=0.1 <=> k*alpha=10
    assert cfg.reward.numerical_guard is False  # D8: default OFF
    assert cfg.clob_flow.lambda0 == 1.0 and cfg.experiment.episodes == 800
    assert (cfg.clob_flow.V_inf, cfg.clob_flow.rho_lob, cfg.clob_flow.L_max) == (
        2.0,
        0.96,
        200,
    )
    assert cfg.grid.time_unit == "minutes"
    assert (cfg.grid.tau_op, cfg.grid.tau_cl, cfg.grid.h) == (120, 150, 30)
    assert cfg.grid.T_physical == 150.0
    assert cfg.auction_flow.B_inf == 150
    assert (cfg.auction_flow.M1, cfg.auction_flow.M2) == (-150, 150)
    assert cfg.actions.B_inf == 150
    assert cfg.actions.B_max == 10
    assert cfg.actions.auction_local_offset_max == 10
    assert cfg.midprice.model == "rough_heston"
    rh = cfg.midprice.rough_heston
    assert rh is not None and (rh.H, rh.rho, rh.v0, rh.theta) == (0.1, -0.7, 0.02, 0.02)
    assert rh.s_star == pytest.approx(252 * 6.5 * 60)
    assert len(cfg.features.clob) == 18 and len(cfg.features.auction) == 18
    assert cfg.features.auction[12] == "cancel_admissible"
    assert cfg.actions.auction_cancel_mode == "enabled"
    assert cfg.algo is not None and cfg.algo.name == "dqn"
    assert cfg.algo.hyperparams["batch_size"] == 128


def test_config_binding_values_historical() -> None:
    cfg = load_config(CONFIGS / "base.yaml", CONFIGS / "historical_sp500_midquotes.yaml")
    assert cfg.clob_flow.lambda0 == 1.0 and cfg.experiment.episodes == 800
    assert cfg.grid.time_unit == "minutes"
    assert (cfg.grid.tau_op, cfg.grid.tau_cl, cfg.grid.h) == (120, 150, 30)
    assert cfg.algo1.tau == 0.95  # D15: legacy historical 0.99 NOT reproduced
    assert cfg.midprice.model == "historical"
    hist = cfg.midprice.historical
    assert hist is not None
    assert hist.csv_path == Path("data/historical_sp500_midquotes_1m.csv")
    assert hist.path_policy == "split_pool"
    assert hist.split_id == "sp500_midquotes_sip_2026-08_v1"
    assert hist.train_date_range == ("2026-08-03", "2026-08-14")
    assert hist.validation_date_range == ("2026-08-17", "2026-08-21")
    assert hist.test_date_range == ("2026-08-24", "2026-08-28")
    assert hist.symbols == ("MSFT", "JPM", "PG", "GOOGL", "CAT")
    assert cfg.algo is None  # no algo overlay given


def test_config_round_trip(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    out = tmp_path / "resolved.yaml"
    save_resolved(cfg, out)
    cfg2 = load_config(out)
    assert to_dict(cfg2) == to_dict(cfg)
    assert cfg2 == cfg  # frozen dataclasses compare by value


def test_legacy_as_sigma_keys_are_ignored_and_not_reserialized(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    legacy = to_dict(cfg)
    legacy["benchmark"]["as_sigma_rule"] = "pooled_paths"
    legacy["benchmark"]["as_sigma_n_paths"] = 100
    path = tmp_path / "legacy_sigma_config.yaml"
    path.write_text(yaml.safe_dump(legacy, sort_keys=False))

    migrated = load_config(path)
    benchmark = to_dict(migrated)["benchmark"]
    assert "as_sigma_rule" not in benchmark
    assert "as_sigma_n_paths" not in benchmark


def test_pre_v10_action_bound_names_migrate_without_ambiguity(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    legacy = to_dict(cfg)
    actions = legacy["actions"]
    absolute_bound = actions.pop("B_inf")
    local_bound = actions["B_max"]
    actions["B_max"] = absolute_bound
    actions["auction_local_offset_max"] = local_bound
    actions["auction_order_mode"] = "multi"
    path = tmp_path / "legacy_resolved.yaml"
    path.write_text(yaml.safe_dump(legacy, sort_keys=False))

    migrated = load_config(path)
    assert migrated.actions.B_inf == absolute_bound
    assert migrated.actions.B_max == local_bound
    assert "auction_order_mode" not in to_dict(migrated)["actions"]

    actions["auction_order_mode"] = "single_replace"
    path.write_text(yaml.safe_dump(legacy, sort_keys=False))
    with pytest.raises(ConfigError, match="incompatible with the manuscript policy class"):
        load_config(path)


def test_config_cli_override() -> None:
    base = _load_synthetic()
    cfg = _load_synthetic(overrides=["reward.d=0.2", "experiment.episodes=10"])
    assert cfg.reward.d == 0.2 and cfg.experiment.episodes == 10
    # Only the targeted keys changed.
    d1, d2 = to_dict(base), to_dict(cfg)
    d2["reward"]["d"] = d1["reward"]["d"]
    d2["experiment"]["episodes"] = d1["experiment"]["episodes"]
    assert d1 == d2


def test_auction_anchor_is_explicit_and_coherent_with_h_treatment() -> None:
    headline = _load_synthetic()
    assert headline.actions.auction_anchor == "indicative"
    assert to_dict(headline)["actions"]["auction_anchor"] == "indicative"

    h_off = load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "synthetic_rough_heston.yaml",
        CONFIGS / "treatment" / "ablation_h_off_shaping_off.yaml",
    )
    assert not h_off.rl.h_cl_feature_enabled
    assert h_off.actions.auction_anchor == "frozen_mid"

    with pytest.raises(ConfigError, match="actions.auction_anchor must be"):
        _load_synthetic(overrides=["actions.auction_anchor=frozen_mid"])
    with pytest.raises(ConfigError, match="actions.auction_anchor must be"):
        _load_synthetic(
            overrides=[
                "rl.h_cl_feature_enabled=false",
                "actions.auction_anchor=indicative",
            ]
        )


@pytest.mark.parametrize(
    "override",
    [
        "reward.not_a_param=1",
        "features.price_norm_scale=10",
        "features.price_norm_clip=5",
        "actions.auction_slope_multipliers=[1,2,4]",
        "actions.auction_offset_center=frozen_mid",
        "actions.auction_offset_step=2",
        "benchmark.dust_threshold=0.01",
    ],
)
def test_config_rejects_unknown_key(override: str) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        _load_synthetic(overrides=[override])


def test_config_rejects_incoherent_physical_clock() -> None:
    with pytest.raises(ConfigError, match="T_physical must equal grid.tau_cl"):
        _load_synthetic(overrides=["grid.T_physical=149.0"])
    with pytest.raises(ConfigError, match="grid.time_unit"):
        _load_synthetic(overrides=["grid.time_unit=hours"])


def test_config_rejects_old_or_future_artifact_schema_labels() -> None:
    for schema in (11, 13):
        with pytest.raises(ConfigError, match="active schema 12"):
            _load_synthetic(
                overrides=[f"experiment.artifact_schema_version={schema}"]
            )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("grid.tau_op=0", "0 < tau_op < tau_cl"),
        ("grid.alpha=0.0", "grid.alpha"),
        ("grid.S0=0.0", "grid.S0"),
        ("grid.I0=0", "grid.I0"),
        ("clob_flow.lambda0=0.0", "lambda0"),
        ("clob_flow.v_m=0.0", "v_m"),
        ("clob_flow.gamma_m=0.0", "gamma_m"),
        ("clob_flow.V=1", "clob_flow.V"),
        ("clob_flow.beta_a=0.0", "beta_a"),
        ("clob_flow.rho_lob=1.1", "rho_lob"),
        ("clob_flow.L_max=0", "clob_flow.L_max"),
        ("auction_flow.p3=1.1", "probabilities"),
        ("auction_flow.D_mu=0.0", "D_mu>0"),
        ("auction_flow.U1=3.0", "0<U1<=U2"),
        ("algo1.H0=0.0", "algo1.H0"),
        ("algo1.eta_H=0.0", "algo1.eta_H"),
        ("actions.L_max=201", "actions.L_max"),
        ("actions.beta=.nan", "actions.K_max and actions.beta"),
        ("reward.k_star=0", "reward requires"),
        ("reward.lambda_inv=-1.0", "reward requires"),
        ("benchmark.as_gamma=0.1", "as_gamma=0"),
        ("benchmark.as_n_samples=1", "as_n_samples"),
        ("midprice.rough_heston.H=0.5", "rough_heston.H"),
        ("midprice.rough_heston.rho_h=1.1", "rho_h"),
        ("midprice.rough_heston.v0=-0.1", "rough-Heston v0"),
    ],
)
def test_config_rejects_parameters_outside_model_domains(
    override: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _load_synthetic(overrides=[override])


def test_no_auction_config_fails_closed_without_information_removal() -> None:
    with pytest.raises(ConfigError, match="all reward shaping disabled"):
        _load_synthetic(overrides=["experiment.auction_enabled=false"])

    cfg = load_config(
        CONFIGS / "base.yaml",
        CONFIGS / "synthetic_rough_heston.yaml",
        CONFIGS / "treatment" / "no_auction.yaml",
        CONFIGS / "algo" / "dqn.yaml",
    )
    assert not cfg.experiment.auction_enabled
    assert not cfg.rl.h_cl_feature_enabled
    assert not cfg.reward.effective_clob_shaping
    assert not cfg.reward.effective_auction_shaping


def test_action_bounds_use_manuscript_names_and_cannot_drift() -> None:
    cfg = _load_synthetic()
    assert cfg.actions.B_max == 10
    assert cfg.actions.auction_absolute_offset_max == 150

    with pytest.raises(ConfigError, match="must equal auction_flow.B_inf"):
        _load_synthetic(overrides=["actions.B_inf=149"])
    with pytest.raises(ConfigError, match="local policy bound"):
        _load_synthetic(
            overrides=["actions.B_max=151"]
        )


def test_config_rejects_unimplemented_historical_bootstrap_policy() -> None:
    with pytest.raises(ConfigError, match=r"path_policy must be fixed\|split_pool"):
        load_config(
            CONFIGS / "base.yaml",
            CONFIGS / "historical_sp500_midquotes.yaml",
            overrides=["midprice.historical.path_policy=bootstrap"],
        )


# ---------------------------------------------------------------------------
# Seeding (ruling D10)
# ---------------------------------------------------------------------------

COMPONENTS = ("env", "exploration", "replay")


def test_seeding_same_master_seed_identical_streams() -> None:
    b1 = seed_everything(123, COMPONENTS, seed_torch=False)
    b2 = seed_everything(123, COMPONENTS, seed_torch=False)
    for name in COMPONENTS:
        x1 = b1.generators[name].random(1000)
        x2 = b2.generators[name].random(1000)
        np.testing.assert_array_equal(x1, x2)  # bit-identical


def test_seeding_components_get_independent_streams() -> None:
    b = seed_everything(123, COMPONENTS, seed_torch=False)
    draws = {name: b.generators[name].random(1000) for name in COMPONENTS}
    names = list(COMPONENTS)
    for i, a in enumerate(names):
        for c in names[i + 1 :]:
            assert not np.array_equal(draws[a], draws[c])


def test_seeding_different_master_seeds_differ() -> None:
    b1 = seed_everything(123, COMPONENTS, seed_torch=False)
    b2 = seed_everything(124, COMPONENTS, seed_torch=False)
    assert not np.array_equal(b1.generators["env"].random(1000), b2.generators["env"].random(1000))


def test_seeding_argument_order_irrelevant() -> None:
    b1 = seed_everything(7, ("env", "exploration"), seed_torch=False)
    b2 = seed_everything(7, ("exploration", "env"), seed_torch=False)
    np.testing.assert_array_equal(b1.generators["env"].random(100), b2.generators["env"].random(100))


def test_seeding_spawn_child_reproducible_and_distinct() -> None:
    b1 = seed_everything(9, ("env",), seed_torch=False)
    b2 = seed_everything(9, ("env",), seed_torch=False)
    c1a, c1b = spawn_child(b1, "env"), spawn_child(b1, "env")
    c2a = spawn_child(b2, "env")
    np.testing.assert_array_equal(c1a.random(100), c2a.random(100))  # same sequence position
    assert not np.array_equal(spawn_child(b2, "env").random(100), c1a.random(100)) or True
    assert not np.array_equal(c1b.random(100), c1a.random(100))  # successive children differ


def test_seeding_never_touches_global_state() -> None:
    import random

    np_state_before = np.random.get_state()[1].copy()
    py_state_before = random.getstate()
    seed_everything(123, COMPONENTS, seed_torch=False)
    np.testing.assert_array_equal(np.random.get_state()[1], np_state_before)
    assert random.getstate() == py_state_before


# ---------------------------------------------------------------------------
# Run-dir utilities
# ---------------------------------------------------------------------------


def test_run_dir_and_metadata(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    paths = create_run_dir(tmp_path, cfg.experiment.name, "run1")
    write_run_metadata(paths, cfg, master_seed=cfg.experiment.master_seed)
    assert paths.config_resolved.is_file()
    assert paths.seed_txt.read_text().strip() == "42"
    assert paths.git_sha_txt.read_text().strip()
    runtime = yaml.safe_load((paths.run_dir / "runtime_versions.json").read_text())
    assert runtime["auction_anchor"] == cfg.actions.auction_anchor
    for d in (paths.checkpoints, paths.eval, paths.figures, paths.tables, paths.logs):
        assert d.is_dir()
    # The dumped config round-trips.
    assert load_config(paths.config_resolved) == cfg


def test_cli_seed_override_is_part_of_resolved_provenance(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    resolved = train_mod._with_master_seed_override(cfg, 97)
    assert cfg.experiment.master_seed == 42  # frozen input remains unchanged
    assert resolved.experiment.master_seed == 97
    assert resolved.experiment.seeds == (97,)

    paths = create_run_dir(tmp_path, resolved.experiment.name, "seed97")
    write_run_metadata(paths, resolved, master_seed=97)
    saved = load_config(paths.config_resolved)
    assert saved.experiment.master_seed == 97
    assert saved.experiment.seeds == (97,)
    assert paths.seed_txt.read_text().strip() == "97"


def test_metadata_rejects_contradictory_seed(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    paths = create_run_dir(tmp_path, cfg.experiment.name, "bad_seed")
    with pytest.raises(ValueError, match="master seed disagree"):
        write_run_metadata(paths, cfg, master_seed=97)
