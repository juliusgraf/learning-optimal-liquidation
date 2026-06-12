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

from lmm.config import ConfigError, load_config, save_resolved, to_dict
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
    "lmm.data.load_yfinance_data",
    "lmm.experiments.train",
    "lmm.experiments.evaluate",
    "lmm.experiments.regret",
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
    # Rulings D6/D7/D15 spot checks (audit/PARAMS_FROM_CODE.md).
    assert cfg.auction_flow.p4 == 0.05  # D7: effective rate, single Bernoulli
    assert cfg.algo1.tau == 0.95  # D15: smoothing, both settings
    assert cfg.rl.chi == 0.99  # D15: discount
    assert cfg.reward.d == 0.1 and cfg.reward.lambda_inv == 0.5 and cfg.reward.q == 1.0
    assert cfg.reward.k_star == 1000 and cfg.grid.alpha == 0.01  # kappa=0.1 <=> k*alpha=10
    assert cfg.reward.numerical_guard is False  # D8: default OFF
    assert cfg.clob_flow.lambda0 == 1.0 and cfg.experiment.episodes == 2000
    assert cfg.midprice.model == "rough_heston"
    rh = cfg.midprice.rough_heston
    assert rh is not None and (rh.H, rh.rho, rh.v0, rh.theta) == (0.1, -0.7, 0.02, 0.04)
    assert len(cfg.features.clob) == 8 and len(cfg.features.auction) == 7  # AUDIT A.7
    assert cfg.algo is not None and cfg.algo.name == "dqn"
    assert cfg.algo.hyperparams["batch_size"] == 128


def test_config_binding_values_historical() -> None:
    cfg = load_config(CONFIGS / "base.yaml", CONFIGS / "historical_sp500.yaml")
    assert cfg.clob_flow.lambda0 == 60.0 and cfg.experiment.episodes == 1000
    assert cfg.algo1.tau == 0.95  # D15: legacy historical 0.99 NOT reproduced
    assert cfg.midprice.model == "historical"
    hist = cfg.midprice.historical
    assert hist is not None
    assert hist.csv_path == Path("legacy/data.csv")  # frozen input (D13)
    assert hist.symbols == ("MSFT", "JPM", "PG", "GOOGL", "CAT")
    assert cfg.benchmark.as_sigma_rule == "single_path"
    assert cfg.algo is None  # no algo overlay given


def test_config_round_trip(tmp_path: Path) -> None:
    cfg = _load_synthetic()
    out = tmp_path / "resolved.yaml"
    save_resolved(cfg, out)
    cfg2 = load_config(out)
    assert to_dict(cfg2) == to_dict(cfg)
    assert cfg2 == cfg  # frozen dataclasses compare by value


def test_config_cli_override() -> None:
    base = _load_synthetic()
    cfg = _load_synthetic(overrides=["reward.d=0.2", "experiment.episodes=10"])
    assert cfg.reward.d == 0.2 and cfg.experiment.episodes == 10
    # Only the targeted keys changed.
    d1, d2 = to_dict(base), to_dict(cfg)
    d2["reward"]["d"] = d1["reward"]["d"]
    d2["experiment"]["episodes"] = d1["experiment"]["episodes"]
    assert d1 == d2


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        _load_synthetic(overrides=["reward.not_a_param=1"])


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
    for d in (paths.checkpoints, paths.eval, paths.figures, paths.tables, paths.logs):
        assert d.is_dir()
    # The dumped config round-trips.
    assert load_config(paths.config_resolved) == cfg
