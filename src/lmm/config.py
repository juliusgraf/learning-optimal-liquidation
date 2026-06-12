"""Configuration dataclasses and YAML loader (Phase 2, fully functional).

Every experiment parameter lives in ``configs/*.yaml`` (ruling D6: values are
sourced from the legacy instantiation sites, see ``audit/PARAMS_FROM_CODE.md``).
Nothing in ``src/`` hard-codes an experiment parameter; library code receives
the dataclasses defined here.

Loading model: ``load_config(base, *overlays, overrides=...)`` deep-merges the
YAML mappings left to right (later files win), then applies dotted CLI
overrides (``reward.d=0.2``), then builds the frozen dataclass tree with a
strict unknown-key check and scalar type coercion. ``save_resolved`` /
``to_dict`` round-trip exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml

__all__ = [
    "ConfigError",
    "ExperimentMeta",
    "GridParams",
    "ClobFlowParams",
    "AuctionFlowParams",
    "RoughHestonParams",
    "HistoricalParams",
    "MidPriceConfig",
    "Algo1Params",
    "RewardParams",
    "RLParams",
    "ActionGridParams",
    "FeatureParams",
    "BenchmarkParams",
    "AlgoConfig",
    "ExperimentConfig",
    "load_config",
    "build_hyperparams",
    "to_dict",
    "save_resolved",
    "add_config_cli",
    "config_from_args",
]


class ConfigError(ValueError):
    """Raised on unknown keys, missing required keys, or bad value types."""


# ---------------------------------------------------------------------------
# Dataclasses mirroring the YAML tree. Fields without defaults are REQUIRED
# in the merged YAML, so configs/ stays the single source of truth (D6).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentMeta:
    """Run identity and bookkeeping."""

    name: str
    setting: str  # "synthetic_rough_heston" | "historical_sp500"
    episodes: int  # E; legacy 2000 (synthetic) / 1000 (historical)
    master_seed: int  # single master seed; ruling D10
    results_root: Path  # gitignored output root


@dataclass(frozen=True)
class GridParams:
    """Time grid and price grid (sec:marketmodel; CLAUDE.md grid convention).

    Grid 0 = t_0 < ... < t_n < tau_op = t_{n+1} < ... < t_m < tau_cl = t_{m+1};
    n = tau_op - 1, m = tau_cl - 1; episode horizon m + 2 steps.
    """

    tau_op: int  # tau_op; main.py:1417 / data.py:1386 => 120
    tau_cl: int  # tau_cl; => 150
    T_physical: float  # physical session length; dt = T/tau_cl = 1.0
    alpha: float  # tick size alpha; => 0.01
    S0: float  # initial mid price S^mid_0; main.py:263 => 100.0
    I0: int  # initial inventory I_0; ctor `I` => 100
    I_max: int  # legacy normalization constant (== I0); clipping removed per D8


@dataclass(frozen=True)
class ClobFlowParams:
    """CLOB-phase exogenous flow (Algorithm 2 `alg:generative_model`, Sec. 2.1)."""

    lambda0: float  # Poisson intensity lambda_0 per side; 1.0 synth / 60.0 hist
    v_m: float  # Pareto scale v_m; main.py:1418 => 2.0
    gamma_m: float  # Pareto shape gamma_m; => 2.5
    V_max: int  # max order volume V (Pareto cap, volume grid max); => 30
    V_inf: float  # top-of-book scale V_inf (Beta multiplier); => 15.0
    beta_a: float  # Beta shape a; => 2.0
    beta_b: float  # Beta shape b; => 5.0
    depth_decay: float  # geometric depth decay rho; => 0.5
    Lc: int  # book levels per side; => 12


@dataclass(frozen=True)
class AuctionFlowParams:
    """Auction-phase exogenous flow (Algorithm 2 lines 12-17; rulings D6-D7)."""

    p1: float  # new exogenous MM arrival prob; main.py:592 => 0.3
    p2: float  # exogenous MM cancellation prob; main.py:597 => 0.2
    p3: float  # new taker arrival prob per side (independent); => 0.3
    # Taker cancellation prob per side. Ruling D7: single Bernoulli(0.05);
    # legacy realized it as Bernoulli(0.1) gated by a fair coin (same law).
    p4: float
    K_min: float  # exogenous MM slope law U_1; K ~ U(K_min, K_max); => 0.1
    K_max: float  # U_2; => 2.0
    price_band_ticks: int  # quote band: S ~ S_mid(frozen) + alpha*U{-band..band}; => 10
    La: int  # cap on # exogenous auction MMs (extra, AUDIT N8); => 12
    L_max: int  # cap on N^zeta / nu-array size (paper script-N); => 100


@dataclass(frozen=True)
class RoughHestonParams:
    """Rough Heston scheme of Richard et al. (sub:rough; ruling D14: keep exactly)."""

    H: float  # Hurst H; main.py:1421 => 0.1
    rho: float  # correlation rho; => -0.7
    v0: float  # V_0; => 0.02
    theta: float  # long-run variance theta; => 0.04
    kappa: float  # mean reversion (paper's lambda); => 0.3
    xi: float  # vol-of-vol (paper's nu); => 0.3
    seconds_per_year: float  # physical-time scaling; main.py:25-33 => 252*6.5*3600


@dataclass(frozen=True)
class HistoricalParams:
    """Historical mid-path replay (sub:historical; ruling D13)."""

    csv_path: Path  # frozen experimental input (legacy/data.csv)
    symbols: tuple[str, ...]  # data.py:1651 => MSFT, JPM, PG, GOOGL, CAT
    normalize_first: float  # --normalize first=100 default (D13)
    n_rows: int  # rows consumed per path; data.py:1654 => 120 (= tau_op)


@dataclass(frozen=True)
class MidPriceConfig:
    """Mid-price model selector; exactly one branch is used per setting."""

    model: str  # "rough_heston" | "historical"
    rough_heston: Optional[RoughHestonParams] = None
    historical: Optional[HistoricalParams] = None


@dataclass(frozen=True)
class Algo1Params:
    """Algorithm 1, hypothetical clearing price (alg:hyp_clearing_price; D2, D15)."""

    tau: float  # smoothing tau; ruling D15 => 0.95 in BOTH settings
    H0_from_mid: bool  # H_0 = initial mid (= 100); main.py:264
    H0: Optional[float] = None  # explicit H_0; required iff H0_from_mid is false


@dataclass(frozen=True)
class RewardParams:
    """Three-regime reward constants (sec:MDP; rulings D4, D5, D8)."""

    k_star: int  # k* in f_c; legacy kappa=0.1 <=> k*alpha=10 <=> k*=1000
    lambda_inv: float  # terminal inventory penalty lambda; main.py:1417 => 0.5
    q: float  # wrong-side penalty q in f_a; => 1.0
    d: float  # cancellation cost unit d (cost d_t*c_t, D4); => 0.1
    numerical_guard: bool  # D8: optional far-out float guard, default OFF
    # |I_tau_cl| threshold when the guard is on — far outside the economic
    # range (|I| <= I0 + auction exposure ~ O(10^3)); binding is logged
    # loudly and asserted never to happen on seeded standard runs (D8).
    numerical_guard_bound: float = 1e9


@dataclass(frozen=True)
class RLParams:
    """Objective (P) constants shared by all algorithms."""

    chi: float  # discount chi; GAMMA main.py:1440 / data.py:1355 => 0.99 (D15)


@dataclass(frozen=True)
class ActionGridParams:
    """Discrete action grids (AUDIT A.8; grids are config choices, D6).

    CLOB: {(0,0)} u {1..volume_max} x {delta_min..delta_max} (361 actions).
    Auction: K in {0} u linspace(K_grid_min, K_grid_max, K_grid_n), offset in
    {-offset_max..offset_max}, c in {0,1} (550 actions). K^a = 0 == abstain.
    """

    clob_volume_max: int  # main.py:1424 => 30 (= V_max)
    clob_delta_min: int  # => 1 (plus the (0,0) no-op)
    clob_delta_max: int  # => 12; N6: defined explicitly, no silent clamping
    auction_K_grid_min: float  # => 1.0
    auction_K_grid_max: float  # K_MAX = 10*I0/V_max; main.py:1430 => 33.3333...
    auction_K_grid_n: int  # => 10 (plus K = 0)
    auction_offset_max: int  # S^a = S_mid + off*alpha, off in {-12..12} => 12


@dataclass(frozen=True)
class FeatureParams:
    """Legacy-pruned RL feature lists (ruling D11; AUDIT A.7 — exact defaults).

    Names use the PAPER sign convention (D3): N_buy = paper N^+ (buying MOs).
    """

    clob: tuple[str, ...]  # 8 dims; feat_clob main.py:1443-1450
    auction: tuple[str, ...]  # 7 dims; feat_auction main.py:1452-1457


@dataclass(frozen=True)
class BenchmarkParams:
    """AS / TWAP benchmarks (sec:benchmark; AUDIT A.9; ruling D16)."""

    z: float  # auction heuristic slope multiplier z; main.py:1048 => 10.0
    dust_threshold: float  # q < dust => no auction order; => 1e-2
    as_n_samples: int  # K-hat regression samples; main.py:1935 => 10000
    as_gamma: float  # AS risk aversion gamma; => 0.0
    as_sigma_rule: str  # "pooled_paths" (synthetic) | "single_path" (historical)
    as_sigma_n_paths: int  # simulated mid paths pooled for sigma (Phase 4)
    twap_delta_mode: str  # "min" => delta = min of the grid = 1


@dataclass(frozen=True)
class AlgoConfig:
    """RL algorithm selection + hyperparameters (configs/algo/*.yaml; D9)."""

    name: str  # "dqn" | "ddpg" | "td3" | "sac"
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level resolved configuration (mirror of the merged YAML tree)."""

    experiment: ExperimentMeta
    grid: GridParams
    clob_flow: ClobFlowParams
    auction_flow: AuctionFlowParams
    midprice: MidPriceConfig
    algo1: Algo1Params
    reward: RewardParams
    rl: RLParams
    actions: ActionGridParams
    features: FeatureParams
    benchmark: BenchmarkParams
    algo: Optional[AlgoConfig] = None  # absent for benchmark-only runs


# ---------------------------------------------------------------------------
# Generic dict -> dataclass builder with strict keys and scalar coercion
# ---------------------------------------------------------------------------


def _coerce(tp: Any, value: Any, path: str) -> Any:
    """Coerce a YAML value to the annotated type; raise ConfigError on mismatch."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is Union:  # Optional[X] and unions
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        raise ConfigError(f"{path}: unsupported union type {tp!r}")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping for {tp.__name__}, got {type(value).__name__}")
        return _build_dataclass(tp, value, path)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected sequence, got {type(value).__name__}")
        elem = args[0] if args else Any
        items = [_coerce(elem, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected mapping, got {type(value).__name__}")
        return dict(value)
    if tp is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{path}: expected path string, got {type(value).__name__}")
        return Path(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected float, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {value!r}")
        return value
    if tp is Any:
        return value
    raise ConfigError(f"{path}: unsupported annotation {tp!r}")


def _build_dataclass(dc_type: type, data: dict[str, Any], path: str = "") -> Any:
    hints = typing.get_type_hints(dc_type)
    field_names = {f.name for f in dataclasses.fields(dc_type)}
    unknown = sorted(set(data) - field_names)
    if unknown:
        raise ConfigError(f"{path or dc_type.__name__}: unknown key(s) {unknown}")
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(dc_type):
        key_path = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], key_path)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{key_path}: required key missing from config")
    return dc_type(**kwargs)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; overlay wins; non-dict values are replaced."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_override(tree: dict[str, Any], spec: str) -> None:
    """Apply one dotted override ``a.b.c=value`` (value parsed as YAML) in place."""
    if "=" not in spec:
        raise ConfigError(f"override {spec!r}: expected 'dotted.key=value'")
    dotted, raw = spec.split("=", 1)
    keys = dotted.strip().split(".")
    if not all(keys):
        raise ConfigError(f"override {spec!r}: empty key component")
    node = tree
    for k in keys[:-1]:
        nxt = node.setdefault(k, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"override {spec!r}: {k!r} is not a mapping")
        node = nxt
    node[keys[-1]] = yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    *yaml_paths: str | Path,
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Load and merge YAML files left-to-right, apply overrides, build the tree.

    Args:
        yaml_paths: e.g. ``configs/base.yaml``, ``configs/synthetic_rough_heston.yaml``,
            ``configs/algo/dqn.yaml``. Later files override earlier ones.
        overrides: dotted CLI overrides, e.g. ``("reward.d=0.2", "experiment.episodes=10")``.
    """
    if not yaml_paths:
        raise ConfigError("load_config needs at least one YAML path")
    merged: dict[str, Any] = {}
    for p in yaml_paths:
        path = Path(p)
        loaded = yaml.safe_load(path.read_text())
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        merged = _deep_merge(merged, loaded)
    for spec in overrides:
        _apply_override(merged, spec)
    return _build_dataclass(ExperimentConfig, merged)


def build_hyperparams(dc_type: type, mapping: dict[str, Any]) -> Any:
    """Build an algorithm hyperparameter dataclass from ``algo.hyperparams``
    with the same strict unknown-key check and scalar coercion as the YAML
    tree (so a typo in configs/algo/*.yaml fails loudly, not silently)."""
    if not isinstance(mapping, dict):
        raise ConfigError(
            f"algo.hyperparams: expected mapping, got {type(mapping).__name__}"
        )
    return _build_dataclass(dc_type, mapping, f"algo.hyperparams ({dc_type.__name__})")


def to_dict(cfg: Any) -> dict[str, Any]:
    """Dataclass tree -> plain YAML-serializable dict (Path -> str, tuple -> list)."""

    def conv(v: Any) -> Any:
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v

    return conv(cfg)


def save_resolved(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write the fully resolved config (round-trips via ``load_config``)."""
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))


def add_config_cli(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--config`` / ``--override`` flags to a parser."""
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="YAML",
        help="config file; repeat to overlay (base first, later files win)",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. -o reward.d=0.2 (repeatable)",
    )


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build the resolved config from parsed ``add_config_cli`` arguments."""
    return load_config(*args.config, overrides=args.override)
