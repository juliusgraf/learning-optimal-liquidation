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
from datetime import date
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
    episodes: int  # E; active settings use one matched budget
    master_seed: int  # single master seed; ruling D10
    results_root: Path  # gitignored output root
    artifact_schema_version: int = 2
    seeds: tuple[int, ...] = (42,)
    ablation_label: str = "H_on__shaping_on__auction_on"
    auction_enabled: bool = True


@dataclass(frozen=True)
class GridParams:
    """Time and price grid, including the explicit auction horizon ``h``."""

    tau_op: int  # tau_op; main.py:1417 / data.py:1386 => 120
    tau_cl: int  # tau_cl; => 150
    h: int  # number of auction action intervals; revised baseline => 30
    time_unit: str  # active synthetic and historical experiments both use minutes
    T_physical: float  # session length in time_unit; dt = T/tau_cl = 1 minute
    alpha: float  # tick size alpha; => 0.01
    S0: float  # initial mid price S^mid_0; main.py:263 => 100.0
    I0: int  # initial inventory I_0; ctor `I` => 100
    I_max: int  # legacy normalization constant (== I0); clipping removed per D8

    @property
    def physical_time_per_grid_unit(self) -> float:
        """Physical duration represented by one integer simulator interval."""
        return self.T_physical / self.tau_cl


@dataclass(frozen=True)
class ClobFlowParams:
    """CLOB exogenous flow; ``V`` is distinct from strategic ``V_max``."""

    lambda0: float  # Poisson intensity lambda_0 per side and per grid time unit
    v_m: float  # Pareto scale v_m; main.py:1418 => 2.0
    gamma_m: float  # Pareto shape gamma_m; => 2.5
    V: int  # exogenous market-order Pareto cap; => 30
    V_inf: float  # top-of-book scale V_inf (Beta multiplier); shared => 2.0
    beta_a: float  # Beta shape a; => 2.0
    beta_b: float  # Beta shape b; => 5.0
    rho_lob: float  # geometric CLOB depth persistence; shared => 0.96
    L_max: int  # maximum exogenous CLOB depth; shared => 200

    @property
    def V_max(self) -> int:
        """Compatibility alias used by the existing matching engine."""
        return self.V

    @property
    def depth_decay(self) -> float:
        return self.rho_lob

    @property
    def Lc(self) -> int:
        return self.L_max


@dataclass(frozen=True)
class AuctionFlowParams:
    """Auction exogenous proposal parameters.

    ``U1``/``U2`` are deliberately distinct from the strategic slope-grid
    index bound ``ActionGridParams.K_max``.
    """

    p1: float  # configured exogenous-MM arrival probability
    p2: float  # configured exogenous-MM cancellation probability
    p3: float  # new taker arrival prob per side (independent); => 0.3
    # Taker cancellation prob per side. Ruling D7: single Bernoulli(0.05);
    # legacy realized it as Bernoulli(0.1) gated by a fair coin (same law).
    p4: float
    D_mu: float  # numerical floor on total active exogenous slope; => 0.1
    U1: float  # exogenous schedule slope lower bound; => 0.1
    U2: float  # exogenous schedule slope upper bound; => 2.0
    M1: int  # exogenous schedule price-offset lower bound in ticks; => -10
    M2: int  # exogenous schedule price-offset upper bound in ticks; => 10

    @property
    def K_min(self) -> float:
        return self.U1

    @property
    def K_max(self) -> float:
        return self.U2

    @property
    def price_band_ticks(self) -> int:
        """Compatibility view for the current symmetric-band generator."""
        return max(abs(self.M1), abs(self.M2))


@dataclass(frozen=True)
class RoughHestonParams:
    """Rough-Heston parameters with rho disambiguated from CLOB depth."""

    H: float  # Hurst H; main.py:1421 => 0.1
    rho_h: float  # price/variance Brownian correlation; => -0.7
    v0: float  # V_0; => 0.02
    theta: float  # revised variance-drift level; => 0.02
    varsigma: float  # variance mean-reversion coefficient; => 0.3
    nu: float  # volatility of volatility; => 0.3
    s_star: float  # trading simulator-clock units per year; minute baseline => 98280

    @property
    def rho(self) -> float:
        return self.rho_h

    @property
    def kappa(self) -> float:
        return self.varsigma

    @property
    def xi(self) -> float:
        return self.nu

    @property
    def time_units_per_year(self) -> float:
        """Number of configured simulator clock units in one trading year."""
        return self.s_star


@dataclass(frozen=True)
class HistoricalParams:
    """Historical mid-path replay (sub:historical; ruling D13)."""

    csv_path: Path  # frozen multi-session experimental input
    symbols: tuple[str, ...]  # data.py:1651 => MSFT, JPM, PG, GOOGL, CAT
    # The frozen artifact remains in provider price units.  This is a modeling
    # coordinate transform applied independently after a session is selected;
    # it must not be confused with source-data preprocessing.
    normalize_first: float
    n_rows: int  # compatibility loader minimum; env regularizes through tau_op
    date: str = ""  # session date provenance (Phase 6: never a hard-coded constant)
    path_policy: str = "fixed"  # active historical config uses "split_pool";
    # "fixed" is retained for diagnostics and "bootstrap" remains reserved
    timezone: str = "America/New_York"
    missing_data_treatment: str = "error"
    split_id: str = ""
    train_date_range: tuple[str, ...] = ()
    validation_date_range: tuple[str, ...] = ()
    test_date_range: tuple[str, ...] = ()
    # Fail-closed provenance requirements for publication datasets.  Empty
    # strings keep older resolved configs/fixtures readable; active configs
    # specify all four and the artifact validator checks the sidecar exactly.
    source: str = ""
    price_type: str = ""
    quote_feed: str = ""
    artifact_normalization: str = ""


@dataclass(frozen=True)
class MidPriceConfig:
    """Mid-price model selector; exactly one branch is used per setting."""

    model: str  # "rough_heston" | "historical"
    rough_heston: Optional[RoughHestonParams] = None
    historical: Optional[HistoricalParams] = None


@dataclass(frozen=True)
class Algo1Params:
    """Projected CLOB clearing-signal initialization and smoothing."""

    H0: float  # explicit initial signal; revised baseline => 100
    eta_H: float  # smoothing coefficient; revised baseline => 0.95

    @property
    def tau(self) -> float:
        return self.eta_H

    @property
    def H0_from_mid(self) -> bool:
        """Compatibility flag: revised configs always use explicit ``H0``."""
        return False


@dataclass(frozen=True)
class RewardParams:
    """Three-regime reward constants (sec:MDP; rulings D4, D5, D8)."""

    k_star: int  # k* in f_c; legacy kappa=0.1 <=> k*alpha=10 <=> k*=1000
    lambda_inv: float  # terminal inventory penalty lambda; shared => 2.0
    q: float  # wrong-side shaping penalty; inactive headline => 0.0
    d: float  # cancellation cost unit d (cost d_t*c_t, D4); => 0.1
    shaping_enabled: bool  # common CLOB/interim/terminal shaping switch
    clawback_shaping: bool  # cancellation clawback ablation; baseline false
    numerical_guard: bool  # D8: optional far-out float guard, default OFF
    # |I_tau_cl| threshold when the guard is on — far outside the economic
    # range (|I| <= I0 + auction exposure ~ O(10^3)); binding is logged
    # loudly and asserted never to happen on seeded standard runs (D8).
    numerical_guard_bound: float = 1e9
    # Optional phase overrides.  ``None`` preserves the historical shared
    # switch, while pilots can keep economic CLOB cash and retain a dense,
    # purchase-sensitive auction signal.
    clob_shaping_enabled: Optional[bool] = None
    auction_shaping_enabled: Optional[bool] = None
    # Potential-based numerical centering: subtract the initial-mid value of
    # every inventory decrement and the remaining terminal inventory.  Across
    # a complete episode this is exactly the policy-invariant constant
    # S_mid_0*I_0, so action rankings and the manuscript objective are
    # unchanged while Bellman targets operate on PnL-scale differences.
    center_initial_inventory_value: bool = False

    @property
    def effective_clob_shaping(self) -> bool:
        return (
            self.shaping_enabled
            if self.clob_shaping_enabled is None
            else self.clob_shaping_enabled
        )

    @property
    def effective_auction_shaping(self) -> bool:
        return (
            self.shaping_enabled
            if self.auction_shaping_enabled is None
            else self.auction_shaping_enabled
        )


@dataclass(frozen=True)
class RLParams:
    """Learning and validation-objective constants shared by all algorithms."""

    chi: float  # revised finite-horizon Bellman factor; must equal one
    discount_mode: str = "undiscounted"
    # Periodic validation selects best.pt on Pi_lambda, never shaped return.
    checkpoint_metric: str = "risk_adjusted_pnl"
    # Dedicated training-only calibration episodes used to fit the frozen
    # common feature normalizer before any learning update.
    normalizer_fit_episodes: int = 32
    validation_size: int = 24
    validation_frequency_episodes: int = 100
    validation_patience_evals: int = 5
    test_size: int = 100
    h_cl_feature_enabled: bool = True


@dataclass(frozen=True)
class ActionGridParams:
    """Strategic discrete action envelope from manuscript Section 9."""

    V_max: int  # strategic CLOB submitted-volume cap; => 30
    L_max: int  # strategic CLOB quote offsets are 0..L_max; => 12
    beta: float  # strategic auction slope step; shared => 1
    K_max: int  # maximum strategic slope index k; shared => 32
    B_max: int  # ambient strategic auction price-offset bound; shared => 150 ticks
    auction_cancel_mode: str = "enabled"  # shared grids: enabled 254 | never 127
    # Numerical coarsening of the integer auction reference-price coordinate.
    # The ambient manuscript action remains integer-valued; a value >1 selects
    # a broad, regular subset without exploding the DQN output head.
    auction_offset_step: int = 1
    # ``frozen_mid`` enumerates the manuscript offset b directly.  The
    # ``indicative`` option instead enumerates a local displacement around the
    # currently observed indicative clearing price and translates it back to
    # an absolute manuscript offset before the order reaches the environment.
    # This is a numerical policy parameterization, not a change to Adm(x).
    auction_offset_center: str = "frozen_mid"
    auction_local_offset_max: Optional[int] = None
    # Optional non-uniform DQN slope subset, expressed as integer multipliers
    # of beta.  Empty preserves the canonical 1..K_max grid.  A geometric
    # subset gives fine control near zero and keeps the largest manuscript
    # slope without multiplying it by every offset/cancellation choice.
    auction_slope_multipliers: tuple[int, ...] = ()
    # ``multi`` is the manuscript ambient action family.  ``single_replace``
    # is a numerical policy-class restriction: once an agent schedule is live,
    # a new schedule must simultaneously cancel/replace it.
    auction_order_mode: str = "multi"

    @property
    def clob_volume_max(self) -> int:
        return self.V_max

    @property
    def clob_delta_min(self) -> int:
        return 0

    @property
    def clob_delta_max(self) -> int:
        return self.L_max

    @property
    def auction_K_grid_min(self) -> float:
        return self.beta

    @property
    def auction_K_grid_max(self) -> float:
        return self.beta * self.K_max

    @property
    def auction_K_grid_n(self) -> int:
        return self.K_max

    @property
    def auction_K_multipliers(self) -> tuple[int, ...]:
        if self.auction_slope_multipliers:
            return self.auction_slope_multipliers
        return tuple(range(1, self.K_max + 1))

    @property
    def auction_K_choice_count(self) -> int:
        return len(self.auction_K_multipliers)

    @property
    def auction_offset_max(self) -> int:
        return self.B_max

    @property
    def auction_template_offset_max(self) -> int:
        """Half-width represented by the policy's auction offset coordinate."""
        if self.auction_local_offset_max is None:
            return self.B_max
        return self.auction_local_offset_max


@dataclass(frozen=True)
class FeatureParams:
    """Legacy-pruned RL feature lists (ruling D11; AUDIT A.7 — exact defaults).

    Names use the PAPER sign convention (D3): N_buy = paper N^+ (buying MOs).
    """

    clob: tuple[str, ...]  # 8 dims; feat_clob main.py:1443-1450
    auction: tuple[str, ...]  # revision default adds cancel_admissible to legacy 7 dims
    # Affine normalization for the OPTIONAL price features ``h_cl_norm`` /
    # ``s_mid_norm`` only (the legacy raw ``h_cl`` / ``s_mid`` ignore these):
    # x_norm = clip((x - S0) / price_norm_scale, -clip, +clip), centered at the
    # initial mid S0. Used by the continuous agents (the raw ~100-valued price
    # features sit next to O(1) features and, before the small-slope clearing
    # guard, can spike); the discrete DQN keeps the legacy raw features.
    # Defaults are inert (no config that uses the raw names is affected).
    price_norm_scale: float = 1.0
    price_norm_clip: float = 20.0


@dataclass(frozen=True)
class BenchmarkParams:
    """AS / TWAP benchmarks (sec:benchmark; bounded common envelope, D23)."""

    z: float  # auction heuristic slope multiplier z; main.py:1048 => 10.0
    dust_threshold: float  # q < dust => no auction order; => 1e-2
    as_n_samples: int  # K-hat regression samples; main.py:1935 => 10000
    as_gamma: float  # AS risk aversion gamma; => 0.0
    as_sigma_rule: str  # "pooled_paths" for both active settings; single_path supported
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


def _migrate_legacy_schema(tree: dict[str, Any]) -> None:
    """Normalize the immediately preceding resolved-config schema in place.

    This deliberately recognizes only keys renamed by the manuscript
    revision.  It keeps committed result fixtures readable without weakening
    the strict unknown-key check for arbitrary misspellings.
    """

    grid = tree.get("grid")
    if isinstance(grid, dict) and "h" not in grid:
        tau_op, tau_cl = grid.get("tau_op"), grid.get("tau_cl")
        if isinstance(tau_op, int) and isinstance(tau_cl, int):
            grid["h"] = tau_cl - tau_op

    clob = tree.get("clob_flow")
    if isinstance(clob, dict):
        legacy = {"V_max": "V", "depth_decay": "rho_lob", "Lc": "L_max"}
        for old, new in legacy.items():
            if old in clob:
                if new in clob:
                    raise ConfigError(f"clob_flow: cannot specify both {old!r} and {new!r}")
                clob[new] = clob.pop(old)

    auction = tree.get("auction_flow")
    if isinstance(auction, dict):
        for old, new in (("K_min", "U1"), ("K_max", "U2")):
            if old in auction:
                if new in auction:
                    raise ConfigError(f"auction_flow: cannot specify both {old!r} and {new!r}")
                auction[new] = auction.pop(old)
        if "price_band_ticks" in auction:
            if "M1" in auction or "M2" in auction:
                raise ConfigError(
                    "auction_flow: price_band_ticks cannot be mixed with M1/M2"
                )
            band = auction.pop("price_band_ticks")
            auction["M1"], auction["M2"] = -band, band
        auction.setdefault("D_mu", 0.1)

    midprice = tree.get("midprice")
    rough = midprice.get("rough_heston") if isinstance(midprice, dict) else None
    if isinstance(rough, dict):
        for old, new in (
            ("rho", "rho_h"),
            ("kappa", "varsigma"),
            ("xi", "nu"),
            ("seconds_per_year", "s_star"),
        ):
            if old in rough:
                if new in rough:
                    raise ConfigError(
                        f"midprice.rough_heston: cannot specify both {old!r} and {new!r}"
                    )
                rough[new] = rough.pop(old)

    # Resolved artifacts written before the explicit-clock schema retain their
    # original meaning: old rough-Heston runs used seconds, while historical
    # paths have always used minutes. Active configs specify this key directly.
    if isinstance(grid, dict) and "time_unit" not in grid:
        model = midprice.get("model") if isinstance(midprice, dict) else None
        grid["time_unit"] = "seconds" if model == "rough_heston" else "minutes"

    algo1 = tree.get("algo1")
    if isinstance(algo1, dict):
        if "tau" in algo1:
            if "eta_H" in algo1:
                raise ConfigError("algo1: cannot specify both 'tau' and 'eta_H'")
            algo1["eta_H"] = algo1.pop("tau")
        from_mid = algo1.pop("H0_from_mid", None)
        if algo1.get("H0") is None and from_mid is True:
            if not isinstance(grid, dict) or "S0" not in grid:
                raise ConfigError("algo1.H0: cannot infer legacy H0 without grid.S0")
            algo1["H0"] = grid["S0"]

    reward = tree.get("reward")
    if isinstance(reward, dict):
        reward.setdefault("shaping_enabled", True)
        reward.setdefault("clawback_shaping", False)

    actions = tree.get("actions")
    if isinstance(actions, dict) and any(
        key in actions
        for key in (
            "clob_volume_max",
            "clob_delta_min",
            "clob_delta_max",
            "auction_K_grid_min",
            "auction_K_grid_max",
            "auction_K_grid_n",
            "auction_offset_max",
        )
    ):
        canonical = {"V_max", "L_max", "beta", "K_max", "B_max"}
        overlap = canonical.intersection(actions)
        if overlap:
            raise ConfigError(
                f"actions: cannot mix legacy grid keys with canonical keys {sorted(overlap)}"
            )
        try:
            actions["V_max"] = actions.pop("clob_volume_max")
            actions.pop("clob_delta_min")
            actions["L_max"] = actions.pop("clob_delta_max")
            old_k_min = actions.pop("auction_K_grid_min")
            old_k_value_max = actions.pop("auction_K_grid_max")
            old_k_index_max = actions.pop("auction_K_grid_n")
            actions["B_max"] = actions.pop("auction_offset_max")
        except KeyError as exc:
            raise ConfigError(
                f"actions: incomplete legacy action-grid schema; missing {exc.args[0]!r}"
            ) from None
        if old_k_index_max <= 0:
            raise ConfigError("actions.auction_K_grid_n must be positive")
        actions["beta"] = old_k_value_max / old_k_index_max
        actions["K_max"] = old_k_index_max
        # ``auction_K_grid_min`` was the old first sampled value.  The revised
        # grid is defined by beta*k, so it is intentionally not carried over.
        _ = old_k_min


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


def _parse_date_range(
    name: str, values: tuple[str, ...], *, required: bool
) -> tuple[date, date] | None:
    if not values:
        if required:
            raise ConfigError(
                f"midprice.historical.{name}: a [start, end] range is required"
            )
        return None
    if len(values) != 2:
        raise ConfigError(
            f"midprice.historical.{name}: expected [start, end], got {values!r}"
        )
    try:
        start, end = (date.fromisoformat(value) for value in values)
    except ValueError as exc:
        raise ConfigError(
            f"midprice.historical.{name}: dates must use YYYY-MM-DD"
        ) from exc
    if end < start:
        raise ConfigError(
            f"midprice.historical.{name}: end {end} precedes start {start}"
        )
    return start, end


def _validate_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Enforce cross-field contracts that a dataclass alone cannot express."""
    if cfg.grid.h != cfg.grid.tau_cl - cfg.grid.tau_op:
        raise ConfigError("grid.h must equal grid.tau_cl-grid.tau_op")
    if cfg.grid.time_unit not in ("minutes", "seconds"):
        raise ConfigError("grid.time_unit must be 'minutes' or 'seconds'")
    if cfg.grid.T_physical <= 0.0:
        raise ConfigError("grid.T_physical must be positive")
    if abs(cfg.grid.physical_time_per_grid_unit - 1.0) > 1e-12:
        raise ConfigError(
            "the simulator clock requires one physical time_unit per integer grid interval: "
            "grid.T_physical must equal grid.tau_cl"
        )
    if cfg.actions.auction_offset_step <= 0:
        raise ConfigError("actions.auction_offset_step must be positive")
    if cfg.actions.K_max <= 0 or cfg.actions.beta <= 0.0:
        raise ConfigError("actions.K_max and actions.beta must be positive")
    if cfg.actions.B_max <= 0:
        raise ConfigError("actions.B_max must be positive")
    if cfg.actions.auction_offset_center not in ("frozen_mid", "indicative"):
        raise ConfigError(
            "actions.auction_offset_center must be frozen_mid|indicative"
        )
    local_offset_max = cfg.actions.auction_template_offset_max
    if local_offset_max < 0 or local_offset_max > cfg.actions.B_max:
        raise ConfigError(
            "actions.auction_local_offset_max must lie in [0, actions.B_max]"
        )
    if local_offset_max % cfg.actions.auction_offset_step != 0:
        raise ConfigError(
            "the policy auction-offset half-width must be divisible by "
            "actions.auction_offset_step"
        )
    multipliers = cfg.actions.auction_K_multipliers
    if (
        not multipliers
        or tuple(sorted(set(multipliers))) != multipliers
        or multipliers[0] <= 0
        or multipliers[-1] > cfg.actions.K_max
    ):
        raise ConfigError(
            "actions.auction_slope_multipliers must be strictly increasing, "
            "unique, positive, and no larger than actions.K_max"
        )
    if cfg.actions.auction_order_mode not in ("multi", "single_replace"):
        raise ConfigError("actions.auction_order_mode must be multi|single_replace")
    if (
        cfg.actions.auction_order_mode == "single_replace"
        and cfg.actions.auction_cancel_mode != "enabled"
    ):
        raise ConfigError(
            "actions.auction_order_mode=single_replace requires cancellation enabled"
        )
    if cfg.rl.chi != 1.0:
        raise ConfigError("the revised finite-horizon objective requires rl.chi=1")
    if cfg.rl.discount_mode != "undiscounted":
        raise ConfigError("the revised objective requires rl.discount_mode='undiscounted'")
    if cfg.experiment.artifact_schema_version < 2:
        raise ConfigError(
            "revised runs require experiment.artifact_schema_version >= 2"
        )
    if cfg.rl.normalizer_fit_episodes <= 0:
        raise ConfigError("rl.normalizer_fit_episodes must be positive")
    if min(
        cfg.rl.validation_size,
        cfg.rl.validation_frequency_episodes,
        cfg.rl.validation_patience_evals,
        cfg.rl.test_size,
    ) <= 0:
        raise ConfigError("validation/test sizes, frequency, and patience must be positive")

    historical = cfg.midprice.historical
    if cfg.midprice.model == "historical":
        if historical is None:
            raise ConfigError(
                "midprice.model=historical requires midprice.historical"
            )
        if cfg.grid.time_unit != "minutes":
            raise ConfigError("historical mid-price paths require grid.time_unit='minutes'")
        # The committed one-session fixture remains readable for unit tests and
        # legacy characterization only.  Publication runs must provide three
        # explicit, chronological, nonoverlapping date ranges.
        compatibility = historical.split_id.startswith("legacy_")
        ranges = {
            "train_date_range": _parse_date_range(
                "train_date_range",
                historical.train_date_range,
                required=not compatibility,
            ),
            "validation_date_range": _parse_date_range(
                "validation_date_range",
                historical.validation_date_range,
                required=not compatibility,
            ),
            "test_date_range": _parse_date_range(
                "test_date_range",
                historical.test_date_range,
                required=not compatibility,
            ),
        }
        if not compatibility:
            train = ranges["train_date_range"]
            validation = ranges["validation_date_range"]
            test = ranges["test_date_range"]
            assert train is not None and validation is not None and test is not None
            if not (train[1] < validation[0] and validation[1] < test[0]):
                raise ConfigError(
                    "historical ranges must be chronological and nonoverlapping: "
                    "train end < validation start and validation end < test start"
                )
        if historical.missing_data_treatment not in ("error", "ffill"):
            raise ConfigError(
                "midprice.historical.missing_data_treatment must be error|ffill"
            )
        if historical.path_policy not in ("fixed", "split_pool", "bootstrap"):
            raise ConfigError(
                "midprice.historical.path_policy must be fixed|split_pool|bootstrap"
            )
    elif cfg.midprice.model == "rough_heston":
        if cfg.midprice.rough_heston is None:
            raise ConfigError(
                "midprice.model=rough_heston requires midprice.rough_heston"
            )
        if cfg.midprice.rough_heston.s_star <= 0.0:
            raise ConfigError("midprice.rough_heston.s_star must be positive")
    else:
        raise ConfigError(f"unknown midprice.model {cfg.midprice.model!r}")
    return cfg


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
    _migrate_legacy_schema(merged)
    return _validate_experiment_config(_build_dataclass(ExperimentConfig, merged))


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
