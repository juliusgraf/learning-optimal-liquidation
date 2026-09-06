"""Configuration dataclasses and YAML loader (Phase 2, fully functional).

Every active experiment parameter lives in ``configs/*.yaml``.  The legacy
instantiation audit in ``audit/PARAMS_FROM_CODE.md`` is provenance only; revised
and author-resolved calibrations in the active config supersede it.  Nothing in
``src/`` hard-codes an experiment parameter; library code receives the
dataclasses defined here.

Loading model: ``load_config(base, *overlays, overrides=...)`` deep-merges the
YAML mappings left to right (later files win), then applies dotted CLI
overrides (``reward.d=0.2``), then builds the frozen dataclass tree with a
strict unknown-key check and scalar type coercion. ``save_resolved`` /
``to_dict`` round-trip exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import typing
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml

__all__ = [
    "ACTIVE_ARTIFACT_SCHEMA_VERSION",
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
    "economic_evaluation_config",
]


ACTIVE_ARTIFACT_SCHEMA_VERSION = 15


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
    setting: str  # "synthetic_rough_heston" | "historical_sp500_midquotes"
    episodes: int  # E; active settings use one matched budget
    master_seed: int  # single master seed; ruling D10
    results_root: Path  # gitignored output root
    artifact_schema_version: int = ACTIVE_ARTIFACT_SCHEMA_VERSION
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
    B_inf: int  # exogenous auction quote-support half-width in ticks; => 150

    @property
    def M1(self) -> int:
        """Lower exogenous quote bound, written ``M_1=-B_inf`` in the paper."""
        return -self.B_inf

    @property
    def M2(self) -> int:
        """Upper exogenous quote bound, written ``M_2=B_inf`` in the paper."""
        return self.B_inf

    @property
    def K_min(self) -> float:
        return self.U1

    @property
    def K_max(self) -> float:
        return self.U2

    @property
    def price_band_ticks(self) -> int:
        """Compatibility view for the symmetric exogenous proposal generator."""
        return self.B_inf


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
    path_policy: str = "fixed"  # supported: "fixed" | "split_pool"
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
    # Pool only the configured training dates across rebased symbols. Each
    # validation/test environment still replays its requested symbol alone.
    training_pool: str = "symbol"


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
    q: float  # purchase shaping attenuation; shared reference preference q=0
    d: float  # cancellation cost unit d (cost d_t*c_t, D4); => 0.1
    shaping_enabled: bool  # common CLOB/interim/terminal shaping switch
    clawback_shaping: bool  # exact cancellation reversal when shaping is active
    numerical_guard: bool  # D8: optional far-out float guard, default OFF
    # |I_tau_cl| threshold when the guard is on — far outside the economic
    # range (|I| <= I0 + auction exposure ~ O(10^3)); binding is logged
    # loudly and asserted never to happen on seeded standard runs (D8).
    numerical_guard_bound: float = 1e9
    # Optional phase overrides. ``None`` preserves the shared switch; explicit
    # values support matched reward treatments.
    clob_shaping_enabled: Optional[bool] = None
    auction_shaping_enabled: Optional[bool] = None
    # Potential-based numerical centering: subtract the initial-mid value of
    # every inventory decrement and the remaining terminal inventory.  Across
    # a complete episode this is exactly the policy-invariant constant
    # S_mid_0*I_0, so action rankings and the manuscript objective are
    # unchanged while Bellman targets operate on PnL-scale differences.
    center_initial_inventory_value: bool = False
    # Training-only telescoping control variate. It preserves J up to the
    # fixed initial potential and never changes reported reward/accounting.
    learning_potential: bool = False
    # Default preserves the manuscript exactly. Values below one are an
    # explicitly labeled alternative objective, never an implicit repair.
    auction_shaping_weight: float = 1.0

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
    relative_price_features: bool = False
    n_step: int = 1
    learning_credit_baseline: bool = False
    learning_clob_inventory_potential: bool = False
    auction_exposure_features: bool = False
    phase_normalization: bool = False
    auction_inventory_asinh: bool = False
    market_return_control_variate: bool = False
    structured_warmup_episodes: int = 0
    learning_starts_after_warmup: bool = False
    market_control_reference: str = 'linear'
    learning_rate_half_life_episodes: float = 0.0
    learning_rate_min_fraction: float = 0.1
    discount_mode: str = "undiscounted"
    # Periodic validation selects best.pt on Pi_lambda, never shaped return.
    checkpoint_metric: str = "risk_adjusted_pnl"
    # Dedicated training-only calibration episodes used to fit the frozen
    # common feature normalizer before any learning update.
    normalizer_fit_episodes: int = 32
    validation_size: int = 24
    validation_frequency_episodes: int = 100
    validation_patience_evals: int = 5
    # A validation candidate is reportable only after both phase learners have
    # performed this many economically relevant optimizer steps.  The active
    # DQN has no behavior lock; the complete auction action set is available
    # from episode zero, so every eligible auction update counts.
    checkpoint_min_clob_updates: int = 0
    checkpoint_min_auction_updates: int = 0
    # The untrained policy is never a candidate. Its economic validation score
    # is retained as a diagnostic comparator; an optional ablation may require
    # a mature candidate to improve on it before being reportable.
    checkpoint_require_initial_improvement: bool = False
    test_size: int = 100
    test_seed_namespace: int = 0
    h_cl_feature_enabled: bool = True


@dataclass(frozen=True)
class ActionGridParams:
    """Strategic discrete action envelope from the manuscript."""

    V_max: int  # strategic CLOB submitted-volume cap; => 30
    L_max: int  # strategic CLOB quote offsets are 0..L_max; => 12
    beta: float  # strategic auction slope step; shared => 1
    K_max: int  # maximum strategic slope index k; shared => 32
    # Manuscript B_inf: the absolute frozen-mid coordinate bound in
    # S_t^a = S_{tau_op}^{mid} + alpha*b_t^a. Validation requires this to equal
    # AuctionFlowParams.B_inf so one named parameter controls the common
    # strategic/exogenous price-deviation band.
    B_inf: int
    # Manuscript B_max: local policy coordinate ell in [-B_max, B_max].
    # This is 10 ticks in the headline run.
    B_max: int
    # Observable anchor for the local auction grid. H-on uses the current
    # indicative price; H-off must not recover the ablated signal indirectly
    # through action semantics and therefore uses the frozen auction-open mid.
    auction_anchor: str
    auction_cancel_mode: str = "enabled"  # full grid: enabled 1,346 | never 673
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
        """The complete manuscript lattice ``{1,...,K_max}``."""
        return tuple(range(1, self.K_max + 1))

    @property
    def auction_K_choice_count(self) -> int:
        return len(self.auction_K_multipliers)

    @property
    def auction_absolute_offset_max(self) -> int:
        return self.B_inf

    @property
    def auction_local_offset_max(self) -> int:
        """Compatibility alias for manuscript ``B_max``."""
        return self.B_max


@dataclass(frozen=True)
class FeatureParams:
    """Common 18-coordinate manuscript feature lists for both phases.

    Names use the PAPER sign convention (D3): N_buy = paper N^+ (buying MOs).
    """

    clob: tuple[str, ...]
    auction: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkParams:
    """AS / TWAP benchmarks (sec:benchmark; bounded common envelope, D23)."""

    z: float  # auction heuristic slope multiplier z; main.py:1048 => 10.0
    as_n_samples: int  # configured number of K-hat regression attempts
    as_gamma: float  # AS risk aversion gamma; => 0.0
    twap_delta_mode: str  # legacy "min" label => manuscript best ask, delta=1


@dataclass(frozen=True)
class AlgoConfig:
    """RL algorithm selection + hyperparameters (configs/algo/*.yaml; D9)."""

    name: str  # "dqn" | "ddpg" | "td3" | "sac"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    backend: str = "native"


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


def economic_evaluation_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Return the common economic-only validation/evaluation contract.

    Training may use dense manuscript shaping, but checkpointing and final
    policy comparison score only cash, fees, marking, and the terminal
    inventory penalty.  Explicit phase switches are forced off so no overlay
    can leak shaping into evaluation.
    """
    return dataclasses.replace(
        cfg,
        reward=dataclasses.replace(
            cfg.reward,
            shaping_enabled=False,
            clob_shaping_enabled=False,
            auction_shaping_enabled=False,
            clawback_shaping=False,
            learning_potential=False,
        ),
    )


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
            if "B_inf" in auction or "M1" in auction or "M2" in auction:
                raise ConfigError(
                    "auction_flow: price_band_ticks cannot be mixed with B_inf/M1/M2"
                )
            auction["B_inf"] = auction.pop("price_band_ticks")
        if "M1" in auction or "M2" in auction:
            if "B_inf" in auction:
                raise ConfigError("auction_flow: B_inf cannot be mixed with M1/M2")
            if "M1" not in auction or "M2" not in auction:
                raise ConfigError("auction_flow: M1 and M2 must be specified together")
            lower, upper = auction.pop("M1"), auction.pop("M2")
            if not isinstance(lower, int) or not isinstance(upper, int) or lower != -upper:
                raise ConfigError(
                    "auction_flow: legacy M1/M2 must define a symmetric integer band"
                )
            auction["B_inf"] = upper
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
        # Configs predating the explicit switch always used the shaped reward,
        # so ``True`` is the only backward-compatible interpretation here.
        # Every active schema-v12 stack is explicit (headline false; treatment
        # overlays true), and therefore never relies on this legacy fallback.
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
        canonical = {"V_max", "L_max", "beta", "K_max", "B_inf", "B_max"}
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
            legacy_absolute_bound = actions.pop("auction_offset_max")
        except KeyError as exc:
            raise ConfigError(
                f"actions: incomplete legacy action-grid schema; missing {exc.args[0]!r}"
            ) from None
        if old_k_index_max <= 0:
            raise ConfigError("actions.auction_K_grid_n must be positive")
        actions["beta"] = old_k_value_max / old_k_index_max
        actions["K_max"] = old_k_index_max
        actions["B_inf"] = legacy_absolute_bound
        # A frozen-mid legacy grid enumerated the entire absolute band. Using
        # the same value for the otherwise inactive local bound preserves it.
        actions["B_max"] = legacy_absolute_bound
        # ``auction_K_grid_min`` was the old first sampled value.  The revised
        # grid is defined by beta*k, so it is intentionally not carried over.
        _ = old_k_min

    if isinstance(actions, dict):
        # Revision-v5/v9 resolved configs called the absolute strategic bound
        # ``B_max`` and the manuscript-local bound
        # ``auction_local_offset_max``. Canonicalize those unambiguously to
        # B_inf (absolute) and B_max (local).
        if "auction_local_offset_max" in actions:
            if "B_inf" in actions:
                raise ConfigError(
                    "actions: cannot mix B_inf with legacy auction_local_offset_max"
                )
            if "B_max" not in actions:
                raise ConfigError(
                    "actions: legacy auction_local_offset_max requires legacy B_max"
                )
            absolute_bound = actions["B_max"]
            local_bound = actions.pop("auction_local_offset_max")
            if local_bound is None:
                # Legacy frozen-mid configs did not use a local coordinate.
                local_bound = absolute_bound
            actions["B_inf"] = absolute_bound
            actions["B_max"] = local_bound
        elif "B_max" in actions and "B_inf" not in actions:
            # Older frozen-mid resolved configs had one absolute B_max only.
            actions["B_inf"] = actions["B_max"]

        legacy_order_mode = actions.pop("auction_order_mode", None)
        if legacy_order_mode == "single_replace":
            raise ConfigError(
                "actions.auction_order_mode=single_replace is incompatible with "
                "the manuscript policy class; review the config and remove the "
                "restriction explicitly"
            )
        if legacy_order_mode not in (None, "multi"):
            raise ConfigError(
                "actions.auction_order_mode: unsupported legacy value "
                f"{legacy_order_mode!r}"
            )

        # Pre-v8 environment configs encoded this choice only implicitly via
        # the H feature flag. Preserve that meaning when reading them; every
        # newly saved resolved config writes the explicit canonical field.
        rl = tree.get("rl")
        h_enabled = (
            bool(rl.get("h_cl_feature_enabled", True))
            if isinstance(rl, dict)
            else True
        )
        actions.setdefault(
            "auction_anchor", "indicative" if h_enabled else "frozen_mid"
        )

    benchmark = tree.get("benchmark")
    if isinstance(benchmark, dict):
        # Pre-removal resolved configs carried an inert AS volatility
        # diagnostic. Accept and discard those two exact legacy keys while
        # retaining strict rejection of arbitrary benchmark misspellings.
        benchmark.pop("as_sigma_rule", None)
        benchmark.pop("as_sigma_n_paths", None)


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
    if cfg.experiment.episodes <= 0:
        raise ConfigError("experiment.episodes must be positive")
    if cfg.rl.n_step < 1:
        raise ConfigError("rl.n_step must be positive")
    if cfg.rl.test_seed_namespace < 0:
        raise ConfigError("rl.test_seed_namespace must be nonnegative")
    if cfg.rl.auction_exposure_features and not cfg.rl.relative_price_features:
        raise ConfigError('auction exposure coordinates require relative prices')
    if cfg.rl.auction_inventory_asinh and not cfg.rl.auction_exposure_features:
        raise ConfigError('auction inventory asinh requires auction exposure coordinates')
    if cfg.rl.structured_warmup_episodes < 0:
        raise ConfigError('structured_warmup_episodes must be nonnegative')
    if cfg.rl.market_control_reference not in ('linear','calibration'):
        raise ConfigError('market_control_reference must be linear or calibration')
    if not 0 <= cfg.reward.auction_shaping_weight <= 1:
        raise ConfigError("auction_shaping_weight must be in [0,1]")
    if (not math.isfinite(cfg.rl.learning_rate_half_life_episodes)
            or cfg.rl.learning_rate_half_life_episodes < 0
            or not 0 < cfg.rl.learning_rate_min_fraction <= 1):
        raise ConfigError("learning-rate half life must be nonnegative and minimum fraction in (0,1]")
    if cfg.algo is not None:
        if cfg.algo.backend not in ("native", "sb3"):
            raise ConfigError("algo.backend must be native or sb3")
        if cfg.algo.backend == "sb3" and cfg.algo.name not in ("ddpg", "td3", "sac"):
            raise ConfigError("the SB3 backend supports ddpg, td3 and sac")
    if not cfg.experiment.seeds or len(set(cfg.experiment.seeds)) != len(
        cfg.experiment.seeds
    ):
        raise ConfigError("experiment.seeds must be nonempty and unique")
    if not 0 < cfg.grid.tau_op < cfg.grid.tau_cl:
        raise ConfigError("grid times must satisfy 0 < tau_op < tau_cl")
    if cfg.grid.h != cfg.grid.tau_cl - cfg.grid.tau_op:
        raise ConfigError("grid.h must equal grid.tau_cl-grid.tau_op")
    if cfg.grid.time_unit not in ("minutes", "seconds"):
        raise ConfigError("grid.time_unit must be 'minutes' or 'seconds'")
    if not math.isfinite(cfg.grid.T_physical) or cfg.grid.T_physical <= 0.0:
        raise ConfigError("grid.T_physical must be finite and positive")
    if abs(cfg.grid.physical_time_per_grid_unit - 1.0) > 1e-12:
        raise ConfigError(
            "the simulator clock requires one physical time_unit per integer grid interval: "
            "grid.T_physical must equal grid.tau_cl"
        )
    if not math.isfinite(cfg.grid.alpha) or cfg.grid.alpha <= 0.0:
        raise ConfigError("grid.alpha must be finite and positive")
    if not math.isfinite(cfg.grid.S0) or cfg.grid.S0 <= 0.0:
        raise ConfigError("grid.S0 must be finite and positive")
    if cfg.grid.I0 <= 0 or cfg.grid.I_max < cfg.grid.I0:
        raise ConfigError("grid.I0 must be positive and grid.I_max must be at least I0")

    clob = cfg.clob_flow
    if not all(
        math.isfinite(value)
        for value in (
            clob.lambda0,
            clob.v_m,
            clob.gamma_m,
            clob.V_inf,
            clob.beta_a,
            clob.beta_b,
            clob.rho_lob,
        )
    ):
        raise ConfigError("CLOB flow parameters must be finite")
    if min(clob.lambda0, clob.v_m, clob.gamma_m, clob.V_inf) <= 0.0:
        raise ConfigError("clob_flow lambda0, v_m, gamma_m, and V_inf must be positive")
    if clob.V <= 0 or clob.V < clob.v_m:
        raise ConfigError("clob_flow.V must be positive and no smaller than v_m")
    if clob.beta_a <= 0.0 or clob.beta_b <= 0.0:
        raise ConfigError("clob_flow beta_a and beta_b must be positive")
    if not 0.0 < clob.rho_lob <= 1.0:
        raise ConfigError("clob_flow.rho_lob must lie in (0,1]")
    if clob.L_max <= 0:
        raise ConfigError("clob_flow.L_max must be positive")

    auction = cfg.auction_flow
    probabilities = (auction.p1, auction.p2, auction.p3, auction.p4)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
        raise ConfigError("auction_flow probabilities p1..p4 must lie in [0,1]")
    if not all(
        math.isfinite(value) for value in (auction.D_mu, auction.U1, auction.U2)
    ):
        raise ConfigError("auction_flow D_mu, U1, and U2 must be finite")
    if auction.D_mu <= 0.0 or auction.U1 <= 0.0 or auction.U2 < auction.U1:
        raise ConfigError(
            "auction_flow requires D_mu>0 and 0<U1<=U2"
        )

    if not math.isfinite(cfg.algo1.H0) or cfg.algo1.H0 <= 0.0:
        raise ConfigError("algo1.H0 must be finite and positive")
    if not math.isfinite(cfg.algo1.eta_H) or not 0.0 < cfg.algo1.eta_H <= 1.0:
        raise ConfigError("algo1.eta_H must lie in (0,1]")
    if cfg.actions.V_max <= 0 or cfg.actions.L_max < 0:
        raise ConfigError("actions.V_max must be positive and actions.L_max nonnegative")
    if cfg.actions.L_max > cfg.clob_flow.L_max:
        raise ConfigError("actions.L_max cannot exceed clob_flow.L_max")
    if (
        cfg.actions.K_max <= 0
        or not math.isfinite(cfg.actions.beta)
        or cfg.actions.beta <= 0.0
    ):
        raise ConfigError("actions.K_max and actions.beta must be positive")
    if cfg.actions.B_inf <= 0 or cfg.actions.B_max <= 0:
        raise ConfigError("actions.B_inf and actions.B_max must be positive")
    if cfg.auction_flow.B_inf <= 0:
        raise ConfigError("auction_flow.B_inf must be positive")
    if cfg.actions.B_inf != cfg.auction_flow.B_inf:
        raise ConfigError(
            "actions.B_inf must equal auction_flow.B_inf: both represent the "
            "manuscript's common absolute price-deviation bound"
        )
    if cfg.actions.B_max > cfg.actions.B_inf:
        raise ConfigError(
            "actions.B_max (local policy bound) cannot exceed actions.B_inf "
            "(absolute price-deviation bound)"
        )
    if cfg.actions.auction_cancel_mode not in ("enabled", "never"):
        raise ConfigError("actions.auction_cancel_mode must be enabled|never")
    if cfg.actions.auction_anchor not in ("indicative", "frozen_mid"):
        raise ConfigError("actions.auction_anchor must be indicative|frozen_mid")
    expected_auction_anchor = (
        "indicative" if cfg.rl.h_cl_feature_enabled else "frozen_mid"
    )
    if cfg.actions.auction_anchor != expected_auction_anchor:
        raise ConfigError(
            "actions.auction_anchor must be 'indicative' when "
            "rl.h_cl_feature_enabled=true and 'frozen_mid' when it is false; "
            f"got {cfg.actions.auction_anchor!r} with "
            f"h_cl_feature_enabled={cfg.rl.h_cl_feature_enabled}"
        )
    if not cfg.experiment.auction_enabled and (
        cfg.rl.h_cl_feature_enabled
        or cfg.reward.effective_clob_shaping
        or cfg.reward.effective_auction_shaping
    ):
        raise ConfigError(
            "experiment.auction_enabled=false requires rl.h_cl_feature_enabled=false "
            "and all reward shaping disabled"
        )
    if cfg.rl.chi != 1.0:
        raise ConfigError("the revised finite-horizon objective requires rl.chi=1")
    if not math.isfinite(cfg.reward.q) or not 0.0 <= cfg.reward.q <= 1.0:
        raise ConfigError("reward.q must lie in [0,1]")
    if (
        cfg.reward.k_star <= 0
        or not math.isfinite(cfg.reward.lambda_inv)
        or cfg.reward.lambda_inv < 0.0
        or not math.isfinite(cfg.reward.d)
        or cfg.reward.d < 0.0
    ):
        raise ConfigError(
            "reward requires k_star>0, lambda_inv>=0, and d>=0"
        )
    if (
        not math.isfinite(cfg.reward.numerical_guard_bound)
        or cfg.reward.numerical_guard_bound <= 0.0
    ):
        raise ConfigError("reward.numerical_guard_bound must be finite and positive")
    if cfg.rl.discount_mode != "undiscounted":
        raise ConfigError("the revised objective requires rl.discount_mode='undiscounted'")
    if cfg.experiment.artifact_schema_version != ACTIVE_ARTIFACT_SCHEMA_VERSION:
        raise ConfigError(
            "experiment.artifact_schema_version must equal the active schema "
            f"{ACTIVE_ARTIFACT_SCHEMA_VERSION}"
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
    if min(
        cfg.rl.checkpoint_min_clob_updates,
        cfg.rl.checkpoint_min_auction_updates,
    ) < 0:
        raise ConfigError("checkpoint maturity update thresholds must be nonnegative")

    benchmark = cfg.benchmark
    if not math.isfinite(benchmark.z) or benchmark.z <= 0.0:
        raise ConfigError("benchmark.z must be finite and positive")
    if benchmark.as_n_samples < 2:
        raise ConfigError("benchmark.as_n_samples must be at least 2")
    if benchmark.as_gamma != 0.0:
        raise ConfigError("only benchmark.as_gamma=0 is implemented")
    if benchmark.twap_delta_mode != "min":
        raise ConfigError("only benchmark.twap_delta_mode=min is implemented")

    historical = cfg.midprice.historical
    if cfg.midprice.model == "historical":
        if historical is None:
            raise ConfigError(
                "midprice.model=historical requires midprice.historical"
            )
        if cfg.grid.time_unit != "minutes":
            raise ConfigError("historical mid-price paths require grid.time_unit='minutes'")
        if (
            not math.isfinite(historical.normalize_first)
            or historical.normalize_first <= 0.0
            or historical.n_rows < cfg.grid.tau_op + 1
        ):
            raise ConfigError(
                "historical normalize_first must be positive and n_rows must cover "
                "minutes 0..tau_op"
            )
        if not historical.symbols or len(set(historical.symbols)) != len(
            historical.symbols
        ):
            raise ConfigError("historical symbols must be nonempty and unique")
        if historical.training_pool not in ("symbol", "all_symbols"):
            raise ConfigError("historical training_pool must be symbol or all_symbols")
        if historical.training_pool == "all_symbols" and historical.path_policy == "fixed":
            raise ConfigError("all_symbols training requires path_policy=split_pool")
        ranges = {
            "train_date_range": _parse_date_range(
                "train_date_range",
                historical.train_date_range,
                required=True,
            ),
            "validation_date_range": _parse_date_range(
                "validation_date_range",
                historical.validation_date_range,
                required=True,
            ),
            "test_date_range": _parse_date_range(
                "test_date_range",
                historical.test_date_range,
                required=True,
            ),
        }
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
        if historical.path_policy not in ("fixed", "split_pool"):
            raise ConfigError(
                "midprice.historical.path_policy must be fixed|split_pool"
            )
    elif cfg.midprice.model == "rough_heston":
        if cfg.midprice.rough_heston is None:
            raise ConfigError(
                "midprice.model=rough_heston requires midprice.rough_heston"
            )
        rough = cfg.midprice.rough_heston
        if not all(
            math.isfinite(value)
            for value in (
                rough.H,
                rough.rho_h,
                rough.v0,
                rough.theta,
                rough.varsigma,
                rough.nu,
                rough.s_star,
            )
        ):
            raise ConfigError("rough-Heston parameters must be finite")
        if not 0.0 < rough.H < 0.5:
            raise ConfigError("midprice.rough_heston.H must lie in (0,0.5)")
        if not -1.0 <= rough.rho_h <= 1.0:
            raise ConfigError("midprice.rough_heston.rho_h must lie in [-1,1]")
        if (
            rough.v0 < 0.0
            or rough.theta < 0.0
            or rough.varsigma < 0.0
            or rough.nu < 0.0
            or rough.s_star <= 0.0
        ):
            raise ConfigError(
                "rough-Heston v0, theta, varsigma, and nu must be nonnegative; "
                "s_star must be positive"
            )
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
