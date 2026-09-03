# Revised artifact and metrics schema

All current confirmation runs live under
`results/revision_v12/<experiment_name>/<run_name>/`. Readers require both the
current `artifact_schema_version` and the exact environment contract identifier
stored in checkpoints and evaluation metadata. Missing or mismatched values are
fatal; the pipeline does not load old checkpoints or result directories.

## Provenance

Every run saves:

- the complete resolved configuration;
- master and component seed information, including train/validation/test and
  normalizer-calibration episode seeds;
- git state, Python/platform information, and package versions;
- the realized decision grid for each train and test episode;
- fitted training-only feature-normalization state;
- the verified historical dataset manifest (digest, assets, timezone,
  missing-data rule, split identifiers/date ranges, and concrete session lists)
  when applicable;
- checkpoint-selection metric (`risk_adjusted_pnl`), ablation label, auction
  switch, explicit auction anchor, and deterministic DQN tie-breaking rule;
- risk-neutral AS calibration values `(A, k)`, their configured/seeded sources,
  and an explicit marker that no irrelevant volatility parameter was fitted;
- the physical clock (`time_unit=minutes`, `tau_op=120`, `tau_cl=150`) in the
  resolved config, runtime metadata, evaluation metadata, and realized-grid
  records.

## Episode accounting

The following economic and shaping quantities are separate fields in both
training metrics and evaluation records:

| Field | Meaning |
|---|---|
| `clob_economic_cash` | sum of actual CLOB execution cash |
| `auction_economic_cash` | clearing price times actual rationed signed auction fill |
| `cancellation_fees` / `cancel_cost` | actual cancellation fees |
| `residual_mark` | terminal inventory marked at frozen auction-open midprice |
| `terminal_penalty` / `inventory_penalty` | `lambda_inv * I_final^2` |
| `clob_shaping_adjustment` | CLOB training reward minus CLOB economic cash |
| `auction_interim_shaping` | net cumulative fictive interim shaping after cancellation clawbacks |
| `auction_shaping_clawback` | signed cumulative shaping originally credited to canceled schedules and subtracted from reward |
| `auction_terminal_shaping` | terminal purchase-side shaping, applied once to aggregate cash |
| `reward_baseline_adjustment` | optional policy-invariant subtraction of initial inventory value from the training reward |
| `training_return` / `return_undisc` | undiscounted resolved reward (centered economic in headline training; shaped in `shaping_on` treatment training; economic-only in validation/final evaluation) |
| `pnl` | marked-to-market PnL, including cancellation fees |
| `risk_adjusted_pnl` | `pnl - inventory_penalty` (`Pi_lambda`) |

Compatibility aliases `liquidation_pnl_gross`, `liquidation_pnl_net`, and
`economic_objective` are written for decomposition checks, but they are not
used to select the reported policy. The canonical identities are:

```text
pnl = clob_economic_cash
    + auction_economic_cash
    + residual_mark
    - initial_mid * initial_inventory
    - cancellation_fees

risk_adjusted_pnl = pnl - terminal_penalty
```

With shaping disabled and reward centering disabled:

```text
return_undisc = risk_adjusted_pnl + initial_mid * initial_inventory
```

With `reward.center_initial_inventory_value=true`, the same value is removed
incrementally as inventory changes and at the terminal mark.  This is a
potential-based numerical transformation: the full-episode adjustment is
exactly `-initial_mid * initial_inventory`, policy rankings are unchanged, and
an unshaped `return_undisc` equals `risk_adjusted_pnl`.

Normalized fields are also written:

- `pnl_per_initial_notional`;
- `risk_adjusted_pnl_per_initial_notional`;
- `pnl_bps`;
- `risk_adjusted_pnl_bps`.

## `metrics.csv`

One row is written per training episode. In addition to the accounting fields,
it contains:

- episode/environment seed and epsilon;
- initial and final inventory, negative-terminal-inventory indicator and
  magnitude, CLOB and auction executed quantities, clearing price, and signed
  terminal fill;
- continuous clearing price, rounded price, residual imbalance,
  `Q_supply`, `Q_demand`, `rho_supply`, and `rho_demand`;
- carry-over slope, fallback use, leave-agent-out price, agent price
  displacement, and self-trade count;
- `H_cl` bias/MAE/RMSE and improvement against contemporaneous/opening mids;
- phase-specific loss, gradient, TD-error, update-count, and replay-size
  diagnostics;
- cumulative phase-specific maturity update counts and whether each periodic
  validation candidate was maturity-eligible and economically reportable;
- `eval_return_mean`, `eval_pnl_mean`,
  `eval_risk_adjusted_pnl_mean`, and `eval_checkpoint_score` at validation
  checkpoints;
- `wall_clock_s`, the only intentionally nondeterministic column.

The initial untrained policy and every pre-maturity validation are diagnostic
only. `best_selection.yaml` records the required and observed phase counts,
the initial-policy validation score, and whether improvement over that score
was required. In the active configuration that requirement is false: every
mature validation is reportable, irrespective of the initial score.
Early-stopping patience starts only after the first eligible validation, and
`best_mature.pt` remains diagnostic. `selection_failure.yaml` is written only
when the run never reaches checkpoint maturity (or when a noncanonical overlay
explicitly reenables the initial-improvement gate and no mature candidate
passes it).

Every nonterminal replay row has Bellman coefficient one. Rewards are stored at
the common `1e-3` scale for all four learners and are never clipped.

## `eval/records.csv`

There is one row per `(policy, test episode)`. Policies share the same
`env_seed` within an episode. The record includes all primary accounting
outputs and terminal diagnostics, notably:

- `pnl`, `risk_adjusted_pnl`, and normalized/bps versions;
- `I_final`, negative-inventory frequency/magnitude;
- CLOB cash, auction cash, cancellation fees, and signed fill;
- economic reward decomposition and economic-only evaluation return;
- continuous/rounded clearing prices and residual;
- pro-rata quantities/ratios, carry-over/fallback state, price displacement,
  and self-trade count.

`self_trade_count` must be zero. Inventory and cash use the actual pro-rata
fill, never requested quantity.

## Evaluation diagnostics

The evaluation directory also contains:

- `realized_grids.jsonl` — test grid, terminal time, and physical time unit per
  policy/episode;
- `proposal_diagnostics.csv` — proposed, accepted, validity-rejected, and
  ineligible counts/rates for each of the six exogenous proposal types;
- `action_diagnostics.csv` — raw normalized proposals, projected five-coordinate
  actions, configured/per-step auction-anchor coordinates,
  saturation/rounding/projection counts, and cancellation execution;
- `clearing_diagnostics.csv` — continuous and rounded prices, residual,
  allocation quantities/ratios, carry-over/fallback, and price displacement;
- `h_forecasts.csv` and `h_forecast_summary.csv` — signed bias, MAE, RMSE, and
  benchmark improvements grouped by physical time-to-close in minutes;
- `traces/<policy>_ep<i>.csv` — per-step anatomy traces, including auction
  anchor label, absolute anchor coordinate, and anchor price.

## Paired policy differences

`eval/policy_difference_<benchmark>.csv` contains:

```text
episode, env_seed, benchmark_value, policy_value,
policy_minus_benchmark, cumulative_policy_minus_benchmark
```

The default value is `risk_adjusted_pnl`; `pnl` is an explicit alternative.
The policy and benchmark rows must have identical episode sets and environment
seeds. These are fixed-policy cumulative differences, not regret.

## Tables and figures

Single-setting tables use risk-adjusted PnL as the primary outcome. Headline
training return is centered economic risk-adjusted PnL; the explicit shaping
treatments retain shaped return in training metrics as a diagnostic.
Evaluation records use the economic-only replay contract. Cross-seed synthetic comparisons report an
IQM and bootstrap interval over per-seed means; policy-vs-benchmark intervals
are computed from paired per-seed differences.

The synthetic cross-treatment generator emits
`synthetic_treatment_contrasts_multiseed.{tex,csv}` plus
`synthetic_treatment_contrasts_by_seed.csv`. The latter retains the positive
and negative run directories, algorithm, master seed, episode count, mean
difference, and a SHA-256 digest of the exactly matched evaluation-seed list.
Every contrast uses the sign first-named treatment minus second-named treatment.

Historical outputs include both forms required for cross-asset comparison:

- `historical_results_full` and `historical_results_improvements` — currency units;
- `historical_results_full_bps` and `historical_results_improvements_bps` — basis points of
  initial notional;
- `historical_results_multiseed` and `historical_results_multiseed_bps` — cross-seed forms.

Figure inputs are saved artifacts only; figure generation never steps an
environment. Main products include training diagnostics, paired policy
difference curves, episode/benchmark anatomy, cancellation behavior,
evaluation distributions, algorithm comparisons, convergence, reward
decomposition, and multiseed policy differences.
