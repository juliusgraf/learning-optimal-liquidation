# Rough-Heston internal refinement

This is an opt-in numerical-accuracy extension of the existing Euler-type
scheme. The default, existing checkpoints, and manuscript results retain the
legacy decision-grid generator. No candidate step has been established as an
accuracy threshold, and no retraining or 520-run campaign was launched.

## Configuration and timing

`midprice.rough_heston.rough_heston_max_step_minutes` defaults to `null`.
`null` retains the legacy draw order and decision-grid recursion. A positive,
finite number enables internal refinement; zero, negative, Boolean, NaN and
infinite values are rejected. Refinement requires `grid.time_unit: minutes`
and `s_star: 98280` (`252 * 6.5 * 60`). Old clock configurations remain readable
in legacy mode.

`configs/rough_heston_refinement.yaml` is a separately named **0.5-minute
candidate**, intended as an overlay after the synthetic configuration. It
clears the legacy fitted forecast coefficients and uses a separate results
root. Clearing coefficients is preparation for a new fit, not a replacement
for fitting them before a training campaign.

The exogenous CLOB generator first constructs its complete realized clock.
The price model inserts auction opening (120 minutes in the active config),
deduplicates anchors, and subdivides each original interval into the smallest
power-of-two number of equal pieces satisfying the bound. Every tighter bound
therefore produces a nested mesh. The actual step may be smaller than the
configured maximum. Unrepresentable or excessively large individual meshes
fail explicitly rather than creating duplicate or zero-length intervals.

Each internal interval is converted to trading years once. The existing
correlated log-price Euler update and full-history Volterra variance recursion
run at every internal point. Computed negative variance is retained; only
coefficient evaluation uses its positive part. Time complexity remains
quadratic in internal node count and retained path memory remains linear.

Only original CLOB observations and the opening value are sampled from the
unrounded price path. Half-up tick projection remains in the environment and
never feeds back into log prices. The auction midprice stays frozen. Internal
points cause no actions, book snapshots, fills, forecast-moment updates,
rewards, or replay transitions. The opening forecast continues to inherit the
last CLOB forecast. A fixed policy may choose different actions in response to
changed revealed prices; its opportunities to act and chronology are unchanged.

## Random-stream audit

The environment currently uses one RNG sequentially for the pre-sampled CLOB
tape, price draws, and auction proposals. Exploration and replay belong to
separate `SeedBundle` components. Adding a named component would change sorted
spawn identities, so this extension does not change that list.

Legacy price simulation still consumes its original two normal draws per
positive observation interval. Refined simulation derives a private price RNG
from a SHA-256 domain-separated encoding of the post-CLOB RNG state, without
consuming an additional parent draw. At each original price interval it
reserves the same two scalar normal draws on the parent. Thus extra substeps
cannot consume order-flow, depth, auction, exploration, or replay draws, and
the auction starts from the exact legacy RNG state. Downstream state-dependent
eligibility can still alter conditional production auction draws.

The diagnostic explicitly replaces price innovations with two independent
Brownian drivers generated on the common finest internal grid, with variance
`diff(minutes) / 98280`. Each driver's increments are summed separately for
coarser meshes, including legacy. Reusing a seed alone would not provide this
coupling. The diagnostic legacy path has the legacy **mesh and scheme**, but
its coupled increments are not the original seeded production path.

For fixed-policy sensitivity, auction indicators and conditional marks use
independent common streams keyed by episode, auction decision, and proposal.
Skipping an ineligible cancellation cannot shift another proposal's marks or
a later decision's indicators. Cancellation selection uses the common stream
with each current eligible-set size. Eligibility, cancellation targets and
accepted updates remain state-dependent; accepted books are not forced equal.
The optional proposal-stream injection leaves default production behavior intact.

## Reproduce the diagnostic

From the repository root, price-only diagnostics require no policy artifacts:

```sh
.venv/bin/python -m lmm.experiments.refinement \
  --episodes 32 --seed 927401 --steps 1 0.5 0.25 \
  --output /tmp/rough-heston-price-diagnostic.json
```

To reproduce the saved fixed-policy diagnostic (requires the local trained run):

```sh
.venv/bin/python -m lmm.experiments.refinement \
  --run-dir results/revision_v20/synthetic_rough_heston/dqn_seed42 \
  --checkpoint best --episodes 32 --seed 927401 --steps 1 0.5 0.25 \
  --output /tmp/rough-heston-dqn-diagnostic.json
```

Outputs refuse overwrite. `--config` can be repeated for a different price-only
calibration, and `--steps`, `--episodes`, and `--seed` are configurable. `--run-dir`
uses that run's entire resolved configuration and verifies checkpoint loading
against its original contract before intentionally changing only the price
mesh in evaluation environments. The saved JSON records configuration, source
hashes, checkpoint hash, runtime versions, per-episode metrics and summaries.
Numerical outputs are reproducible in the same runtime; timing and allocation
measurements vary.

Price moments pool unrounded revealed nodes, including initial and opening
prices. Return moments pool original decision-interval log returns without
annualization; opening log returns have one observation per episode.
Coarse–fine discrepancies use matching revealed times and report both raw and
rounded prices against the finest **candidate**, which is not an exact reference.

Negative-node frequency pools all internal nodes, including initial and opening,
before positive-part truncation. It is inherently mesh-dependent. The separate
negative-time frequency weights the left-endpoint negativity indicator by each
interval's duration, then averages episodes equally. Forecast errors use the
existing episode metrics over original decision observations, then equal episode
weights. Paired shortfall differences subtract the legacy-mesh episode result;
positive differences mean increased cost. Standard errors use the paired episode
differences, not independent coarse and fine estimates.

Runtime and peak memory cover complete environment construction/reset with
coupled paths. `tracemalloc` reports peak traced allocations, not process RSS;
common finest-driver generation and policy loading are outside that measurement.
Policy episode time is recorded separately and includes its reset.

## Calibration and artifact compatibility

Refined environment contracts carry a versioned `rh-dyadic-v1` identity bound
to the exact rough-Heston parameters, maximum step, and grid. Legacy environment
identities and checkpoint dictionaries omit the newly introduced default fields,
so existing DQN/SB3 checkpoints remain readable. Refined contracts reject legacy
checkpoints and checkpoints from different refined meshes. Resolved configs
always serialize the explicit setting. Resume compares the complete config;
new training refuses existing output directories.

Fitted forecast overlays now carry `algo1.clob_forecast_price_generator`.
Legacy overlays default to `legacy`; nonempty weights with another generator
identity fail config validation. The fitter writes the correct refined identity:

```sh
.venv/bin/python scripts/diagnose_forecast_credit.py \
  --setting synthetic_rough_heston \
  --config configs/rough_heston_refinement.yaml \
  --episodes 256 --output results/rough_heston_refinement_v1/forecast_fit
```

This is a command for a future training-only fit; it was not executed for the
fixed-policy diagnostic. Apply its `forecast_overlay.yaml` **after** the refinement
overlay for training. Use the intended clearing-mechanism overlay consistently
in fitting and training. Changing `--override
midprice.rough_heston.rough_heston_max_step_minutes=...` requires a new fit/output
root. Existing campaign forecast contracts bind full configs and reject changed
cached fits. Existing publication roots and manifests are not rewritten.

For a new training campaign, refit forecast weights on training paths, refit the
training-only feature normalizer and inventory reference (the training entrypoint
does these for fresh runs), then retrain and reselect policies under the chosen
generator. Preserve validation/test separation and version all these artifacts
together. The user subsequently selected 0.25 minutes for a synthetic-only replacement
within the existing v20 roots. The prepared `--replace-synthetic` launch mode
is documented in [rerun_v20.md](rerun_v20.md). It refreshes 320 synthetic runs
and retains 200 historical runs; it has not been launched during preparation.

The saved diagnostic is **fixed-policy numerical sensitivity**: the original DQN
checkpoint, embedded normalizer/inventory reference, forecast coefficients and
all other calibration are held fixed. Its environments intentionally bypass the
new training compatibility check and are labeled accordingly. It creates no new
training artifacts. **Retrained-policy comparisons have not been performed** and
must not be inferred from these results.

## Files and verification

- `src/lmm/market/midprice.py`: nested grid, coupled increment aggregation,
  private refined RNG, internal recursion, raw-path and negative-variance diagnostics.
- `src/lmm/env/mdp.py`: passes the realized clock to the price model before simulation.
- `src/lmm/market/auction.py`: optional per-proposal streams for diagnostics.
- `src/lmm/experiments/refinement.py`: reproducible price and fixed-policy sensitivity CLI.
- `src/lmm/config.py`, `src/lmm/agents/{dqn,sb3}.py`, and
  `scripts/diagnose_forecast_credit.py`: configuration, provenance and compatibility.
- Synthetic/refinement YAMLs, root README, and the localized rough-Heston paragraph
  in `paper/main.tex`: configuration and legacy-result provenance documentation.
  Pre-existing manuscript edits were preserved.
- `tests/test_rough_heston_refinement.py`: grid/nesting, time conversion/coupling,
  analytic constant variance, direct full-history reference, negative variance,
  exact legacy episode fingerprint, independent streams, market chronology,
  rounding/freezing, learning counts, artifact validation and diagnostic smoke tests.

Measured results and remaining validation are in [results.md](results.md).
