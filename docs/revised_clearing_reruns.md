# Revised auction clearing: implementation and reruns

The purple projection in the supplied `main-14.tex` and finding 1 of
`review-2026-09-07-151509.tex` are implemented as **`max_volume_v2`**. The
documents are specification/reference material; unrelated review suggestions
are outside this change. Retained paper numbers, v19 campaign outputs and
fitted forecast weights remain evidence for **`nearest_tick_v1` only**.

## Mechanism and numerical conventions

For a nonnegative continuous root `p_star`, compare unique integer indices
`floor(p_star/alpha)` and `ceil(p_star/alpha)`. Evaluate prices as `alpha*k`.
At each price, keep each exogenous participant separate and sum all live
strategic linear and external benchmark schedules into **one signed agent**
before taking positive/negative parts. Include sell market orders in supply
and buy market orders in demand. Select lexicographically by matched volume,
negative root distance, then price. Monotone supply and demand guarantee that
at least one global tick-volume maximizer is among these candidates, including
books with a volume plateau.

Numerical conventions are part of v2 and live in
[`clearing.py`](../src/lmm/market/clearing.py):

- Grid membership and distance ties use `max(1e-12, 8*ulp(p_star/alpha))` in
  **tick units**. A root within this distance of an integer is treated as
  on-grid. Indices at or above `2**48` are rejected as beyond the supported
  floating-point precision range.
- Matched volumes tie when their difference is at most
  `1e-12 + 1e-12*max(volume_minus, volume_plus)` in inventory units. These
  tolerances resolve numerical noise; no economic tolerance is fitted.
- Nonlinear v2 bisection refines to `alpha*1e-12` price units or adjacent
  representable floats, handles exact endpoint/zero roots and cannot stall
  on a sub-ULP interval. The legacy external solver retains its `1e-10` price
  tolerance. Nonlinear schedules must be continuous and monotone, and admit
  a nonnegative root with a positive linear background slope. For a signed
  external schedule, admissibility checks the complete book's net supply at
  zero, rather than incorrectly requiring the background's `R` alone to be
  nonnegative.
- Explicit market-order sides must be finite, nonnegative and agree with
  `net_market_volume = buy - sell` within `math.isclose(rtol=1e-12, atol=1e-12)`.
  Compatibility callers supplying only net volume imply its minimal one-sided
  decomposition. Unknown equal opposing flow adds a price-independent volume
  constant, so does not change the maximizing tick; report actual total volumes
  only when both sides are supplied.
- The inherited linear admissibility slack is `1e-12`; allocation balance is
  checked with `math.isclose(rtol=1e-10, atol=1e-10)`.

For linear aggregates the mathematical imbalance bound is `D*alpha`, up to
the above numerical tolerances. The review example has `D=10`, root `100.004`,
selected price `100.01`, matched volume `0.93`, and residual `0.06`; nearest
clearing matches `0.90` at `100.00`. The nearest-tick bound `D*alpha/2` fails
for v2. The background linear `D` gives **no such bound** for nonlinear books.

`clear_linear`, `clear_with_external_schedule`, terminal allocation and
`Eq2Cache` share quantity/projection helpers. The public `solve_*` wrappers
retain their continuous-root API when `alpha` is omitted; supplying `alpha`
requests the same tick projection. `solve_monotone_clearing` remains a raw root
solver because net excess alone cannot identify matched volume. Raw Algorithm 1
is continuous and unchanged. The environment explicitly passes its configured
mechanism for all indications, settlement and leave-agent-out diagnostics.

Opening inherits the last CLOB forecast without projection. Each auction action
uses the previous indication; cancellation and submission affect the next
indication. Final settlement uses that final post-action book, with no extra
arrival. Pro-rata ratios, zero-volume execution, signed cash/inventory, fees,
cancellation credit reversal and reward accounting are preserved.

## Compatibility

| Mechanism | Artifact schema | Environment contract |
|---|---|---|
| `nearest_tick_v1` | 15 | `shaped-j-economic-eval-sb3-2026-09-05-v15` |
| `max_volume_v2` | 16 | `max-volume-auction-2026-09-07-v16` |

The base/setting configs retain v1 and the saved v19 weights. Missing mechanism
and forecast provenance fields in supported old configs/checkpoints mean v1.
No old checkpoint can load into v2, including direct native or SB3 API loading.
Report readers can read supported legacy outputs, but reject mixed mechanisms
and disagreements between config and metadata. Earlier unsupported schemas
remain rejected. Saved outputs and paper files are not rewritten.

Apply `configs/clearing/max_volume_v2.yaml` **after the setting**. It selects
schema 16 and clears old forecast weights. An empty tuple means raw Algorithm 1,
not a fitted v2 forecast. A subsequent fitted overlay must declare
`algo1.clob_forecast_mechanism: max_volume_v2`; nonempty weights tagged for a
different mechanism are rejected. Existing v19 coefficients are not retagged.

Training refits normalization and the inventory reference on its dedicated
training streams before initial validation, then trains and selects policies
from mature economic-validation checkpoints. Standalone reference commands
produce audit files; training deterministically repeats that same fit and does
not import legacy calibration. AS's CLOB impact regression is independent of
auction clearing; its audit is regenerated and its settlement is reevaluated.

## One-command revised campaign

After reviewing and committing the worktree, run from the repository root:

```bash
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1
```

The launcher selects the repository `.venv/bin/python` automatically. It requires
a clean worktree for the full campaign; the user manages that commit/cleanup.
The default output root is **`results/revision_v20`**, separate from retained v19.
`LMM_PYTHON` and `LMM_RESULTS_ROOT` remain explicit overrides for batch hosts.
Inspect the exact plan without fitting or training using:

```bash
scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --dry-run
```

The command performs the following stages in order:

1. Fit two revised forecasts (synthetic and pooled historical) under
   `_forecasts/<setting>/`. Each uses 256 no-order **training** paths; the
   additional 256 validation paths are diagnostic only. Historical training
   pools the five configured symbols; MSFT is the validation representative.
   Test paths do not enter fitting or coefficient selection.
2. Dispatch the existing **440-run main matrix** with up to ten simultaneous
   workers and one native/Torch computation thread per worker. Every worker
   receives the v2 overlay and its setting's verified forecast overlay. Training
   refits normalization and the inventory reference on dedicated training
   streams before validation, then trains and reselects a policy. The AS
   reference calibration is regenerated by the existing benchmark path.
3. For each run, evaluate the selected mature economic-validation checkpoint,
   initial policy, AS and TWAP on the same final seeds, compute paired benchmark
   differences and write the completion manifest. No legacy checkpoint is
   imported. Periodic replay-heavy checkpoints remain disabled.
4. Only after every run succeeds, generate the research report once under the
   new root. Its scope identifies `max_volume_v2`. This creates revised report
   outputs; it does not edit the manuscript or copy replacement paper numbers.

The count is `4 algorithms × 10 seeds × (1 synthetic + 5 historical symbols +
5 synthetic controls) = 440`. The controls are fixed anchor, dense economic,
sparse economic, H-feature-off and no-auction. The prior **520** plan also
included two subsequent 40-run H-conditioned economic comparisons. Those extra
80 runs are not part of this command. The superseded manual planner and its
unexecuted command bundle have been removed. The two forecast fits and per-run
calibration/evaluation stages are not additional policy training runs.

Production caps remain 800 training episodes, economic validation uses 128
paths, and final evaluation uses 100 matched paths with one recorded trace per
policy, as in the existing launcher. The ten canonical seeds, historical dates
and simulation namespaces are preserved. The historical holdout remains
previously inspected, not a fresh holdout.

Forecast completion manifests bind file hashes to the mechanism, resolved
config, git commit, fitter, runtime versions and historical inputs. Exact
completed fits can be reused. Incomplete, modified, smoke-sized or mismatched
fits fail before learning jobs are dispatched; an existing opposite-mechanism
run root is rejected. A failed worker stops new dispatch and drains active jobs.
Rerunning the same command skips only validated exact-config completed runs;
it does not silently resume partial training. Keep incomplete artifacts for
diagnosis and choose a fresh root or move those specific failed outputs aside.
Queue state and per-job logs live under `_orchestration/`.

For an isolated smoke matrix (two forecast paths per split, two noncanonical
seeds, four training episodes and three evaluation paths), use a separate root:

```bash
LMM_RESULTS_ROOT=results/revised_clearing_smoke \
  scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --smoke --symbol MSFT
```

A smoke forecast cannot pass the full 256-path contract. Smoke results are
implementation checks, never policy or paper evidence. A standalone audit of
training reference calibration is also available via
`scripts/refit_clearing_reference.py --help`; production training already performs
its own seeded fit, so a duplicate audit stage is unnecessary.

Explicit legacy reproduction remains available:

```bash
LMM_RESULTS_ROOT=/absolute/scratch/legacy-reproduction \
  scripts/run_multiseed.sh --jobs 10 --threads-per-job 1 --legacy-clearing
```

That option retains v1 and the old fitted weights; without a root override it
uses `results/revision_v19`. Existing outputs are never overwritten. Standalone
legacy launchers/base configs retain their old defaults.

For the financial-market rationale and the limits of the exchange analogy, see
[auction market practice](auction_market_practice.md).

## Verification and outstanding empirical work

`tests/test_tick_projection.py` includes the review counterexample, on-grid and
zero roots, zero and tied volumes, market orders, aggregate no-self-trade,
cancellation, nonlinear external schedules, float boundaries, cache/wrapper
consistency, chronology and signed settlement. Its independent bounded tick
oracle checks 150 linear, 150 capped-external and 150 smooth signed-external
random books, plus balanced pro-rata execution. Provenance tests cover native
and SB3 checkpoint isolation and resolve every algorithm/setting/control worker configuration.

Tests and small temporary fitting smoke checks are implementation validation,
not revised scientific results. Full forecast/reference fits, all retraining
and policy reselection, matched final evaluation and updated paper results
remain unrun; the user will start the campaign with the command above.
