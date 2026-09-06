# H-anchored, economic-cash-flow training comparator

This follow-up implements the author's explicit 6 September 2026 definition.
It is separate from the completed 440-run v19 campaign and adds 40 synthetic
runs (four algorithms, ten seeds). Existing headline checkpoints are reused.
It has not been run at production scale.

## Reward definition

Both schemes retain H in observations and use the same H-centred auction quote
grid. The simulator, signed auction controls, risk coefficient, forecast fit,
architectures, feature transformations, reward scale, optimizer settings,
training budget, warmup, validation and test protocol are unchanged.

For the new cash-flow scheme, the reward is:

- CLOB: actual execution proceeds S_execution times executed quantity;
- interim auction: minus the actual cancellation fee, zero otherwise;
- final auction transition: also add actual signed auction cash, residual
  inventory marked at the frozen CLOB midprice, and minus lambda I_final^2.

No forecast-relative cash multipliers, fictive order credits, shaping clawbacks,
or extra terminal preferences are included. All potential differences (including
mark-to-market and the CLOB depletion corridor), the market-return control
variate, and the per-execution initial-value subtraction are disabled.

The sum of these literal cash-flow rewards is final economic wealth. Subtracting
the fixed initial endowment S0 I0 gives the reported economic objective. That
constant subtraction is not distributed across replay transitions. The terminal
mark and inventory penalty must remain: cash received alone is not bar-J-lambda.

The larger gross-cash Bellman targets are an intentional consequence of removing
all reward centering; the smoke test establishes finite updates, not comparable
convergence rates or stable learning throughout a full run. No method-specific
retuning or favorable result selection is performed for this comparison.

## Interpretation and previous experiments

The new contrast is headline J training with its existing reward conditioning
minus literal economic-cash-flow training without reward conditioning. It tests
the complete training scheme, not the isolated manuscript preference terms.
The old fixed-anchor preference comparison and dense-versus-sparse comparison
remain useful for separating mechanisms and must retain their original labels.

The existing `mechanism_economic_sparse` is not this comparator: it has a fixed
anchor and still retains CLOB/mark conditioning, the market control variate and
initial-value centering. No existing completed v19 cell is relabeled as cash-flow-only.

Potential-based auction credit is mathematically applicable to either objective;
it is reserved to the shaped side in this particular author-requested comparison.
The completed economic-dense arm is evidence of its separate economic learning
effect, not evidence that manuscript J preferences themselves are superior.

## Launch and outputs

After committing the reviewed changes and cleaning the tree:

```bash
cd /Users/juliusgraf/Learning-Market-Making
source .venv/bin/activate
python -m lmm.experiments.cashflow_comparison --jobs 10 --threads-per-job 1
```

Use `--dry-run` to inspect the 40 commands without training. The follow-up root
defaults to `results/revision_v19_cashflow`, outside the frozen v19 result tree.
The runner validates the saved synthetic headline controls before training,
then uses the existing queue and train/evaluate/paired-reference pipeline.
Full runs require the same clean-worktree provenance guard as the original runner.

Results per run include metrics, initial/selected/final checkpoints, validation
and evaluation records. After every run succeeds, `_comparison/index.html`
contains the direct paired economic comparison. That directory also contains:

- `comparison.csv`: economic means and paired differences, with pointwise 95%
  bootstrap intervals over the ten training-seed means;
- `by_seed.csv`: every paired seed-level result;
- `comparison.pdf` and `comparison.png`: the focused comparison plot;
- `manifest.json`: source identities and hashes of the completed input artifacts.

Positive differences favor headline training. Both schemes use economic checkpoint
selection; gross training-return levels must not be compared as if both were PnL.
Use `--report-only` to rebuild these outputs without any learning. The report
checks config matching, episode/environment-seed pairing, completion hashes, and
the raw economic reward/replay identities. The default 440-run launcher and
original publication report are unchanged.

## Bounded verification

Nineteen targeted tests passed across cash-flow reward accounting, identical
forecast/market/action trajectories under common actions, mechanism contracts,
and the queue/protocol tests. Eight smoke runs (four algorithms times two
schemes, one nonpublication seed) each completed four training episodes and
three test episodes, followed by the generated paired report.
Every cash-flow learner performed 166 CLOB and 120 auction updates with finite
logged final losses. These checks verify implementation/pipeline operation;
four training episodes cannot establish comparative learning or performance.

Smoke artifacts are in `/tmp/lmm-cashflow-smoke-20260906`. Neither protected TeX
file was modified. The exact new treatment is
`configs/treatment/mechanism_economic_cashflow.yaml`.
