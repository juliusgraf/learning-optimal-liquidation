# Unified simulator calibration and promotion gates

This document separates diagnosed defects, provisional numerical choices, and
evidence required before any result enters the paper. A setting is not promoted
because one seed or one asset happens to beat a benchmark.

All active simulator rates in this document use the common minute clock and
one shared simulator. The setting overlays change only the exogenous mid-price:
rough Heston or SIP midpoint replay. The common chronology is
`tau_op=120` minutes, `tau_cl=150` minutes, CLOB `lambda0` per minute, and
auction proposal probabilities per one-minute decision. The minute-clock
unified-simulator contract stores new runs in the revision-v5 result namespace.

## Diagnosed defects in revision-v2 results

1. The historical input was a normalized yfinance one-minute close proxy, not
   a quote midpoint. Revision v3 uses SIP bid/ask events, preserves a raw event
   archive and raw-USD midpoint CSV, and rebases only inside the model after a
   session is selected.
2. The old settings used insufficient absolute support and persistence. The
   historical CLOB additionally used `lambda0=60`; the synthetic CLOB used
   `V_inf=15`, `rho_lob=0.5`, and only 12 absolute tick levels. Both produced
   zero carry-over in matched policy-free checks. The shared replacement uses
   `lambda0=1`, `V_inf=2`, `rho_lob=0.96`, and 200 exogenous levels.
3. The auction training signal could reward a purchase-side schedule when
   `q=1`, retained fictive reward after cancellation, and was orders of
   magnitude larger than the economic checkpoint criterion. Learned agents
   repeatedly replaced schedules, paid cancellation costs, and created signed
   terminal exposure even after nearly complete CLOB liquidation.
4. A shared 5,000-transition warm-up delayed auction learning until roughly
   episode 167. The 1,022-way absolute auction head was simultaneously too
   large near the live indicative price and too narrow for observed absolute
   indicative-price displacement.
5. Bellman rewards carried the policy-invariant initial inventory notional.
   Revision v3 subtracts that value incrementally. This is an exact
   potential-based transformation and does not change complete-episode policy
   rankings.

## Shared publication-candidate values

These values are in `configs/base.yaml` and `configs/algo/*.yaml`; neither
active setting overlay contains a simulator or learning override:

| Component | Previous synthetic / historical | Shared candidate | Reason |
|---|---:|---:|---|
| CLOB arrival intensity `lambda0` (per side/minute) | 1 / 60 | 1 | Treat one simulator arrival as an aggregate burst instead of one exchange print. |
| Top depth scale `V_inf` | 15 | 2 | Keeps displayed depth commensurate with the reduced aggregate-arrival rate and the 100-unit inventory. |
| Geometric depth persistence `rho_lob` | 0.50 | 0.96 | Joint synthetic/historical calibration retaining economically relevant depth away from the top level. |
| Absolute book depth `L_max` | 12 | 200 | Covers the observed within-session absolute-tick displacement required by Algorithm 1. |
| Network widths | 64, 64 | 128, 128 | Capacity for the common 18-feature state without making the output head the bottleneck. |
| CLOB/auction replay warm-up | 5,000/5,000 | 2,000/512 | Both phase networks update before the first validation checkpoint. |
| Ambient auction offset | `[-25,25]` | `[-150,150]` | Covers observed absolute indicative displacement. |
| Policy offset templates | 51 absolute | 21 local (`H +/- 10` ticks) | Executes an absolute manuscript offset while removing irrelevant far-away choices. |
| DQN slope choices | 10 linear | `{1,2,4,8,16,32}` | Fine control near zero and retention of the old maximum with six choices. |
| Auction templates with cancellation | 1,022 | 254 | Lower output/sample complexity. |
| Agent order mode | multiple live | cancel-and-replace | Prevents accidental stacking; the current replacement survives the same-step cancel, matching chronology. |
| Training signal | manuscript shaping | centered economic objective | Aligns learning and checkpoint selection; shaping remains an explicit treatment. |

The joint calibration passes a 100-episode policy-free gate for both price
sources. Rough Heston has 8% fallback use, 92% positive carry-over, and median
carry-over slope 728.11; the old synthetic book had 100% fallback and zero
carry-over on the same seeds. Across 100 episodes for each of five SIP assets,
pooled fallback use is 2%, positive carry-over is 98%, and median carry-over
slope is 855.15; per-asset fallback is at most 4% and positive carry-over is at
least 96%. These validate simulator structure, not learned-policy superiority.

## Cancellation semantics

Cancellation is enabled in the headline configuration. At auction open there
is no prior live agent schedule, so manuscript admissibility forces `c=0`.
After the agent submits a positive-slope schedule, cancel-all is available at
every later auction decision. In `single_replace` mode, submitting a new
positive-slope schedule while one is live requires `c=1`: all strictly prior
orders are removed and the current order remains live. Only the explicit
`no_cancellation` treatment disables the coordinate.

The DQN stage-1 curriculum holds the auction no-op while the CLOB policy is
learned. This changes the behavior policy during those training episodes; it
does not remove cancellation from the environment or the post-unlock action
set.

## Bounded structural evidence

A 100-episode DQN pilot on the legacy MSFT proxy (seed 42), with 100 fixed
validation episodes and 100 disjoint test episodes, selected episode 49 and
obtained mean risk-adjusted PnL 8.01 versus 3.37 for AS and -5.75 for TWAP. The
paired test differences were 4.64 `[3.38, 5.95]` versus AS and 13.76
`[11.36, 16.31]` versus TWAP. The selected policy liquidated in the CLOB and
submitted no auction schedule. This proves that the reward/action/curriculum
changes can prevent the previous collapse; it does **not** validate the new
historical experiment because the pilot used the superseded Yahoo proxy, one
asset, and one seed.

A matched 100-episode minute-clock rough-Heston diagnostic (seed 314159) moved
the selected DQN's mean paired difference from -70.41 to -2.98 against AS and
from -67.35 to +14.03 against TWAP. The improvement in relative edge was 67.44
against AS (bootstrap interval [53.56, 82.89]) and 81.38 against TWAP
([67.19, 97.10]). The selected unified checkpoint was the safe initial policy;
a 300-episode follow-up did not select a learned checkpoint. This establishes
that the shared calibration removes the synthetic collapse and improves bounded
performance, but it is not evidence that trained DQN beats AS. Only the
multi-seed 800-episode confirmation can support that claim.

## Promotion gates

Run these in order:

1. Build the SIP true-midquote artifact and verify its digest/provenance.
2. Run `lmm-diagnose-simulator --assert-ready` separately with each active
   mid-price overlay; require every series/symbol's fallback rate at or below
   10%, positive carry-over in at least 90% of episodes, no nonfinite state,
   and no invalid quote-age observation.
3. Run the 100-episode stage-1 pilot on every asset. Require a finite policy,
   no catastrophic terminal-inventory tail, and positive paired mean against
   both AS and TWAP. Treat a confidence interval crossing zero as inconclusive.
4. Run at least three seeds of the 800-episode confirmation only after all
   stage-1 gates pass. Select checkpoints on a fixed 100-episode validation
   panel and evaluate once on the disjoint test stream.
5. Report cross-seed paired differences in basis points. A paper claim that RL
   beats a benchmark requires a positive cross-seed interval, not merely a
   positive pooled episode mean.

If the auction policy still degrades after its curriculum unlock, keep the
economic objective and tune in this order: lower auction learning rate, lower
auction exploration scale, enlarge the no-op prior margin, then extend the
auction-only curriculum. Do not re-enable the unscaled fictive shaping term to
manufacture denser rewards.
