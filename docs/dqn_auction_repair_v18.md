# DQN auction follow-up: objectives, diagnosis and bounded verification

This report supersedes the DQN specification and auction findings in
[the earlier v18 report](pathology_repair_v18.md). Earlier artifacts remain
unchanged. The current executable configuration uses one-step Double DQN,
auction reference centering, explicit exploration coverage for zero-slope
controls, and corrected shared warm-up projection. No market, reward,
inventory-penalty or benchmark parameters changed in this follow-up.

## Training and evaluation are different, intentionally

Headline training optimizes the author-approved **weighted shaped criterion**
`J_omega`, with auction shaping weight `omega=0.0001`, `q=0`, and CLOB shaping
enabled. This is the weighted amendment to J in the protected manuscript;
it must not be described as the unchanged original unweighted formula.
The episode's `training_return` is `J_omega - S0*I0`. The subtraction is a
policy-independent constant. Replay also uses the existing telescoping
potential and policy-independent exogenous-market control variate, preserving
expected policy differences. SAC additionally uses its standard entropy
regularization during learning.

Validation, checkpoint selection, early stopping, AS/TWAP comparisons and final
evaluation use **bar-J-lambda**. The implementation is

```
pnl = CLOB cash + actual signed auction cash + frozen-mid residual mark
      - S0*I0 - cancellation fees
risk_adjusted_pnl = pnl - lambda*terminal_inventory**2
```

The paper's bar-J already subtracts `S0*I0`; that value is not subtracted a
second time. `economic_evaluation_config` forces all objective-changing shaping
switches off. The production trainer only accepts `risk_adjusted_pnl` as its
checkpoint metric. Shaping-off treatments intentionally train on the economic
criterion with common invariant numerical conditioning. None of these learning
algorithms is asserted to attain a global maximum.

Relevant code: [reward definitions](../src/lmm/env/rewards.py),
[replay/accounting loop](../src/lmm/rl/loops.py),
[economic evaluation configuration](../src/lmm/config.py),
[production selection](../src/lmm/experiments/train.py), and
[final evaluation](../src/lmm/experiments/evaluate.py).

## What caused the negative auction contribution

1. **The shared warm-up did not execute its declared abstention mode.** A
   zero-slope proposal retained a random nonzero offset. Projection to the
   discrete grid could choose a positive-slope order at that offset instead of
   the canonical `(K,ell,c)=(0,0,0)`. Reconstructing the old draws produces
   phantom orders for 53 of 64 tested seeds. Each affects eight designated
   abstention episodes in the 32-episode cross-design. Zero-slope proposals
   now have exactly zero offset. This repair applies to all four algorithms.
   The [projection audit](verification_v18/auction_repair/warmup_projection.json)
   preserves the original and corrected actions.

2. **Uniform action exploration barely covered the two zero-slope controls.**
   With 1,346 auction actions, waiting and cancelling without submitting another
   schedule each received only 1/1,346 of uniform exploration. The original
   seed-881 saved replay contained just 12 cancellation-only rows out of 6,000
   auction rows. Conditional on epsilon exploration, DQN now draws with equal
   probability from the admissible zero-slope controls or the complete
   admissible grid. Full action support is retained. Evaluation remains the
   exact masked Q argmax.

3. **The auction Q reference could move through learned action parameters.**
   The old representation used an unconstrained shared non-noop coefficient
   and an uncentered learned action embedding. The auction now uses
   `Q(s,a)=V(s)+u(s)·(e(a)-e(noop))/sqrt(32)`, with the global coefficient
   fixed at zero. Thus `Q(s,noop)=V(s)` exactly; no-op training cannot generate
   action-encoder gradients. An exact mask also removes small GEMM/GEMV
   roundoff differences that Adam could amplify. This is a learned reference,
   not a prescribed no-order policy. The CLOB representation remains unchanged.

4. **Uncorrected three-step returns included later exploratory actions.**
   The earlier v18 candidate used three-step replay without off-policy return
   correction. A later exploratory cancellation or signed trade was therefore
   included in an earlier action's target, even though the greedy continuation
   could choose differently. The active DQN again uses its standard one-step
   Double-DQN target. Phase-junction continuation and the undiscounted objective
   are preserved, and terminal settlement is included exactly once.

The first three changes improved the original seeds but did not finish the
repair: an unused seed (929), checked after freezing that candidate, still lost
0.138 bps from its auction on 512 validation paths (path SE 0.043). An explicit
known-fee Q decomposition also failed to remove losses consistently and remains
an **inactive rejected diagnostic**. Restoring one-step targets was the final
accepted change. These sequential failures are included in the audit, rather
than reporting only the first promising small sample.

## What the auction number means

For a fixed learned CLOB policy and the same exogenous path, auction value is
the economic objective with the learned auction minus that with no strategic
auction orders. Its exact decomposition is

```
Z*(auction_price - frozen_mid)
+ lambda*(auction_open_inventory**2 - terminal_inventory**2)
- cancellation_fees
```

It measures direct use of the auction conditional on the CLOB policy. It is
distinct from retraining without auction access, which can change CLOB behavior.
The publication output retains both comparisons. A small negative number is
not automatically a bug: J and bar-J-lambda differ, and the shaped optimizer
can rationally accept a small economic cost. The earlier large negative
contributions and action-value errors could not be dismissed on that basis.

## Verification protocol and evidence

No full 800-episode production run or final-test evaluation was launched.
This follow-up used 26 capped 200-episode training attempts (5,200 episodes),
including every rejected prototype and the shared-warm-up checks for native
DDPG, TD3 and SAC. These are development comparisons, not 26 independent
confirmatory replications. Prior development in the earlier v18 report is
additional and remains separately recorded.

Each bounded policy was selected solely by mean economic objective on 128
fixed validation paths, subject to the production phase-update maturity
threshold. It was then checked on 128 fresh validation flow paths, paired with
AS, TWAP and its exact same-CLOB auction-noop counterfactual. The candidate's
configuration, code hashes, seed 947, 200-episode cap, 512-path count and flow
seed 942287 were frozen before the final three-seed check; see the
[one-step protocol](verification_v18/auction_repair/one_step_candidate/protocol.json).
Seed 947 had not previously trained a candidate. Seeds 853 and 929 were known
development seeds. The further 881/MSFT checks are additional diagnostics.

On the frozen set of 512 validation paths, the three primary synthetic checks
and the additional original-seed check give these direct auction contributions:

| Training seed | Auction value (bps) | Conditional path 95% interval (bps) |
|---|---:|---:|
| 853 | +0.554 | [0.288, 0.820] |
| 881, additional diagnostic | -0.148 | [-0.334, 0.038] |
| 929 | +0.437 | [0.260, 0.614] |
| 947, previously unused | +0.850 | [0.618, 1.081] |

The four-policy mean is +0.423 bps. Seed 881 remains negative; its interval
includes zero, which neither proves a zero effect nor makes the estimate
positive. The bounded mean changed sign, providing evidence of improvement;
a population-level auction benefit still requires the ten-seed experiment.
Residual Q-advantage overestimation also remains measurable.

For seeds 853 and 929, checking the rejected three-step candidate on **the same**
512 paths gives -0.117 and -0.220 bps. Switching to the one-step candidate
raises those figures to +0.554 and +0.437. The MSFT development check has
auction value +0.649 bps (128 paths, SE 0.150) and economic objective 3.889 bps,
versus 1.681 for AS and -1.751 for TWAP. The original seed 881's 128-path
check has objective 7.964 bps versus AS 3.984 and TWAP -0.688, while retaining
its slightly negative auction contribution (-0.047, SE 0.107).

An additional attribution check holds each **old CLOB controller fixed** and
replaces only its auction controller with the one-step candidate's controller.
The identical frozen normalizers, CLOB cash and CLOB executed quantities are
checked explicitly. Auction value changes from -0.117 to +0.211 bps for seed
853 and from -0.220 to +0.378 for seed 929. The paired improvements are +0.327
(SE 0.093) and +0.597 (SE 0.104), respectively. Thus the changed auction
controller improves outcomes even at the old CLOB policy's inventory exposure.
These are attribution diagnostics only; no phase-spliced policy is selected
or substituted into the production experiment. See
[the matched transfer records](verification_v18/auction_repair/auction_transfers.csv).

These intervals describe path uncertainty conditional on each trained policy;
they are **not** intervals across independent training seeds. The source data
are preserved in [the path checks](verification_v18/auction_repair/path_checks.csv)
and [pathwise records](verification_v18/auction_repair/paths).
The active initial notional is `100*100=10000`, so one raw cash unit in these
diagnostic CSVs equals one basis point of initial notional.

The corresponding 128-path checks had economic objectives 2.953, 3.807 and
3.796 bps, exceeding both matched reference policies for those three seeds.
On the larger 512-path cohort, objectives were -0.307, +0.812 and +0.410 bps
(path SEs 1.433, 1.344 and 1.460). Net PnL was positive in all three cases.
This difference illustrates substantial market-path variation; it must not
be hidden by reporting only the favorable 128-path cohort. Resolving the
auction failure does not establish positive economic value or benchmark
superiority for every training seed and every path cohort.

All 590 non-slow, non-network tests pass. This includes the objective contract,
all four warm-up projections, exact masked exploration, reference gradients,
checkpoint compatibility and publication pipeline. The deterministic-bandit
test additionally verifies actual optimizer updates; its old fixture had
silently remained in the new warm-up and was corrected.

The final one-step DQN completed 60,000 additional CLOB and 20,000 additional
auction gradient updates on fixed saved replay. Every recorded loss, gradient
and Q diagnostic was finite. Maximum absolute Q was 58.05 (CLOB) and 64.04
(auction); maximum recorded MSE was 4.20 and 4.13. Its 48-path economic score
declined from +0.195 to -4.294 bps (endpoint path SE 4.522). This deterioration
is retained: the stress supports numerical stability but not stable policy
performance under repeated fitting to fixed data. The saved learning-rate
episode was 127, reflecting the diagnostic helper's evaluation counter, not
episode 199 or a production end-of-training rate. Production evaluation does
not reset the training episode counter.

Further diagnostics and numerical stress results are recorded in the same
audit directory. The repeated fixed-replay stress is a numerical check,
not evidence of monotone out-of-sample performance. Even finite stable
critics can overfit a fixed replay buffer. Economic checkpoint selection
remains necessary. The full ten-seed experiment is still required for
publication-level algorithm comparisons, and the previously inspected
historical test week remains a reused historical holdout.

## Reproduction and manuscript changes

The active DQN configuration is [configs/algo/dqn.yaml](../configs/algo/dqn.yaml).
The continuous learners retain native SB3 algorithms, architectures, rates and
noise settings; only their common warm-up projection changes in this follow-up.
Lambda, k-star, q, beta, liquidity, price generation, clearing, fee schedules
and benchmarks are unchanged from the preceding v18 calibration.

## Quick follow-up on the remaining negative seed

For the selected seed-881 policy on the same 512 paths, direct auction value is
`-0.151497 + 0.052934 - 0.049502 = -0.148064` bps: adverse price edge plus
squared-inventory risk relief, less fees. Cancellation fees are secondary.
The 322 paths entering with at most one inventory unit lose 0.527 bps on
average (conditional SE 0.095); mean absolute inventory grows by 2.279 units.
The controller sometimes creates unwanted long or short exposure after the
CLOB has nearly liquidated the parent. All four strata above one unit have
positive mean contributions. This is a residual policy-learning weakness,
not evidence that the accounting or reward split is wrong.

The already-saved episode-200 auction controller was compared with the selected
episode-150 auction controller, holding the episode-150 CLOB policy, normalizer
and all 512 paths fixed. Auction contribution improves from -0.148 to +0.085 bps;
the paired improvement is +0.233 bps (SE 0.077). This required no additional
training. It supports learning progress in the auction controller but does not
establish a better jointly trained policy: the original joint validation scores
were 8.413 at episode 150 and 8.282 at episode 200. The production rule selects
the joint economic objective, not auction contribution in isolation.

No production checkpoint, minimum-update threshold, parameter, reward, or
admissibility rule was changed on the basis of this one-seed diagnostic.
A low-inventory trading prohibition would restrict the paper's signed auction
controls, and selecting a replacement using these diagnostic paths would
weaken the experimental protocol. The identified implementation/numerical
failures have bounded verification; universal economic performance and full
generalization remain empirical questions for the ten-seed campaign.
The [stratified evidence](verification_v18/auction_repair/remaining_seed881)
and [later-controller comparison](verification_v18/auction_repair/transfers/late_auction881)
are retained. A launch dry-run confirms 440 jobs, ten workers and one numerical
thread per worker, with automatic report generation after successful completion.

The complete numerical audit can be regenerated with
`python scripts/summarize_dqn_auction_repair.py`. It includes every attempt,
confirmation path, terminal-action check, stress result and artifact hash.
Raw checkpoints and replay buffers remain under `results/_auction_repair_v18`.
New diagnostics use fresh directories and do not overwrite previous results.

Both protected files remain untouched. The exact
[unapplied manuscript patch](manuscript_recommendations_v18.patch) now also
recommends the phase-specific Q formula, the auction exploration mixture and
one-step replay. It preserves the earlier required weighted-J and calibration
amendments. Regenerate it with `python scripts/recommend_v18_manuscript.py`;
`git apply --check docs/manuscript_recommendations_v18.patch` only validates
applicability and does not modify either TeX file.
