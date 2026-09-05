> This proposal was subsequently authorized. The active weight is .0001; use [the v16 review patch](manuscript_recommendations_v16.patch), which names it omega_sh to avoid collision with both cancellation indicators and the DQN action bias. The text below records the earlier proposal stage.

# Proposed correction requiring an explicit model decision

The [exact unapplied TeX patch](proposed_auction_weight.patch) uses omega_a
for the scalar weight to avoid collision with the manuscript's theta
cancellation indicators. The theta notation below refers to that same
scalar, as do the names of the early diagnostic runs. The
[follow-up report](calibration_followup.md) records the mixed bounded results;
this proposal has not met the adoption criterion.

The active headline still uses the manuscript's literal J with auction
shaping weight 1. The option below is implemented only for separately labeled
diagnostic experiments. Neither protected TeX file has been edited.

## Why k-star, q and beta cannot by themselves align the objectives

Consider an approximately price-taking final auction sale z>0, at indicative
and clearing prices both close to S, after CLOB inventory has reached zero.
The terminal cash and negative residual mark cancel the sale's principal.
Ignoring price impact for this local argument, the economic increment is
-lambda*z^2, while the manuscript's extra submission credit is approximately
S*z. The shaped increment is therefore S*z-lambda*z^2. For sufficiently small
positive z it rewards a trade that the economic objective penalizes.

The purchase parameter q has no effect on this positive-sale counterexample;
k-star affects only CLOB rewards. Shrinking beta suppresses the amount of
exposure available, rather than removing the incentive. At S=100 and
lambda=0.5 the unconstrained local shaped optimum oversells by about 100 units.
Actual endogenous clearing and finite grids limit this amount but do not
remove the underlying reward conflict. This is a local counterexample, not
a claim that every optimal full-session policy must oversell by 100 units.

The saved feasible-policy probes in
`results/calibration_followup/audit2/records.csv` expose the same phenomenon.
With alpha=.01, beta=.1, exogenous slope support [1,20], q=0, k-star=10000
and lambda=.5, repeatedly submitting a maximum slope at offset -10 after
liquidating the CLOB inventory gives mean shaped return 915.34 and economic
objective -43.28 over 16 paired paths. A no-order auction gives economic
objective 1.66. The resulting auction fill is 9.19 units, versus opening
inventory about .64. These are diagnostic policies, not learned results.

## Exact alternative under review

Introduce a dimensionless auction-shaping weight theta, leaving actual cash,
terminal inventory risk, admissible controls, the auction mechanism and
cancellation fees unchanged:

```
u_t(theta) = theta * [K_t H_t (H_t-S_t^a) + q*(-K_t H_t (H_t-S_t^a))_+]
r_auction(theta) = u_t(theta) - cancellation_fee
                   - sum(original weighted credits of canceled live schedules)
G(theta) = S_clear*Z + S_open*I_final - lambda*I_final^2
           + theta*q*(-S_clear*Z)_+
```

The current manuscript is exactly theta=1. The proposed execution-scale
choice is theta=alpha/S0: at S0=100 and alpha=.01, theta=.0001. A fictive
one-unit sale then earns about one tick of credit, rather than an extra full
share price. The local overselling incentive becomes theta*S/(2*lambda);
it is reduced, not claimed to vanish. q=0 remains the reference preference
because real purchases are paid in full. Economic evaluation always removes
all shaping, regardless of theta.

The draft calibration is stored in
`configs/diagnostic/proposed_credit_scale.yaml`. The active base configuration
does not enable it. Any adoption must explicitly change the paper's u_t and
G definitions (near lines 402 and 409 of `paper/main.tex`) and label the
objective J_theta. It is not merely numerical conditioning of the existing J.
The inventory/price potential and learning-rate decay, in contrast, do not
change the specified complete-return ranking.

This alternative keeps dense shaping, signed auction controls, actual
pro-rata settlement and exact cancellation clawback. It does not impose a
hard inventory bound, force auction participation or provide reference-policy
actions to learners. Whether the finite-sample algorithms succeed under it
must be assessed separately from this dimensional argument.
