# V19: auction credit, forecast reliability and DDPG regression

The repair preserves the author-approved weighted shaped headline objective
`J_omega` (`omega=.0001`, `q=0`, both shaping terms enabled) and economic
validation/evaluation on `Jbar_lambda`. It fixes a confounded treatment design,
calibrates the forecast's CLOB reliability using training paths, and stabilizes
DDPG's critic without changing native library update rules. It does **not**
make the shaped objective economically equivalent or turn every treatment
and auction contribution positive. The author explicitly chose to retain
those preferences after the zero-inventory fictive-reward issue was explained.

All 43 bounded training attempts are retained, totaling 7,600 episodes, with
at most 200 per run. No 800-episode production run or new final-test evaluation
was launched. These are adaptive development checks, not ten-seed publication
results. The initially planned production matrix had 520 runs. Before launching it,
the final protocol was reduced to 440 runs by making the two individual
preference arms optional; see [the scope decision](final_run_scope_v19.md). The current model is described in [model.md](model.md).

## 1. Why the previous negative treatments did not answer the credit question

The original motivation is sound: settlement after thirty auction decisions
makes one-step temporal-difference credit propagation difficult. But v18's
“shaping off” arm still gave dense auction guidance through the replay
potential. With observed own and exogenous slope/quote moments, define

\[
\Delta=\frac{W_a-SG_a+W_e-SG_e+D}{G_a+G_e},\qquad
\widehat z=G_a\Delta-(W_a-SG_a).
\]

The auction potential was already

\[
\Phi(s)=(S-S_0)I+\Delta\widehat z-\lambda(I-\widehat z)^2,
\qquad \Phi(\text{terminal})=0.
\]

A newly submitted schedule changes projected execution and residual risk
immediately, even though actual cash settles later. Replay adds
`Phi(next)-Phi(now)`. Its episode sum is `-Phi(initial)`, zero in the rebased
headline setup, so it preserves the episodic objective. Terminal correction
is essential; removing the last correction would introduce a preference for
the final potential. This is the distinction established by
[Ng, Harada and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
and the episodic treatment in
[Grześ (2017)](https://kar.kent.ac.uk/id/document/97755).

Consequently, v18's reward comparison tested **extra non-telescoping manuscript
preferences on top of existing dense guidance**. It did not test the original
sparse-auction credit-assignment difficulty. V19 adds a switch that removes
only `Delta*z_hat-lambda*(I-z_hat)^2` from the auction potential. CLOB
conditioning and market-return centering stay fixed. With frozen midprice,
the remaining auction mark term provides no action-dependent intermediate
credit. Both versions still have terminal-zero potentials.

This gives the correct experiment, but not a theorem that potential guidance
helps finite neural-network training. In the 150-episode, seed-1931 mechanism
check, DDPG dense guidance helped early validation (episode 50: +1.21 versus
−2.95 bps), but selected independent confirmation favored sparse by .90 bps.
DQN also favored sparse confirmation by 1.24 bps. A state potential changes
value-target representation and approximation error even when optimal policy
ordering is preserved. The new raw learning comparison exposes that effect
instead of inferring useful credit from the word “shaping.”

## 2. The manuscript rewards embed preferences as well as insight

The CLOB reward deduction below the forecast is approximately

\[
\frac{Sv(H-S)_+}{k^\star\alpha}\simeq v(H-S)_+,
\]

because `S≈100` and `k_star*alpha=100`. Thus a one-unit increase in sale price
below a fixed forecast can improve the shaped reward by roughly twice its
cash improvement. This can discourage economically reasonable early sales
when the forecast is optimistic. A value of 10,000 for `k_star` is a clipping
scale expressed in ticks, not a claim that a 10,000-tick gap is realistic.
Its local preference coefficient is `S/(k_star*alpha)≈1`, which is not weak.

The fictive auction term `omega*K*H*(H-S_a)` rewards projected gross sales
without conditioning on remaining inventory. For an illustrative zero-edge,
zero-inventory sale of quantity `z`, economic value is `-lambda*z^2`, while
a local approximation to shaped value adds `omega*H*z`. Its continuous
optimum is `omega*H/(2*lambda)=.5` units at the current parameters. This is an
analytic certificate that the preferences can differ, not a quantitative
explanation of every learned loss. Actual auction impact, forecasts,
persistent schedules, rounding and later cancellation also matter.

V19 retains both terms and exact cancellation clawbacks. It does not add a
terminal rebate, inventory gate, execution clamp, benchmark fallback or
change to the economic settlement. In the matched seed-1931 fixed-anchor
check, the combined manuscript terms increased DQN confirmation by **1.44
bps**, while the DDPG effect was **−.05 bps**. The CLOB-only effects were +1.27
and +.32 bps; auction-only effects +.14 and +.33 bps. These are useful
mechanism observations from one training seed, not population conclusions.
All eight cells and both algorithms are in
[mechanism_effects.csv](verification_v19/mechanism_effects.csv) and the full
[learning ledger](verification_v19/learning_summary.csv).

## 3. Pooled H accuracy concealed a CLOB problem

V18's synthetic forecast was better near settlement but worse earlier in the
CLOB than contemporaneous midprice. Averaging both phases obscured this.
The raw estimator summarizes projected clearing from current auction inputs;
that is not automatically the best estimate of the eventual future close
when the midprice and order flow will continue to evolve.

V19 uses four equal CLOB time bins and a restricted forecast combination:

\[
H_t^{cl}=S_t^{mid}+w_{b(t)}(H_t^{raw}-S_t^{mid}),\qquad
w_b=\operatorname{clip}_{[0,1]}
\frac{\sum_{i\in b}x_i y_i}{\sum_{i\in b}x_i^2},
\]

where `x=H_raw-mid` and `y=actual_close-mid`. A zero denominator gives weight
zero. Each setting's four coefficients use 256 independent no-order
**training** paths, with no intercept or fitted drift. Historical fitting
uses the existing pooled training sessions. The coefficients are shared by
all algorithms and treatment arms and frozen before their learning. The
original raw recursion and exogenous carryover are unchanged; during the
auction, the original indicative calculation is retained. Because H enters
J, this changes the numerical forecast used by the preference, while keeping
the same reward formula and control problem.

| CLOB bin (minutes) | Synthetic weight | Historical weight |
|---|---:|---:|
| 0–30 | 0.000000 | 0.192784 |
| 30–60 | 0.190153 | 0.314044 |
| 60–90 | 0.234565 | 0.386180 |
| 90–120 | 0.414256 | 0.525631 |

On separate 256-path validation samples, averaging errors within each path
before averaging paths:

| Setting / phase | Raw H MAE | Mid MAE | Calibrated H MAE |
|---|---:|---:|---:|
| Synthetic CLOB | .21699 | .19472 | .18301 |
| Historical MSFT CLOB | .10167 | .12042 | .09566 |
| Synthetic auction | .05171 | .14762 | .05171 |
| Historical MSFT auction | .03696 | .13238 | .03696 |

Errors are price units. Synthetic CLOB MSE improves against raw H
(.16365→.12428), but is **slightly worse than mid's .12390**. Historical MSFT
CLOB MSE improves against both (.01816 raw, .02415 mid, .01603 calibrated).
These samples do not validate every stock, policy-induced state distribution
or unseen date regime. Protocols, per-path errors and summaries are in
[forecast_synthetic](verification_v19/forecast_synthetic/protocol.json) and
[forecast_historical](verification_v19/forecast_historical/protocol.json).
New evaluation summaries explicitly separate market phase and horizon.

H information and quote anchoring are now independent switches. Moving the
center of a bounded ±10-tick grid changes executable orders, not just feature
coordinates. The headline retains its indicative anchor. Fixed-mid controls
isolate information and reward effects. At seed 1931, indicative minus fixed
anchor gave −.20 bps for DQN and +.41 for DDPG; this does not justify replacing
the headline grid solely to improve one algorithm's auction result.

## 4. DDPG's late regression was critic exploitation

The v18 CAT seed 811 policy deteriorated in both shaped and economic returns,
so objective mismatch alone could not explain its regression. Its actor
learned excessive sales while its critic remained optimistic. The error
feedback mechanism is consistent with
[Fujimoto, van Hoof and Meger (2018)](https://proceedings.mlr.press/v80/fujimoto18a.html).

V19 adds LayerNorm after each hidden critic linear layer and before ReLU,
with learned affine parameters. Both critic hidden layers remain width 256,
and the scalar output remains unbounded. The online and target critic use
the same architecture. The native SB3 DDPG actor, single critic, actor update
frequency of one, absence of target smoothing, learning rates and Polyak
coefficient remain unchanged. This uses the library's policy factory, not
handwritten replacement losses or a disguised TD3 update. The exact native
DDPG construction is documented in
[SB3 2.7.1](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ddpg/ddpg.html).

Critic normalization has precedent as an extrapolation-error intervention in
[Ball et al. (2023), §4.2](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf).
That paper studies a different SAC/offline-assisted setting; efficacy in this
DDPG market-making problem is an empirical result, not a transferred theorem.

Paired 200-episode CAT-811 checks, changing only critic normalization:

| Economic validation (bps) | Original critic | LayerNorm critic |
|---|---:|---:|
| Initial | 8.95 | 8.95 |
| Episode 50 | 9.09 | 7.75 |
| Episode 100 | 8.00 | 10.56 |
| Episode 150 | 4.62 | 11.79 |
| Episode 200 | 2.80 | 12.05 |
| Selected policy, separate 128-path confirmation | 2.52 | 8.72 |
| Same-CLOB auction contribution on confirmation | −7.02 | 1.01 |

A frozen-policy audit on 64 additional validation paths compares final critic
estimates with realized conditioned returns. Mean overestimation falls from
**6.85 to .42** in the auction and **15.41 to 1.43** in the CLOB. The audit
averages within paths before averaging across paths. Its final-policy economic
score improves from −9.01 to +6.36 bps and signed terminal inventory from
−21.31 to +3.91. This directly checks the proposed failure mechanism, beyond
selecting a better point on a curve. See
[critic calibration](verification_v19/critic_calibration/summary.csv).

Independent CAT seed 1979 supports improved late stability: original
validation falls from 5.92 at episode 150 to 4.38 at 200; normalization gives
6.92→6.98. Selected confirmation improves 10.08→10.90 bps. Early transients
remain, and a synthetic seed-1907 normalization-only check does not show a
uniform learning-speed improvement. DDPG still has no monotone-convergence
guarantee. With both final changes (normalization and calibrated H), CAT-811
finishes at 10.17 validation, 7.43 confirmation and +.24 auction contribution.

## 5. Independent confirmation and remaining negative effects

The final candidate retains the indicative headline anchor. In two additional
synthetic seeds, all algorithms have positive economic confirmation scores.
The candidate modifies H for every algorithm and also the critic for DDPG;
the DDPG normalization-only trials above separately identify that intervention.

| Algorithm / seed | Old score | Candidate score | Candidate auction contribution |
|---|---:|---:|---:|
| DQN / 1951 | 3.16 | 3.93 | −.41 |
| DQN / 1973 | 3.39 | 4.83 | .04 |
| DDPG / 1951 | 4.39 | 5.54 | .33 |
| DDPG / 1973 | 4.09 | 2.87 | .83 |
| TD3 / 1951 | 3.25 | 3.54 | −.07 |
| TD3 / 1973 | 5.23 | 3.64 | .52 |
| SAC / 1951 | 3.59 | 5.08 | .57 |
| SAC / 1973 | 4.85 | 4.47 | 1.25 |

Scores and auction contributions are bps of initial notional. The two-seed
mean changes are +1.10 for DQN, −.04 for DDPG, −.65 for TD3 and +.55 for SAC.
Do not present improved forecast accuracy as uniform policy improvement.
DQN-1951 and TD3-1951 also remain slightly below AS (−.02 and −.41 bps),
although all eight are above TWAP in this confirmation sample.

Calibrating H often encourages earlier CLOB liquidation. This can improve
total economic score while reducing inventory available for auction risk
relief. Lower auction contribution is therefore not by itself evidence that
the forecast or auction has become worse. Conversely, it cannot support a
claim that the revised model exploits the auction more strongly everywhere.
The negative DQN/TD3 synthetic auction contributions remain; they are not
clipped, bypassed or removed by seed selection.

A separate 150-episode MSFT check, seed 1987, uses the final specification:

| Algorithm | Economic confirmation | Gap to AS | Gap to TWAP | Auction contribution |
|---|---:|---:|---:|---:|
| DQN | 2.29 | .96 | 5.59 | .24 |
| DDPG | 2.77 | 1.45 | 6.08 | 3.20 |
| TD3 | 3.81 | 2.49 | 7.12 | 4.82 |
| SAC | 1.92 | .59 | 5.22 | 1.33 |

All four improve on their initial economic validation and beat both references
on these common confirmation paths. This is promising transfer evidence,
especially for continuous auction control, but one stock/seed cannot establish
the paper's overall ranking. The complete manifest keeps each attempted
configuration, seed, source digest, validation curve, selection and separate
confirmation; [paired_confirmation.csv](verification_v19/paired_confirmation.csv)
contains the matched changes. No test result enters checkpoint selection.

## 6. What changes mathematically, and what does not

| Change | Mathematical effect |
|---|---|
| Four frozen forecast weights | Replaces CLOB H by a training-calibrated combination of raw H and mid; the forecast-dependent J uses that signal |
| DDPG critic LayerNorm | Changes the function class/conditioning of Q approximation; leaves native DDPG targets and optimization rule intact |
| Auction-potential ablation | Changes temporal allocation of replay credit with zero net episode adjustment; leaves the economic objective fixed |
| Independent H/anchor flags | Allows an information intervention without simultaneously moving the finite action grid |
| Separate preference controls | Identifies objective-changing CLOB and auction preferences individually and jointly |
| Phase-specific H summaries and raw paired learning plot | Corrects the estimands and reporting; does not affect execution |

No new liquidity, impact, price-volatility, risk-penalty, q, beta or k-star
value was tuned in v19. Their economic interpretation remains the v18 one:
`k_star` governs local opportunity-cost preference; `q=0` adds no purchase
subsidy; `beta=.25` gives maximum individual schedule slope 8, not a cap on
aggregate execution. Multiple signed schedules and overselling remain feasible.
The historical midprice experiment remains a stylized market around replayed
prices. Quote data do not identify auction slope or flow parameters. See
[the existing calibration audit (archived)](https://github.com/juliusgraf/Learning-Market-Making/blob/83c35645edfe2d15733f91ede4a54e1bd9660fde/docs/pathology_repair_v18.md); these values should
be presented as explicit normalized scenario choices, not exchange estimates.

## 7. Verification, publication scope and the next run

The bounded checks establish finite learning computations, corrected reward
accounting/terminal telescoping, a causal DDPG critic diagnosis, improved CLOB
forecast MAE and working mechanism-specific reporting. They do not establish
that dense guidance helps all algorithms, that J dominates economic-only
training, that continuous methods always beat DQN, or that auction contribution
is positive for every synthetic seed. Those assertions would contradict some
retained development evidence.

The defensible paper question is how auction forecasts, temporal credit and
extra economic preferences affect learned liquidation and auction use. A
strong empirical result would combine economic reference gaps with positive
auction value and evidence from the appropriate matched controls. The paper
can also explain a preference/credit tradeoff, provided it does not claim a
uniform gain that the full results fail to support. The previously inspected
historical week is still development-exposed; fresh simulation seeds alone
cannot repair that limitation. Broader historical generalization needs a
subsequently frozen date block.

V19's final production matrix uses ten seeds, 800-episode caps, economic maturity
selection, five synthetic contrasts and a new dense-minus-sparse validation
figure. A complete 72-job smoke matrix generated all five figure groups and
three tables without running the production campaign. Unit/integration tests
include native DDPG checkpoint-resume equality, potential terminal correction,
unchanged exogenous settlement under forecast adjustment, independent
information/anchor flags, and rejection of unmatched credit-comparison paths.
The final offline suite passed **595 tests**, with 23 slow/network tests
deselected and 14 expected legacy/error-path warnings. Output is retained in
[tests_final_pass.log (archived)](https://github.com/juliusgraf/Learning-Market-Making/blob/83c35645edfe2d15733f91ede4a54e1bd9660fde/docs/verification_v19/tests_final_pass.log). The smoke output is purely
operational evidence and must not be interpreted as learned performance.

See [the launch/output guide](research_outputs.md) for the exact command, the
440-run layout and where each result will appear. The source ledger below
records what external research supports; numerical claims above come from
this repository's explicitly labelled development artifacts.
