# Manuscript changes required for the unified minute-clock simulator

`paper/main.tex` is intentionally not modified. These are author-facing edits
to apply after the new experiments have been accepted.

## Physical clock

1. **Market Model / opening chronology (currently near line 82).** State that
   numerical time is measured in minutes in both experiments. With
   `tau_op=120` and `tau_cl=150`, the CLOB lasts two hours and the auction lasts
   30 minutes. The deterministic auction spacing of one unit is one minute.
2. **Timing definition (currently near line 287).** Keep the formulas unchanged,
   but explicitly say that `t_i`, `tau_op`, and `tau_cl` are minute-valued
   physical times and that `1+floor(t_i)` enforces at least one minute between
   CLOB decisions.
3. **Generative algorithm and parameter discussion (currently near lines
   660--773).** Give `lambda_0` units of arrivals per side per minute. Describe
   `p_1,...,p_4` as probabilities per one-minute auction decision, not merely
   “per step.”
4. **AS and TWAP benchmarks.** State that `T-t` is measured in minutes, so the
   AS intensity parameter `A` is per minute and the estimated volatility is per
   square-root minute. The implemented headline has `gamma=0`, so the reported
   AS quotes do not depend on the volatility estimate, but its unit should still
   be recorded correctly.
5. **Section 6 introduction (currently near line 644).** Replace the distinction
   between “120 one-second steps plus 30 seconds” and historical minutes. Both
   settings now represent a 120-minute CLOB followed by a 30-minute auction and
   inherit exactly the same CLOB, auction, action, reward, and learning
   parameters; only the exogenous mid-price source differs.
6. **Rough-Heston subsection (currently near lines 789 and 802).** Replace
   “2 minute continuous phase, followed by a 30 second closing auction” with
   “2 hour continuous phase, followed by a 30 minute closing auction.” Replace
   `s^star=252*6.5*3,600` by `s^star=252*6.5*60=98,280` trading minutes per year
   and say explicitly that `bar(t_i)=t_i/s^star` converts minute-valued simulator
   time to trading years. The annualized parameters `H`, `rho`, `V_0`, `theta`,
   `varsigma`, and `nu` are not multiplied by 60.
7. **Figures.** Time axes and captions should say “minutes.” Regenerate them
   from the minute-clock result namespace; do not relabel old second-clock
   figures.

## Other outstanding consistency edits

These were deliberately removed from `main.tex` at the author's request, but
the manuscript will need them before it describes the current code:

- Explain that historical paths use true bid/ask midquotes, selected without
  look-ahead, with invalid/crossed/zero-size/stale quotes rejected. The source
  artifact is raw USD; rebasing to `S0=100` is an in-environment coordinate
  transform after the split is selected.
- Clarify cancellation: cancel-all is conditionally available at every auction
  decision after a prior positive-slope agent order is live. It is unavailable
  at auction open only because there is no prior order. In single-replace mode,
  replacement cancels prior schedules and the current schedule survives.
- State the reward split explicitly: the headline policy is trained with
  shaping (`q=1`) and exact cancellation clawback, whereas validation,
  checkpoint selection, and final learned/AS/TWAP comparisons use only the
  economic objective. The centering adjustment is retained in both; over a
  complete episode it is the policy-independent constant `-S_0 I_0`.
- State that the untrained policy and pre-maturity validations are diagnostics,
  not reportable candidates. Checkpoint eligibility requires at least 5,000
  CLOB and 2,000 auction maturity-counted updates; for DQN, auction updates
  before the full auction action set unlocks at episode 200 do not count.
  Early-stopping patience starts at the first eligible validation, and the two
  phase networks are then selected jointly on economic validation performance.
  The initial policy's score is retained only as a safety floor: if no mature
  candidate beats it, report selection failure rather than the initial or a
  collapsed mature policy.
- Augment the latent auction configuration by the outstanding per-order
  shaping-credit ledger (the reduced feature vector need not change). If
  `phi_s=x_s+f^a(x_s)` with
  `x_s=K_s^a H_s^cl(H_s^cl-S_s^a)`, replace the interim formula by
  `r_t=phi_t-d_t c_t-c_t sum_{s in L_t} phi_s`, where `L_t` is the set of
  schedules live immediately before the current action. State that the sum
  uses the original submission-time credits, and that the current replacement
  is inserted only after the cancellation.
- Replace blanket claims of benchmark superiority with held-out paired,
  multi-seed evidence generated under the current environment contract.

## Parameter tables

The generative table should use one shared simulator column for both settings.
Only the mid-price subsection should split into rough-Heston and historical SIP
inputs. Update the shared rows as follows:

| Parameter | Current manuscript table | Shared implemented value | Required presentation change |
|---|---:|---:|---|
| time unit | absent | minutes | add explicitly |
| `tau_op` | 120 | 120 min | number unchanged; add unit |
| `tau_cl` | 150 | 150 min | number unchanged; add unit |
| `lambda_0` | 1 | 1/min/side | add units |
| market-order scale/shape `(v_m,gamma_m)` | manuscript values | `(2,2.5)` | retain and state that `gamma_m` is the order-size tail exponent |
| `V_inf` | 15 | 2 | replace |
| CLOB depth persistence `rho_lob` | 0.5 | 0.96 | replace |
| exogenous book depth | 12/conflated | 200 | separate from the strategic quote bound |
| agent CLOB quote bound | conflated with depth | 12 | label separately |
| exogenous auction support `B_inf` | absent | 150 | add; set `M_1=-B_inf=-150`, `M_2=B_inf=150` |
| auction validity floor `D_mu` | absent | 0.1 | add |
| exogenous slope bounds `(U_1,U_2)` | manuscript values | `(0.1,2.0)` | use these shared values |
| proposal probabilities `(p_1,p_2,p_3,p_4)` | stale values | `(1.0,0.0,0.3,0.05)` per minute | replace and explain persistent MM liquidity |
| ambient absolute strategic offset `B_max` | 25 | 150 | executed `b_t^a` remains relative to the frozen mid |
| local policy-template half-width | absent | 10 | add as a numerical parameterization, not as the definition of `b_t^a` |
| auction slope step `beta` | 10/3 | 1 | replace |
| auction slope index bound | 10 | 32 | policy subset is `{1,2,4,8,16,32}` |
| DQN auction templates | absent/legacy | 254 with cancellation; 127 without | add computational row |
| inventory penalty | 0.5 | 2.0 | replace |
| wrong-side shaping `q` | 1 | 1 in headline training | evaluation turns shaping off, not `q` |
| cancellation cost `d` | 0.1 | 0.1 | unchanged |
| cancellation shaping clawback | absent/no clawback | exact reversal of canceled orders' original `phi_s` | add to headline reward definition |
| projected-price initialization `H_0` | absent | 100 | add |
| projected-price smoothing `eta_H` | absent | 0.95 | add; code uses `H_i=(1-eta_H)H_{i-1}+eta_H tilde S_i` |
| order mode | absent | single cancel-and-replace | add implementation row |

The numerical section can use this compact implementation remark:

> For tractability, the DQN uses slopes
> $\beta\{1,2,4,8,16,32\}$ and 21 local templates
> $\ell_t^a\in\{-10,\ldots,10\}$ around the lagged indicative price. The
> executed manuscript coordinate is
> $b_t^a=\lfloor(H_t^{\mathrm{cl}}-S_{\tau^{\mathrm{op}}}^{\mathrm{mid}})/\alpha+1/2\rfloor+\ell_t^a$
> inside the ambient bound $[-B_{\max},B_{\max}]$, where $B_{\max}=150$, and
> $S_t^a=S_{\tau^{\mathrm{op}}}^{\mathrm{mid}}+\alpha b_t^a$. Together with
> the cancellation flag and the two zero-slope
> no-order actions, this gives $2+6\times21\times2=254$ templates. Under
> `single_replace`, a new positive-slope schedule cancels and replaces any
> earlier live agent schedule; cancel-all remains admissible whenever such a
> prior schedule exists.

For DDPG, TD3, and SAC, the raw proposal coordinate is continuous but the local
auction template is also half-up rounded to the same 21 integers before it is
translated into absolute `b`. Their executed slope support is all 33 levels
`0,...,32`, rather than DQN's six positive slope levels. This belongs in the
continuous-control implementation paragraph so “continuous” is not mistaken
for an unrounded exchange action.

The manuscript's formal action-grid paragraph remains correct: `B_max=150` is
the absolute bound on the strategic frozen-mid coordinate `b_t^a`. `B_inf=150`
has a different role: it defines the exogenous quote bounds `M_1=-B_inf` and
`M_2=B_inf`; the equality is numerical, not definitional. The numerical-policy
paragraph must additionally disclose the state-dependent local map above. It
must not describe the 21 network outputs as the entire ambient action set.

The reason for this fallback is empirical and representational. In 500
policy-free episodes, a fixed absolute `[-10,10]` grid failed to cover the
indicative center in 73.0% of synthetic and 69.5% of historical-MSFT auction
states. In matched seed-42 bounded DQN runs, indicative centering improved
held-out mean risk-adjusted PnL by 70.81 (synthetic) and 71.38 (historical
MSFT) relative to that absolute grid. Neither bounded run beat AS or TWAP;
these diagnostics justify the parameterization only, not a performance claim.

Crucial empirical qualification: the earlier unscaled, non-clawed-back reward
could be gamed by repeated replacements. The headline now reverses the exact
fictive credit of every canceled schedule. Pathwise, the undiscounted auction
sum therefore contains only credits of schedules still live at clearing,
minus cancellation fees; repeated replacement cannot accumulate old credits.
This is a changed training objective, so every numerical result must be
regenerated. `q=1` still neutralizes purchase-side terminal cash in the
training target, and the current evidence does not by itself support a blanket
claim that the shaped headline beats AS. An additional shaping weight would be
a separate model parameter and must not be introduced silently.

In the AS calibration paragraph, replace `k=alpha K` by
`k=gamma_m K` (or use a new symbol for the Pareto tail exponent). In
Avellaneda--Stoikov, the factor multiplying the impact coefficient is the
power-law order-size tail exponent. In this code that exponent is `gamma_m`;
`alpha` denotes the price tick and only converts price-unit quote distances to
ticks.

The rough-Heston table should be updated as follows:

- add `s^star = 98,280` trading minutes/year;
- change `theta` from the stale table value `0.04` to the configured and textual
  value `0.02`;
- describe `theta` as the variance-drift level in
  `theta - varsigma V_t`. Under that parameterization the long-run variance is
  `theta/varsigma`, so calling `theta` itself the long-run variance is incorrect;
- retain `H=0.1`, `rho=-0.7`, `V_0=0.02`, `varsigma=0.3`, and `nu=0.3`.

The RL hyperparameter table should likewise have one value per algorithm, not
separate synthetic/historical columns. All four learners use `2x128` hidden
layers and CLOB/auction replay warm-ups of `2,000`/`512`. DQN confirmation uses
epsilon warm-up/decay `50/600`, auction actions unlocked at episode 200, and an
800-episode budget in both settings. Add checkpoint maturity thresholds
`5,000/2,000` (CLOB/auction) and state that DQN's pre-unlock auction updates do
not count. Report only results regenerated under the revision-v9 contract.
