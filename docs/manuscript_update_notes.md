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
- Describe the centered economic training reward used by both settings. Its
  complete-episode adjustment is the policy-independent
  constant `-S_0 I_0`; shaping remains a treatment rather than the headline
  objective.
- Replace blanket claims of benchmark superiority with held-out paired,
  multi-seed evidence. The legacy single-asset pilot is diagnostic only.

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
| `V_inf` | 15 | 2 | replace |
| CLOB depth persistence `rho_lob` | 0.5 | 0.96 | replace |
| exogenous book depth | 12/conflated | 200 | separate from the strategic quote bound |
| agent CLOB quote bound | conflated with depth | 12 | label separately |
| ambient auction `B_max` | 25 | 150 | replace |
| auction slope step `beta` | 10/3 | 1 | replace |
| auction slope index bound | 10 | 32 | policy subset is `{1,2,4,8,16,32}` |
| local auction offset | absent | `+/-10` ticks around indicative price | add policy-parametrization row |
| DQN auction templates | absent/legacy | 254 with cancellation; 127 without | add computational row |
| inventory penalty | 0.5 | 2.0 | replace |
| wrong-side shaping `q` | 1 | 0/inactive in headline | identify shaping treatment |
| cancellation cost `d` | 0.1 | 0.1 | unchanged |
| order mode | absent | single cancel-and-replace | add implementation row |

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
800-episode budget in both settings. The bounded pilot values (`20/150`, unlock
at 100) must not be reported as confirmation-run hyperparameters.

Do not claim from the bounded synthetic diagnostic that trained DQN beats AS:
the shared calibration improved the paired DQN-AS edge from -70.41 to -2.98
and beat TWAP by 14.03 per episode, but validation selected the safe initial
policy even after a 300-episode follow-up. Use the held-out multi-seed
800-episode runs for any superiority statement.
