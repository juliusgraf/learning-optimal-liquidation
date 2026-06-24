# Output schemas (Phase 4)

Every run writes `results/<experiment_name>/<run_name>/` (CLAUDE.md
engineering conventions). All floats are written with `%.17g` (repr-exact),
so byte-identical files certify bit-identical runs (ruling D10). Missing
values (e.g. losses before `min_buffer` is reached, `eval_return_mean` off
the eval cadence, NaN placeholders) are empty cells.

## Discounted-return convention

`return_disc` = Σ_steps χ^{t} · r_t + χ^{τ_cl} · r_terminal, where t is the
DECISION TIME on the paper grid (CLOB decision times are real-valued and
their count varies per episode; auction times are the integers τ_op..m) and
the terminal reward — although the env returns it combined with the final
step reward — is re-discounted at χ^{τ_cl} in the reported value. This is
consistent with the Bellman target, which bootstraps r_terminal at one χ from
t_m (item 1, no fold; see `docs/rl_design.md` §5). `return_undisc` is the
plain episode sum and is the headline evaluation number (CLAUDE.md).

## metrics.csv (train.py; one row per training episode)

| column | meaning |
|---|---|
| episode | 0-based training episode index |
| env_seed | the episode's env seed (drawn from the `env_train` stream) |
| epsilon | exploration rate used this episode |
| return_undisc | undiscounted episode return (training policy) |
| return_disc | discounted return per the convention above |
| clob_reward_sum | Σ of CLOB-phase step rewards |
| auction_step_reward_sum | Σ of auction step rewards EXCLUDING the terminal part |
| terminal_reward | r_{τ_cl} (clearing + inventory penalty + wrong-side terms) |
| S_cl | terminal clearing price (corrected Eq. (1)) |
| Z_tau_cl | terminal auction execution Z_{τ_cl} |
| I_final | I_{τ_cl} = I_{τ_op} − Z_{τ_cl} (no clipping, D8) |
| H_at_tau_op | the H_cl cache at the first auction decision (Algorithm 1's last CLOB output) |
| cancel_count | number of auction steps with c_t = 1 |
| n_steps | decisions taken (CLOB steps + 30 auction steps) |
| n_clob_steps | CLOB decisions (varies; Assumption assump:presence grid) |
| n_degenerate_fallbacks | D17 fallbacks in the episode's Eq. (2)/(1) solves |
| loss_{clob,auction} | mean minibatch loss over the episode's gradient steps |
| grad_norm_{clob,auction} | mean pre-clip global gradient norm |
| td_abs_mean_{clob,auction} | mean over updates of mean abs TD error |
| td_abs_max_{clob,auction} | max over updates of max abs TD error |
| n_grad_steps_{clob,auction} | gradient steps taken this episode |
| buffer_{clob,auction} | replay sizes at episode end |
| eval_return_mean | mean undiscounted greedy return on the fixed `env_eval` seed list (eval episodes only) |
| wall_clock_s | episode wall time — LAST column, EXCLUDED from determinism comparisons |

### Continuous agents (DDPG/TD3/SAC) — same schema, remapped columns

The continuous-action variants (Phase 5; `docs/continuous_action_extension.md`)
use the **same** `metrics.csv` columns. The loss/TD columns hold **critic**
statistics: `loss_{phase}` is the mean critic loss (mean over the twin critics
for TD3/SAC), `grad_norm_{phase}` the critic global gradient norm, and
`td_abs_*_{phase}` the critic TD errors. The `epsilon` column holds the
exploration-noise scale (`exploration_noise_std`; `0` for SAC, whose policy is
intrinsically stochastic). Actor loss, the SAC temperature `alpha`, and the
policy entropy are emitted to the per-update diagnostics and `logs/run.log` but
are not written to `metrics.csv` (the CSV writer ignores the extra keys, so the
DQN schema is byte-for-byte unchanged).

## eval/records.csv (evaluate.py; one row per (policy, episode))

Columns: `policy` (dqn | initial | as | twap), `episode`, `env_seed` (shared
across policies within an episode — CRN), then the return decomposition
exactly as in metrics.csv: `return_undisc`, `return_disc`, `clob_reward_sum`,
`auction_step_reward_sum`, `terminal_reward`, `S_cl`, `Z_tau_cl`, `I_final`,
`H_at_tau_op`, `cancel_count`, `n_steps`, `n_clob_steps`,
`n_degenerate_fallbacks`.

## eval/metadata.yaml

`master_seed`, `checkpoint`, `early_stopping`, `n_episodes`, `policies`, the
CRN statement, the return-convention statement, `reward_params_shared_by_all_policies`
(the single RewardParams applied to every policy — AUDIT C.4) and
`as_calibration` (A, k, sigma). `checkpoint` is the resolved path of the
evaluated learned-policy snapshot; `early_stopping` is `true` when that is
`best.pt` (the default — see "Reported checkpoint" below).

### Reported checkpoint (early stopping)

`evaluate.py` evaluates the learned policy from `best.pt` by **default**
(`early_stopping: true`): the best-VALIDATION checkpoint, selected by `train.py`
on the `env_eval` seed stream, which is **disjoint** from the `env_final_eval`
test stream used for these records. This is standard model selection, not
test-set cherry-picking, and it discards training episodes that *degraded* the
policy (e.g. TD3's late collapse). Pass `--checkpoint final` for the
last-episode model. The full eval trajectory is in `metrics.csv`
(`eval_return_mean`) / the `training_diagnostics` figure, so instability remains
visible.

## eval/regret_<benchmark>.csv (regret.py)

Columns: `episode`, `env_seed`, `v_benchmark`, `v_policy`, `regret`,
`cum_regret`; the final `cum_regret` is PRegret(T) with T = (m+2)E printed to
stdout. `--returns discounted` (default; the paper's V_0) selects
`return_disc`, `--returns undiscounted` selects `return_undisc`.

## checkpoints/ (train.py)

- `initial.pt` — untrained networks, saved before training (the
  "initial-DQN" baseline).
- `best.pt` — best periodic-eval mean return so far.
- `final.pt` — end of training.
- `ckpt_ep{N}.pt` + `ckpt_ep{N}_trainstate.pt` — resumable pair: the agent
  checkpoint includes replay contents and RNG states; the sidecar holds the
  training loop's episode counter, best-eval value and the `env_train`
  seed-stream state. `lmm-train --resume <ckpt_ep{N}.pt>` restores both.

Agent checkpoints contain: hyperparams, episode/env-step counters, both
Q-networks and both targets, both optimizers, the exploration generator
state, the torch global RNG state, and (resumable pairs only) the full
replay buffers including their sampling-generator states.

## Cross-seed aggregates (multi-seed reporting)

`scripts/run_multiseed.sh` runs the pipeline under several master seeds, then
`scripts/make_multiseed_outputs.sh` writes per-setting aggregates to
`results/<setting>/_multiseed/{tables,figures}/` (built via `make_tables
--multiseed` / `make_figures --multiseed`). Each **seed contributes one number
per policy** (its 100-episode mean return); these are aggregated across seeds
with the **IQM** (interquartile mean; `scipy.stats.trim_mean(·, 0.25)`) and a
percentile-**bootstrap 95% CI** over seeds (Agarwal et al. 2021, `rliable`).

| Artifact | What |
|---|---|
| `tables/eval_summary_multiseed.{tex,csv}` | synthetic: per-algo IQM [95% CI], mean-of-seed-means, seed count, IQM improvement vs AS/TWAP |
| `tables/dqn_results_multiseed.{tex,csv}` | historical: per-ticker IQM across seeds; final row pools all ticker×seed runs into IQM [95% CI] |
| `figures/algorithm_comparison_multiseed.{pdf,png}` | per-algo IQM bars with bootstrap-CI whiskers, AS/TWAP IQM reference lines |

With few seeds (e.g. 3) the CIs are wide and IQM ≈ mean (no trimming below n=4)
— the honest multi-seed signal, not a defect.
