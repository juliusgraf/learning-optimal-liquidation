# Referee-to-experiment audit, 6 September 2026

Read-only audit of all nine pages of `report_X_on_v1-2.pdf` (four pages) and
`Report-2.pdf` (five pages), supplied from the user's Downloads directory.
The reports concern earlier manuscript versions. Their proposed changes are
recommendations to assess, not instructions to change the current model.
No training or protected manuscript edits were performed for this audit.

## Decision

The existing campaign covers the principal economic, information, auction-access,
and learning-credit questions. It does not cover every referee suggestion.
The most relevant missing control is a learned CLOB policy followed by the same
static auction rule used by AS. Add this comparison before claiming that the
benchmark advantage survives equal auction rules. It does not require repeating
the 440-run campaign.

## Verified production matrix

Counts below come from the 440 saved `config_resolved.yaml` files, not merely
from configurations available to launch. Each synthetic cell has four algorithms
and ten training seeds. There are 100 held-out evaluation episodes per policy.

| Label used below | Saved treatment | Training preferences | H feature | Auction anchor | Auction credit | Runs |
|---|---|---|---|---|---|---:|
| A | headline | weighted manuscript J | on | indicative H | on | 40 |
| B | mechanism_fixed_anchor | weighted manuscript J | on | frozen mid | on | 40 |
| C | mechanism_economic_dense | economic | on | frozen mid | on | 40 |
| D | mechanism_economic_sparse | economic | on | frozen mid | off | 40 |
| E | mechanism_h_feature_off | economic | off | frozen mid | on | 40 |
| F | no_auction | economic | off | inactive | inactive | 40 |

Another 200 runs are headline policies for five historical-midprice settings.
The five matched contrasts are C-D (auction credit), C-E (H feature), A-B
(anchoring), B-C (combined preferences), and E-F (auction access). All policies
are selected and evaluated on the economic objective. Mechanism ablations are
synthetic only; headline and direct settlement decompositions cover all markets.

## Concrete interpretation

H is an engineered forecast of the eventual simulated clearing price. It can
help a policy decide whether to sell now or retain inventory, and how to quote,
size and cancel auction orders. Prediction accuracy improves over contemporaneous
midprice in the recorded data; incremental economic gains from the feature are
clear for DDPG, uncertain for the other methods.

Anchoring places the limited auction quote range around H. This could make a
small action range track the anticipated clearing price, but also changes the
available executable orders. The indicative anchor lowers mean synthetic economic
performance for all four algorithms relative to a frozen-mid anchor; the DQN,
DDPG and TD3 pointwise intervals exclude zero. It has an intended design benefit,
not a demonstrated economic advantage in these runs.

The manuscript preferences discourage CLOB sales below H and provide fictive
auction submission rewards, with cancellation clawbacks. Their intended purpose
is to guide behavior using auction knowledge. They are not economic cash flows
and can change policy rankings. The combined-preference effect is not clearly
positive for any algorithm in production.

The separate terminal-corrected potential supplies dense auction execution/risk
credit without changing the episode objective. C-D demonstrates economic gains
for all four algorithms. This supports the delayed-credit motivation, not a claim
that the specific manuscript preference terms improve economic performance.

## Mapping the reports

| Report concern | Current evidence | Remaining work or claim restriction |
|---|---|---|
| X, comment 1, pp. 1-2: fictive rewards mistaken for PnL | All 440 evaluations use economic cash/mark/risk accounting; B-C tests extra preferences | Explain J versus economic PnL and risk-adjusted objective; disclose corrected reward formula, weight and clawbacks |
| X, comment 5, pp. 3-4: same learner with no auction, auction without signal, auction with signal without interim preferences | F, E, C respectively, with matched economic training | Describe potential credit separately; C still has objective-preserving credit, D removes its auction component |
| Report 2, section 1, p. 2: decompose CLOB and auction contributions | Existing same-CLOB no-order economic decomposition; separately retrained E-F | These are not the same-AS-auction experiment and do not isolate H's causal effect during CLOB |
| Report 2, section 1, p. 2: constrain RL to AS's auction rule | Not in the production matrix or current decomposition | Add hybrid-policy evaluation; retrain under the fixed rule if claiming adaptation to that constraint |
| Report 2, section 3, pp. 4-5: independent training repetitions and fixed-policy evaluation distributions | Ten training seeds and 100 additional held-out episodes per policy; seed-level intervals and raw episode records | Explain train/validation/test separation and avoid relying on illustrative single paths; literal 1,000 episodes per trained policy was not run |
| X, comment 2, p. 2: overselling, model identity, benchmark inventory cap | Benchmark auction schedule is now capped; learned auction settlement remains signed, with terminal inventory penalty and short-position diagnostics | Explicitly describe sell-only CLOB plus signed auction control; do not claim a strictly inventory-constrained liquidation model |
| X, comment 3, p. 2: theoretical state versus learner state | Common 18-coordinate feature representation is implemented | Reconcile manuscript representation and theoretical claims; common features alone do not prove Markov sufficiency |
| X, comments 4-5, p. 3: chronology, historical experiment, clearing theorem and regret claims | Numerical runs do not resolve these writing/theory questions | Describe sequential stylized phases and historical midprice input accurately; audit theorem assumptions and remove unsupported regret claims |
| Report 2, sections 1-3, pp. 2-4: execution priority, exogenous competition, information observability, order-flow assumptions, adverse selection | Current mechanism runs hold the simulator fixed | Explain and delimit the model; use separate realism/robustness studies only if making broader deployment claims |

## The user's proposed controls

**No access to H:** E zeros the explicit H observation, fixes the auction anchor,
and removes H-dependent manuscript rewards. It retains other book moments and
auction-exposure conditioning, which convey clearing-relevant information.
Consequently C-E isolates the incremental explicit H feature. It is not a market
with no information about future clearing. A-E is an available total package
comparison, but bundles forecast access, anchoring and preferences.

Merely zeroing the observation while retaining headline rewards is not complete
removal of H: the rewards remain an H-derived training channel. The older
`ablation_h_off_shaping_on.yaml` explicitly documents this and is not a completed
v19 production cell. A clean headline-based information experiment needs a precise
definition of what replaces H in every dependent component; it cannot preserve
the exact H-dependent J while also removing every use of H during training.

**No auction, session ends at CLOB:** F does exactly this. At tau_op it terminates
and marks residual inventory at the final CLOB midprice, subtracting lambda I^2.
It does not force remaining units to execute at a market-order liquidation cost.
E-F isolates access under matched economic training and H-off information.
A-F is also computable, but changes several mechanisms and is a package effect.

## Focused additional comparison

For a learned CLOB policy pi_C, learned auction policy pi_A, and the benchmark's
static auction rule b_A, evaluate the frozen hybrid (pi_C, b_A) on the same saved
test seeds. The rule uses the hybrid's own execution history and remaining
inventory, with the same quantity cap and parameters as the benchmark. Recompute
auction clearing, fills and residual risk; replacing a number in existing records
would not implement this intervention.

With V denoting economic performance, the identity

V(pi_C, pi_A) - V(b_C, b_A)
= [V(pi_C, pi_A) - V(pi_C, b_A)]
+ [V(pi_C, b_A) - V(b_C, b_A)]

separates the incremental learned auction rule conditional on the learned CLOB
policy from the CLOB-policy difference under a shared auction rule. This frozen
hybrid evaluation needs additional rollouts, not additional training. It is a
post-training intervention and may disadvantage a CLOB policy trained to expect
dynamic auction control.

If the claim is about how the CLOB policy learns when anticipating a constrained
auction, retrain that control with b_A imposed throughout training and evaluate
economically. A balanced synthetic extension is 40 runs (four algorithms times
ten seeds), with its objective, information, budget and selection protocol matched
to a declared existing comparator. This is distinct from the cheaper frozen-policy
test. Neither comparison alone attributes a gain specifically to H.

Do not reinstate all omitted arms automatically. CLOB-only and auction-only
manuscript preference arms were not run in production; they are needed only to
claim individual term effects. Likewise, no raw-versus-calibrated-H production
arm or historical mechanism matrix exists. Narrow the corresponding claims or
declare a focused follow-up, rather than presenting development probes as final
balanced evidence.

## Write-up recommendation

Retain the economic headline table and five existing synthetic contrasts in one
compact treatment figure. Use the economic dense/H-off/no-auction sequence for
the first referee's comparison, and the dense-versus-sparse learning figure for
the original delayed-credit motivation. Put quote anchoring and combined
preferences in the same figure even though their results are negative/uncertain.
Report direct auction cash edge, fees and inventory-risk relief separately.
Add the shared-static-auction comparison to address the second referee's most
specific missing empirical control. Broader model/theory objections require an
accurate scope and manuscript revision, not just more ablation runs.
