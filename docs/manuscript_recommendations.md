> Superseded review artifact. Use [the exact v17 patch](manuscript_recommendations_v17.patch) and [current report](model_refinement_v17.md). Do not combine the old alternative parameter patches below. Neither protected source has been edited.

# Exact manuscript recommendations (not applied)

The numerical substitutions below describe v15 code/paper alignment, **not
an empirically validated publication recommendation**. Read the newer
[calibration audit](calibration_followup.md) first. Its optional
[auction-weight patch](proposed_auction_weight.patch) is a separate,
mutually alternative model proposal; do not apply both parameter patches.

Both protected files remain byte-identical. The companion
[unified patch](manuscript_recommendations.patch) specifies exact replacements
and insertions against their current contents; `git apply --check` succeeds.
It is a review artifact, not an instruction to silently update the manuscript.

| Original location | Required alignment |
|---|---|
| `paper/main.tex:450` | State that headline training uses shaped J, while validation, checkpoint selection and final comparisons use bar-J-lambda; unshaped training is a treatment. |
| `paper/main.tex:452` | Disclose the revised stylized numerical calibration and its interpretation for historical midprice replay. |
| `paper/main.tex:455` | Identify the shaped control problem as the headline; add the exact relative-price normalization and telescoping training-potential equations. |
| `paper/main.tex:471` | Replace epsilon's three branches by 1 before episode 5, `1-0.95(e-5)/90` from 5 through 95, and 0.05 thereafter. |
| `paper/main.tex:476` | Replace the forced no-order DQN initialization description with the ordinary initialization used in code. |
| `paper/main.tex:518` | Describe five-step DQN replay with phase-boundary truncation and terminal G included exactly once. Continuous methods retain one-step replay. Disclose the sampled off-policy approximation. |
| `paper/main.tex:548` | Replace Huber by squared TD loss. |
| `paper/main.tex:572` | Describe Stable-Baselines3 2.7.1, uniform warm-up proposals, phase-boundary replay adapter, standard actor initialization, common norm clipping and SAC initial entropy coefficient 0.001. |
| `paper/main.tex:584` | Update warm-up, network size, reward scale, DDPG actor learning rate, DQN loss/Polyak coefficient and epsilon schedule; add a replay-horizon row. The patch lists every value. |
| `paper/results/tables_params/params_generative.tex:30` | Replace lambda 2→0.5, q 1→0, k-star 1000→100000, alpha 0.01→0.05, beta 1→0.002. Update the q and k-star comments because headline shaping is active. |

No market-clearing equation, chronology, signed-auction admissibility rule,
reference policy formula or definition of J/bar-J-lambda needs replacement.
The parameter changes preserve the model's structure but change its numerical
calibration; do not describe them as an invariant reparameterization.

The numerical-results discussion must distinguish shaped training return from
economic performance. The saved development checks support positive mean
economic performance for all four selected policies and synthetic mean
outperformance of the references. They do **not** support a claim that every
method converges monotonically, that all outperform AS on historical data, or
that the auction always produces a large benefit. Use the complete results and
limitations in [pathology_repair.md](pathology_repair.md) when revising that
discussion. Do not substitute these development episodes for the final
publication test matrix.
