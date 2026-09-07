# Writing the paper with Prism

Use Prism for section-level scientific writing and LaTeX revision. Keep this
repository and its frozen experiment artifacts as the reproducibility record.
The highest-value preparation is a small, ordered evidence packet; giving the
assistant every historical document at once makes conflicting versions harder
to resolve.

This workflow does not depend on automatic Git synchronization, patch execution,
Python execution, or persistent assistant memory inside Prism. Current Prism
import and integration features were not established by the official documentation
search available during this review. The supplied ZIP is a portable LaTeX project
archive, not a tested Prism integration. Use the import/upload options shown in
your workspace; if necessary, unpack it and add the project files directly.

## Read first and resolve conflicts in this order

| Material | Role |
|---|---|
| `docs/prism_handoff/technical_review.md` | Current alignment checklist and newly identified patch/metric issues |
| `docs/research_writeup_plan_v19.md` | Current completed-study narrative, claim limits, and integration order |
| Saved resolved configs, numeric tables and audit CSVs | Evidence for what ran and what the measured effects were |
| `docs/rl_design.md`, continuous-action description and this review's corrections | Explanation of the executable model and learning design |
| `docs/manuscript_recommendations_v19.patch` | Older proposed equation, algorithm and parameter edits; inspect selectively |
| `docs/manuscript_drafts/research_framing_v19.tex` | Proposed prose for abstract, introduction, results and conclusion |
| `paper/main.tex` and parameter input | Current editable manuscript; preserve notation/theory unless an identified correction applies |
| Initial scope, development diagnoses and referee audit | Historical rationale and limitations, not additional completed experiments |

An unresolved conflict between mathematical intent and the experiment should be
reported explicitly. The code tells you what generated the results; it does not
by itself decide whether a modeling assumption is scientifically desirable.

## The practical workflow

1. **Start from a frozen copy.** The archive preserves the original manuscript
   and source paths. Its main document is `paper/main.tex`; compile from `paper/`
   so the existing relative inputs resolve. Keep the untouched baseline ZIP and
   its hash manifest outside the writing workspace. Do not upload checkpoints,
   replay buffers, raw market archives or every diagnostic dashboard.
2. **Establish the brief before editing.** Give Prism the prompt below and the
   review/writing plan. Ask it to restate the model, three research questions and
   outstanding discrepancies. Correct that understanding before accepting prose.
3. **Align the specification first.** Work through review items R1–R5 in bounded
   edits: reward definitions, forecast, features/conditioning, learner pseudocode,
   then parameter tables. Check every affected use of H, J, terminal reward and
   normalization. Do not apply the entire old patch as a single accepted change.
4. **Write the experimental protocol.** Identify the initial 440 runs and the two
   subsequent 40-run controls; describe selection, seed pairing, confidence
   intervals and reused historical dates. Name the exact arms behind each effect.
5. **Write results from exhibits.** Follow the plan: economic performance and
   auction value; forecast information; delayed auction learning. Attach the
   relevant table/CSV for that section. Require a source row for every number.
   Use the existing prose blocks as a starting draft. Assemble the planned four
   main figures from saved records later; the current PDFs are evidence exhibits,
   not yet the final four-figure manuscript layout.
6. **Write the abstract and conclusion last.** Reconcile them with the actual
   completed results. Retain the limitations and uncertain/negative contrasts.
7. **Round-trip after each major section.** Export/download the revised LaTeX
   files into a separate local directory. Compare them with the frozen baseline
   and bring back only reviewed manuscript/bibliography/figure changes. Run a
   local compile and a claim-to-source audit in Codex before taking the next
   snapshot. Let one editor own a file at a time; do not edit the same manuscript
   simultaneously in Prism and the repository.

Maintain a short change log with: manuscript location, old statement, accepted
change, source path/row, and resolution status. Keep this visible with the writing
brief so continuing work does not depend on remembering an earlier chat.

## Starter prompt

```text
You are helping revise an existing research manuscript about end-of-day
liquidation through a CLOB followed by a signed closing auction.

Read docs/prism_handoff/technical_review.md and
docs/research_writeup_plan_v19.md first, then paper/main.tex.
Use saved numerical tables and resolved configurations as evidence for the
completed experiments. The older manuscript patch is advisory and predates
two completed follow-ups. Do not apply it wholesale. Proposed prose blocks
have designated insertion locations and must not be appended as one section.

The study is complete: 440 main runs plus two later 40-run synthetic comparisons.
Do not invent results, controls, citations or new experiments. Distinguish:
(1) weighted preference objective J,
(2) potential/reference adjustments used only for learning,
(3) marked-to-market economic PnL and terminal-risk-adjusted PnL.
The headline trains on (1); policy selection and comparisons use (3).
Return aliases are not comparable across arms with different centering settings.

Keep the two-sided auction, actual pro-rata settlement, frozen residual mark,
projected continuous controls, and partial-observation qualification explicit.
Historical inputs are midprice paths in a simulated market, with reused test
dates. Dense auction credit helps; forecast value depends on the learner;
extra preference terms have no established marginal benefit at this calibration.

First return a concise model summary and a discrepancy checklist with source
locations. Then revise one requested section at a time. For each proposed
change provide the LaTeX, the evidence path or mathematical derivation, and
any unresolved issue. Preserve existing labels and citation keys where possible.
Do not silently change definitions or fill missing evidence with plausible text.
```

## Example section request

```text
Revise only the reward specification and its explanatory paragraphs.
Resolve R1 and R4 of the technical review. Define weighted J consistently,
include omega_A=10^-4 in both fictive auction terms, and make the cancellation
clawback use the credit actually assigned when the order was submitted.
Keep cash, cancellation fees and terminal risk unweighted. Explain q=0.
Then state the centering and replay transformations separately and prove their
finite-episode identity. Identify every equation or later reference needing a
corresponding update. Do not rewrite results or introduction in this pass.
```

## Evidence map for the results sections

| Question | Evidence in the repository/archive | Interpretation |
|---|---|---|
| Economic performance | `results/revision_v19/_publication/tables/economic_performance.csv` and matching PDF | Risk-adjusted outcome; separate raw PnL and terminal risk |
| Direct auction value | Main `tables/auction_mechanism.csv`, matching PDF, `docs/analysis_v19/historical_macro.csv` | Same-CLOB no-order comparison, including inventory-risk relief |
| Retrained auction access | Main `tables/treatments.csv`, access rows | Economic H-off access versus no-auction; not forced liquidation |
| H accuracy | `docs/analysis_v19/forecast_summary.csv` | Prediction of eventual simulated clearing on realized policy paths |
| Incremental H value | Main `tables/treatments.csv`, information rows | H-on minus H-off with common frozen-mid anchor and economic guidance |
| Dense auction credit | Main treatment table and `figures/credit_assignment.pdf` | Economic dense minus sparse; CLOB conditioning retained |
| Entire scheme versus cash flow | `results/revision_v19_cashflow/_comparison/` | Bundles preference and conditioning changes |
| Marginal preferences at headline anchor | `docs/analysis_dense_h_v19/summary.csv` and dense-H comparison folder | Headline minus economic dense H; all four intervals contain zero |

Report all numbers in their stated units. A basis point is 1e-4 of initial
notional; at S0=100 and I0=100, one model currency unit happens to equal one
basis point. Do not treat that numerical coincidence as a general unit identity.
Do not compare raw `return_undisc` between the uncentered cash-flow extension and
the headline: it includes 10,000 initial-notional units in the former.

## What to leave out of the initial writing context

`legacy/`, `audit/`, archived development trials, test fixtures, checkpoints,
raw market data, and old run dashboards are not needed to draft the paper.
The research source ledger points to theoretical references; it does not replace
reading those references before adding or strengthening literature claims.
Optional, rejected or proposed controls must not be described as completed runs.

Keep executable `.py` sources as optional supporting evidence. Prism need not
execute them: return implementation questions to this repository, where code,
tests, saved configurations and provenance can be inspected together.
