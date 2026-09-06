# Matched economic training with H anchoring and dense credit

The author-requested comparator `mechanism_economic_dense_h` trains toward the
economic objective while retaining the headline's H observation, H-centred
auction quote grid and complete reward conditioning. It is distinct from the
completed fixed-anchor economic-dense and H-anchored raw-cash-flow treatments.

## What is held fixed

Both arms use the same simulator, admissible actions, forecast calibration,
networks, observation normalization, exploration, optimizer settings, reward
scale, episode cap, checkpoint selection and evaluation protocol. Both retain
auction and CLOB inventory potentials, initial-inventory-value centering and
the same market-return control variate. The auxiliary `learning_credit_baseline`
remains false in both arms, as in the headline configuration.

Only the manuscript preference switches differ: the economic comparator turns
off CLOB reward multipliers, fictive auction credits and their clawbacks, and
extra terminal auction preferences. The final manuscript purchase parameter q
is already zero in the headline; this treatment does not add a separate q study.

Writing B_t for the extra manuscript preference reward and C_t for common
conditioning, the replay rewards satisfy, on a common state/action trajectory:

    headline:   r_economic,t + B_t + C_t
    comparator: r_economic,t       + C_t

Thus their difference is B_t alone. Common conditioning can take different
realized values once independently learned policies choose different actions;
the conditioning function and its parameters are identical. The potential
telescopes with terminal correction, and the market baseline preserves policy
differences. Training-return logs exclude replay-only conditioning. Both
policies are selected and evaluated on economic risk-adjusted PnL.

The resulting test contrast isolates the combined manuscript preferences under
the actual headline H anchor, conditional on the declared learning and stopping
protocol. Positive means the preferences helped; negative means they hurt; an
interval crossing zero is inconclusive, not proof of equivalence. It does not
identify the effect of each preference term individually.

## Scope and launch

This extension adds 40 synthetic runs (four algorithms times ten canonical
training seeds) and reuses the saved v19 synthetic headlines. No historical
training or previous full campaign is repeated. The existing raw-cash-flow
results and original v19 publication outputs are preserved.

After committing reviewed changes and cleaning the tree:

```bash
cd /Users/juliusgraf/Learning-Market-Making
source .venv/bin/activate
python -m lmm.experiments.cashflow_comparison --conditioned --jobs 10 --threads-per-job 1
```

Append `--dry-run` to inspect the 40 jobs. The new result root defaults to
`results/revision_v19_economic_dense_h`. The completed comparison is generated
at `_comparison/index.html` within that root, with `comparison.csv`,
`by_seed.csv`, `comparison.pdf`, `comparison.png`, and the input-hash manifest.
The control performance column is named `economic_dense_bps`; positive
`difference_bps` means headline J minus conditioned economic training.
Use `--conditioned --report-only` to regenerate from saved outputs.

The original command without `--conditioned` still targets the separate raw
cash-flow experiment. The default 440-run launcher is unchanged.

Once this extension completes, three comparisons are possible using saved
results: J versus conditioned economic training (preferences), conditioned
economic versus raw economic training (conditioning package), and J versus
raw economic training (complete scheme). All retain H anchoring. The earlier
fixed-anchor dense-versus-sparse study remains the separate auction-credit
ablation. These distinctions should remain explicit in the write-up.

## Verification performed

- All 25 targeted tests passed, including all four algorithms' matched-path
  tests proving that replay differences equal only manuscript preferences.
- Eight smoke runs (four algorithms times headline/new comparator), with four
  training episodes and three test episodes each, completed through evaluation
  and report generation. These are pipeline checks, not performance evidence.
- The smoke comparator retains nonzero intermediate auction credit while its
  recorded economic training return equals the economic objective.
- All 40 archived raw-cash-flow/headline pairs still pass validation after the
  reporter extension. No archived outputs were regenerated or modified.
- The generated smoke plot was visually inspected. Artifacts are in
  `/tmp/lmm-dense-h-smoke-20260906`.

Neither protected TeX file was changed. Production training has not been launched.
