# Bounded refinement diagnostic — 2026-09-07

32 common exogenous paths, environment seeds 927401–927432, and independently coupled Brownian drivers. The fixed policy is `results/revision_v20/synthetic_rough_heston/dqn_seed42/checkpoints/best.pt`. Its saved calibration, forecast coefficients, embedded feature normalizer and inventory reference are held fixed. Auction primitives are coupled by decision and proposal. No policy was retrained or reselected.

The entire resolved configuration, checkpoint/source hashes, per-episode observations, weighting definitions and runtime versions are saved in [diagnostic_dqn_seed42.json](diagnostic_dqn_seed42.json). This is one policy checkpoint and a small numerical diagnostic, not a replacement experiment campaign.

## Prices and variance

RMSE compares unrounded prices at the original observation times with the 0.25-minute candidate. Prices are rebased model-price units (tick size 0.01). Negative-node rates pool internal nodes before truncation; negative-time rates use duration-weighted left endpoints and equal episode weights.

| Maximum step | Mean nodes | Price RMSE vs 0.25m | Max absolute difference | Rounded-price RMSE | Negative nodes | Negative time | Minimum V |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 42.4 | 0.100055 | 0.553294 | 0.100334 | 7.006% | 7.347% | -0.010152 |
| 1m | 173.4 | 0.106574 | 0.509557 | 0.106730 | 7.405% | 7.376% | -0.012790 |
| 0.5m | 344.9 | 0.088310 | 0.470929 | 0.088470 | 8.363% | 8.441% | -0.012094 |
| 0.25m | 688.6 | 0.000000 | 0.000000 | 0.000000 | 9.444% | 9.404% | -0.011814 |

| Maximum step | Revealed price mean / SD | Interval log-return mean / SD | Opening log-return mean / SD |
|---|---:|---:|---:|
| legacy | 100.0482 / 0.2972363 | 6.276883e-06 / 0.0007652 | 0.000259706 / 0.004264078 |
| 1m | 100.0514 / 0.2866445 | 1.133376e-05 / 0.0007424235 | 0.0004689345 / 0.00396193 |
| 0.5m | 100.0674 / 0.2741198 | 2.044338e-05 / 0.000720899 | 0.000845845 / 0.003691554 |
| 0.25m | 100.0502 / 0.2825225 | 1.165253e-05 / 0.0007345522 | 0.0004821233 / 0.003761425 |

Price and interval-return moments pool original observations (initial/opening included for prices). Opening returns have one value per episode. Second moments are retained in the JSON. All simulated outputs were finite.

The discrepancies are not monotone: the 1-minute candidate has slightly greater RMSE against 0.25 minutes than legacy does in this sample. Negative computed variance remains material and its node frequency rises with refinement. Neither 0.25 minutes nor any other candidate is established as accurate by this comparison.

## Fixed-policy sensitivity

Positive shortfall changes mean higher cost. Paired changes subtract the legacy-mesh result on the same exogenous path. Parentheses contain standard errors of paired episode differences. Forecast MAE/RMSE are means of the existing episode-level errors over the original decision observations, in model-price units.

| Maximum step | Mean shortfall (bps) | Paired change (SE) | Mean penalized shortfall (bps) | Paired change (SE) | Forecast MAE | Forecast RMSE |
|---|---:|---:|---:|---:|---:|---:|
| legacy | -11.6593 | +0.0000 (0.0000) | -10.9239 | +0.0000 (0.0000) | 0.123033 | 0.165432 |
| 1m | -11.1905 | +0.4687 (1.0284) | -10.3372 | +0.5867 (1.0344) | 0.109720 | 0.148148 |
| 0.5m | -12.4332 | -0.7739 (0.8762) | -11.8899 | -0.9660 (0.9359) | 0.110104 | 0.146998 |
| 0.25m | -11.6847 | -0.0255 (1.2316) | -11.3130 | -0.3891 (1.3603) | 0.112116 | 0.151068 |

The JSON also reports paired forecast-error changes. These small-sample policy results do not establish policy robustness or performance under a newly trained refined generator. The diagnostic legacy column uses finest-grid-aggregated Brownian increments and proposal-level auction coupling; it is not a rerun of the original seeded publication evaluation.

## Computational costs

Wall-clock mean for environment construction/reset; maximum tracemalloc peak across 32 resets. Includes path simulation and preparation, excludes common finest-driver generation and policy loading. Traced allocations are not process resident memory. The unchanged full-history recursion has quadratic work and linear retained storage in the internal node count. Measurements are machine/runtime dependent.

| Maximum step | Mean reset (ms) | Relative to legacy | Peak traced allocation (KiB) | Mean fixed-policy episode (ms) |
|---|---:|---:|---:|---:|
| legacy | 8.605 | 1.00x | 226.6 | 70.667 |
| 1m | 15.341 | 1.78x | 250.2 | 72.717 |
| 0.5m | 23.884 | 2.78x | 282.5 | 76.004 |
| 0.25m | 45.102 | 5.24x | 357.5 | 86.619 |

Runtime: Python 3.14.4, NumPy 2.4.6, macOS-26.6.2-arm64-arm-64bit-Mach-O.

## Verification and remaining validation

- `python -m pytest -q -m "not slow and not network and not market_data"`: **715 passed, 34 deselected**, 14 expected fixture/legacy warnings; 211.94 seconds. An earlier in-flight run saw a now-corrected test import; the final run passed.
- The 17 new refinement tests passed again after the final grid edge-case guard/docstring cleanup (0.62 seconds). Relevant auction, chronology, model and checkpoint tests also passed in targeted runs.
- Ruff passed for the changed model/config/checkpoint modules and new diagnostic/tests; `git diff --check` passed.
- A two-path-per-split forecast-fitter smoke test wrote to `/tmp/rh-refined-forecast-smoke`; the refined overlay reloaded with its generator identity. Those smoke coefficients were not used in the reported fixed-policy comparison and are not a validated calibration.
- `pdflatex -interaction=nonstopmode -halt-on-error` compiled the modified manuscript to a temporary 39-page PDF. Existing user edits and published result files were retained.

Remaining work: use more paths, finer candidate meshes and additional parameter regimes to assess numerical behavior and cost; repeat fixed-policy sensitivity across policies/checkpoints; then choose a mesh based on explicit application tolerances. For new experiments, refit training-only forecasts, normalization and reference conditioning, and retrain/reselect policies under the chosen generator. Retrained-policy comparisons and the full 520-run campaign were not run. Existing manuscript results retain legacy-generator provenance.

See [README.md](README.md) for exact commands, configuration rules, random-stream ownership, artifact compatibility and the changed-file inventory.
