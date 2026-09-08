# Synthetic v20: pre-replacement versus latest results

Before: archived decision-grid generator. Latest: maximum internal step 0.25 minutes, new training-only calibration and retrained/reselected policies. Historical report table values agree exactly. These are descriptive campaign changes, not coupled-mesh numerical errors or causal estimates of refinement alone.

All shortfalls and treatment effects below are in basis points. Intervals are pointwise 95% seed bootstrap intervals within each campaign; no interval for the between-campaign change is claimed.

## Headline inventory-penalized shortfall (lower is better)

| Policy | Before | Latest | Latest − before |
|---|---:|---:|---:|
| DQN | -3.36 | -4.19 | -0.83 |
| DDPG | -4.06 | -4.03 | +0.03 |
| TD3 | -3.82 | -4.40 | -0.58 |
| SAC | -3.73 | -4.54 | -0.81 |
| AS | -0.12 | -1.41 | -1.28 |
| TWAP | 6.45 | 5.61 | -0.83 |

## Matched treatment effects (positive favors first condition)

| Contrast / learner | Before [95% CI] | Latest [95% CI] |
|---|---:|---:|
| Auction access, matched information and economic training (on - off) / ddpg | 0.37 [0.16, 0.57] | 1.05 [0.75, 1.26] |
| Auction access, matched information and economic training (on - off) / dqn | 0.61 [-0.12, 1.38] | 0.57 [0.12, 1.08] |
| Auction access, matched information and economic training (on - off) / sac | 0.75 [0.54, 0.97] | 0.77 [0.14, 1.41] |
| Auction access, matched information and economic training (on - off) / td3 | 0.55 [0.20, 0.87] | 0.62 [0.30, 0.93] |
| Auction anchor (indicative - frozen mid) / ddpg | 0.04 [-0.42, 0.60] | -0.37 [-0.72, -0.06] |
| Auction anchor (indicative - frozen mid) / dqn | -0.77 [-1.43, -0.24] | -0.44 [-0.90, 0.07] |
| Auction anchor (indicative - frozen mid) / sac | -0.27 [-0.55, 0.09] | -0.92 [-1.40, -0.48] |
| Auction anchor (indicative - frozen mid) / td3 | -0.69 [-1.04, -0.37] | -0.42 [-0.67, -0.07] |
| Combined manuscript preferences (on - off) / ddpg | -0.19 [-0.50, 0.08] | -0.20 [-0.46, 0.09] |
| Combined manuscript preferences (on - off) / dqn | 0.25 [-0.21, 0.81] | 0.02 [-0.32, 0.35] |
| Combined manuscript preferences (on - off) / sac | -0.23 [-0.56, 0.04] | -0.02 [-0.24, 0.24] |
| Combined manuscript preferences (on - off) / td3 | 0.49 [-0.15, 1.29] | -0.13 [-0.34, 0.06] |
| Dense auction credit (on - off) / ddpg | 1.80 [0.94, 2.69] | 2.08 [0.95, 3.23] |
| Dense auction credit (on - off) / dqn | 0.91 [0.31, 1.60] | 0.73 [0.18, 1.19] |
| Dense auction credit (on - off) / sac | 1.05 [0.44, 1.65] | 0.74 [0.24, 1.26] |
| Dense auction credit (on - off) / td3 | 1.11 [0.37, 1.69] | 1.78 [0.82, 2.70] |
| H observation, fixed anchor and economic training (on - off) / ddpg | 0.88 [0.20, 1.53] | -0.32 [-0.94, 0.11] |
| H observation, fixed anchor and economic training (on - off) / dqn | -0.25 [-0.77, 0.25] | -0.44 [-0.78, -0.09] |
| H observation, fixed anchor and economic training (on - off) / sac | -0.18 [-0.49, 0.13] | 0.44 [-0.00, 0.95] |
| H observation, fixed anchor and economic training (on - off) / td3 | 0.03 [-0.90, 0.74] | 0.03 [-0.52, 0.57] |
| cashflow / DQN | 3.28 [0.97, 6.90] | 3.53 [1.10, 6.64] |
| cashflow / DDPG | 3.54 [2.57, 4.61] | 3.65 [2.84, 4.57] |
| cashflow / TD3 | 4.41 [3.68, 5.26] | 4.83 [4.03, 5.70] |
| cashflow / SAC | 4.80 [3.77, 5.80] | 5.16 [4.08, 6.27] |
| economic_dense_h / DQN | -0.01 [-0.31, 0.29] | -0.01 [-0.60, 0.53] |
| economic_dense_h / DDPG | 0.33 [-0.06, 0.67] | 0.04 [-0.60, 0.57] |
| economic_dense_h / TD3 | 0.05 [-0.14, 0.22] | -0.12 [-0.31, 0.06] |
| economic_dense_h / SAC | -0.32 [-0.74, 0.09] | -0.16 [-0.30, -0.05] |

## Synthetic forecast errors (model-price units)

| Phase / metric | Before | Latest |
|---|---:|---:|
| auction / h_mae | 0.0488 | 0.0504 |
| auction / mid_mae | 0.1400 | 0.1471 |
| auction / mae_gain | 0.0911 | 0.0967 |
| auction / h_rmse | 0.0876 | 0.0908 |
| auction / mid_rmse | 0.1813 | 0.1964 |
| clob / h_mae | 0.1802 | 0.1726 |
| clob / mid_mae | 0.1926 | 0.1837 |
| clob / mae_gain | 0.0124 | 0.0111 |
| clob / h_rmse | 0.2902 | 0.2775 |
| clob / mid_rmse | 0.2958 | 0.2831 |

The CSV includes every synthetic economic, auction-mechanism and treatment mean and interval endpoint, plus forecast metrics. `comparison_inputs.json` binds all source files. Reproduce with `.venv/bin/python paper/results/compare.py`.
