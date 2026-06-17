#!/usr/bin/env bash
# Historical S&P 500 setting, DQN, looping the 5 tickers (or --symbol T).
# Usage: scripts/run_historical_dqn.sh [--seed N] [--smoke] [--symbol TICKER]
source "$(dirname "$0")/_common.sh"
parse_common_args "$@"
# Historical runs use the 500-ep budget (configs/historical_sp500.yaml); scale
# the DQN epsilon decay to ~0.6x that (300), overriding the 600 in
# configs/algo/dqn.yaml which is tuned for the synthetic 1000-ep budget.
EXTRA_OVERRIDES=(-o algo.hyperparams.epsilon_decay_episodes=300)
run_historical configs/algo/dqn.yaml dqn
