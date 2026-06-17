#!/usr/bin/env bash
# Historical S&P 500 setting, SAC, looping the 5 tickers (or --symbol T).
# Usage: scripts/run_historical_sac.sh [--seed N] [--smoke] [--symbol TICKER]
source "$(dirname "$0")/_common.sh"
parse_common_args "$@"
run_historical configs/algo/sac.yaml sac
