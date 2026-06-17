#!/usr/bin/env bash
# Historical S&P 500 setting, TD3, looping the 5 tickers (or --symbol T).
# Usage: scripts/run_historical_td3.sh [--seed N] [--smoke] [--symbol TICKER]
source "$(dirname "$0")/_common.sh"
parse_common_args "$@"
run_historical configs/algo/td3.yaml td3
