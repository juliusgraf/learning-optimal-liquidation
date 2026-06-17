#!/usr/bin/env bash
# Synthetic rough-Heston setting, SAC. Usage: scripts/run_synthetic_sac.sh [--seed N] [--smoke]
source "$(dirname "$0")/_common.sh"
parse_common_args "$@"
run_experiment synthetic_rough_heston configs/synthetic_rough_heston.yaml configs/algo/sac.yaml sac
