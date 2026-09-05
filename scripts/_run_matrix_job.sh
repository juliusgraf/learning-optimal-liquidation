#!/usr/bin/env bash
# Internal single-experiment entry point used by the bounded matrix queue.
# Positional arguments: seed algorithm block smoke(0|1).
# block is synthetic, a historical ticker, or one non-headline treatment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
seed="$1" algo="$2" block="$3" smoke="$4"
case "$algo" in dqn|ddpg|td3|sac) ;; *) echo "invalid algorithm: $algo" >&2; exit 2 ;; esac
case "$seed" in ''|*[!0-9]*) echo "invalid seed: $seed" >&2; exit 2 ;; esac
args=(--seed "$seed" --no-resume-ckpt --skip-completed)
case "$smoke" in 1) args+=(--smoke) ;; 0) ;; *) exit 2 ;; esac
parse_common_args "${args[@]}"
case "$block" in
  synthetic)
    run_experiment synthetic_rough_heston configs/synthetic_rough_heston.yaml "configs/algo/${algo}.yaml" "$algo"
    ;;
  MSFT|JPM|PG|GOOGL|CAT)
    SYMBOL="$block"
    run_historical "configs/algo/${algo}.yaml" "$algo"
    ;;
  ablation_h_off_shaping_off|ablation_h_off_shaping_on|ablation_h_on_shaping_off|no_auction|no_cancellation)
    EXTRA_CONFIGS=("configs/treatment/${block}.yaml")
    EXTRA_OVERRIDES=(-o "experiment.name=synthetic_rough_heston__${block}")
    RUN_NAME_SUFFIX="__${block}"
    run_experiment "synthetic_rough_heston__${block}" configs/synthetic_rough_heston.yaml "configs/algo/${algo}.yaml" "$algo"
    ;;
  *) echo "invalid matrix block: $block" >&2; exit 2 ;;
esac
