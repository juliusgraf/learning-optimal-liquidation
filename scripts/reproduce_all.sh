#!/usr/bin/env bash
# Reproduce every reported result from scratch (Phase 8 fills this in).
#
# Planned shape:
#   lmm-train    --config configs/base.yaml --config configs/synthetic_rough_heston.yaml \
#                --config configs/algo/dqn.yaml --run-name dqn_seed42
#   lmm-evaluate --config ... --run-dir results/synthetic_rough_heston/dqn_seed42
#   lmm-regret   --run-dir ...
#   lmm-make-figures --run-dir ...
#   lmm-make-tables  --run-dir ...
set -euo pipefail
echo "Phase 8: not yet implemented" >&2
exit 1
