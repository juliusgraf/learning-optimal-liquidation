#!/usr/bin/env bash
# Build the paired cross-treatment publication table from a complete balanced
# synthetic matrix. The canonical synthetic runs are reused as the
# H-on/shaping-on headline arm; H-on/shaping-off is a distinct treatment.
#
# Usage: scripts/make_treatment_outputs.sh
#          [--seeds "42 7 99 123 2024 314 577 811 1618 2718"] [--publication]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

SEEDS="42 7 99 123 2024 314 577 811 1618 2718"
PUBLICATION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    --publication) PUBLICATION=1; shift ;;
    -h|--help)
      echo "usage: $0 [--seeds \"42 7 99 123 2024 314 577 811 1618 2718\"] [--publication]"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for requested_seed in $SEEDS; do
  case "$requested_seed" in
    ''|*[!0-9]*) echo "invalid nonnegative integer seed: $requested_seed" >&2; exit 2 ;;
  esac
done
SEEDS=$(echo "$SEEDS" | tr ' ' '\n' | grep -v '^$' | sort -un | tr '\n' ' ')
requested_seed_count=$(echo "$SEEDS" | wc -w | tr -d ' ')
if [[ "$requested_seed_count" -lt 2 ]]; then
  echo "--seeds must contain at least two distinct master seeds" >&2
  exit 2
fi
if [[ "$PUBLICATION" -eq 1 ]]; then
  canonical_count=0
  for canonical_seed in 42 7 99 123 2024 314 577 811 1618 2718; do
    case " $SEEDS " in
      *" $canonical_seed "*) canonical_count=$((canonical_count + 1)) ;;
    esac
  done
  if [[ "$requested_seed_count" -ne 10 || "$canonical_count" -ne 10 ]]; then
    echo "--publication requires exactly the canonical seeds: 42 7 99 123 2024 314 577 811 1618 2718" >&2
    exit 2
  fi
fi

RESULTS_ROOT="${LMM_RESULTS_ROOT:-results/revision_v19}"
[[ -d "$RESULTS_ROOT" ]] || { echo "no $RESULTS_ROOT directory" >&2; exit 1; }

# Parallel arrays: resolved experiment.name and stable run-name suffix.  The
# first row is the reused headline arm.
SETTINGS=(
  synthetic_rough_heston
  synthetic_rough_heston__ablation_h_off_shaping_off
  synthetic_rough_heston__ablation_h_off_shaping_on
  synthetic_rough_heston__ablation_h_on_shaping_off
  synthetic_rough_heston__no_auction
  synthetic_rough_heston__no_cancellation
)
SUFFIXES=(
  ""
  __ablation_h_off_shaping_off
  __ablation_h_off_shaping_on
  __ablation_h_on_shaping_off
  __no_auction
  __no_cancellation
)
ALGOS=(dqn ddpg td3 sac)

group=()
missing=()
for seed in $SEEDS; do
  for ((i = 0; i < ${#SETTINGS[@]}; i += 1)); do
    setting="${SETTINGS[$i]}"
    suffix="${SUFFIXES[$i]}"
    for algo in "${ALGOS[@]}"; do
      run_dir="$RESULTS_ROOT/$setting/${algo}${suffix}_seed${seed}"
      if [[ ! -f "$run_dir/config_resolved.yaml" \
            || ! -f "$run_dir/seed.txt" \
            || ! -f "$run_dir/eval/records.csv" \
            || ! -f "$run_dir/eval/metadata.yaml" ]]; then
        missing+=("$run_dir")
        continue
      fi
      saved_seed=$(tr -dc '0-9' < "$run_dir/seed.txt")
      if [[ "$saved_seed" != "$seed" ]]; then
        echo "ERROR: $run_dir/seed.txt contains $saved_seed; expected $seed" >&2
        exit 1
      fi
      group+=("$run_dir")
    done
  done
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: incomplete six-arm x four-algorithm x ${requested_seed_count}-seed treatment matrix:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

OUT="$RESULTS_ROOT/synthetic_rough_heston/_cross_treatment/tables"
args=(
  --cross-treatment
  --run-dir "${group[@]}"
  --out "$OUT"
)
[[ "$PUBLICATION" -eq 1 ]] && args+=(--publication)

python3 -m lmm.experiments.make_tables "${args[@]}"
echo "### paired cross-treatment outputs complete (seeds: $SEEDS)"
echo "### artifacts remain under $OUT; export is explicit/manual"
