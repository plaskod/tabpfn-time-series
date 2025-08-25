#!/usr/bin/env bash
# run_eval.sh
# Runs evaluate.py over multiple datasets and a top_k sweep.
# Retrieval is enabled by passing --few_shot_k. We do NOT sweep prediction_length here;
# datasets/terms define it internally. We treat CTX_LEN as both base context length and
# the proxy for prediction length in k_max scheduling (per budget).

set -euo pipefail

GPUS="0,1,2,3"
BATCH_SIZE=2048
PYTHON_BIN="python"
MODEL_PREFIX="tabpfn-ts"
ROW_BUDGET=10000

DATASETS=(
  bizitobs_service
  hierarchical_sales/D
  hierarchical_sales/W
  LOOP_SEATTLE/D
  SZ_TAXI/15T
  SZ_TAXI/H
  M_DENSE/H
  M_DENSE/D
  bitbrains_fast_storage/H
  bizitobs_application
  bitbrains_rnd/H
  bizitobs_l2c/H
)

declare -A TERMS_BY_DATASET=(
  ["bizitobs_service"]="short medium long"
  ["hierarchical_sales/D"]="short"
  ["hierarchical_sales/W"]="short"
  ["LOOP_SEATTLE/D"]="short"
  ["SZ_TAXI/15T"]="short medium long"
  ["SZ_TAXI/H"]="short"
  ["M_DENSE/H"]="short medium long"
  ["M_DENSE/D"]="short"
  ["bitbrains_fast_storage/H"]="short"
  ["bizitobs_application"]="short medium long"
  ["bitbrains_rnd/H"]="short"
  ["bizitobs_l2c/H"]="short medium long"
)

# base_context_len per dataset (from your table)
declare -A CTX_LEN=(
  ["LOOP_SEATTLE/D"]=30
  ["M_DENSE/D"]=30
  ["M_DENSE/H"]=48
  ["SZ_TAXI/15T"]=48
  ["SZ_TAXI/H"]=48
  ["bitbrains_fast_storage/H"]=48
  ["bitbrains_rnd/H"]=48
  ["bizitobs_application"]=60
  ["bizitobs_l2c/H"]=48
  ["bizitobs_service"]=60
  ["hierarchical_sales/D"]=30
  ["hierarchical_sales/W"]=8
)

# max_k per (dataset|term) (your tuned caps)
declare -A MAX_K=(
  ["LOOP_SEATTLE/D|short"]=100
  ["M_DENSE/D|short"]=100
  ["M_DENSE/H|short"]=100
  ["M_DENSE/H|medium"]=15
  ["M_DENSE/H|long"]=10
  ["SZ_TAXI/15T|short"]=103
  ["SZ_TAXI/15T|medium"]=18
  ["SZ_TAXI/15T|long"]=12
  ["SZ_TAXI/H|short"]=103
  ["bitbrains_fast_storage/H|short"]=103
  ["bitbrains_rnd/H|short"]=103
  ["bizitobs_application|short"]=82
  ["bizitobs_application|medium"]=15
  ["bizitobs_application|long"]=10
  ["bizitobs_l2c/H|short"]=103
  ["bizitobs_l2c/H|medium"]=18
  ["bizitobs_l2c/H|long"]=12
  ["bizitobs_service|short"]=82
  ["bizitobs_service|medium"]=15
  ["bizitobs_service|long"]=10
  ["hierarchical_sales/D|short"]=166
  ["hierarchical_sales/W|short"]=624
)

TOPK_CANDIDATES=(1 2 3 5 8 10 15 20 40 60 80 90 100 150 200 300 500)

# evaluate.py supports --terms, set to 1 to pass it through.
EVAL_HAS_TERM=1

k_max_for() {
  local ds="$1" term="$2" pred="$3"
  local ctx="${CTX_LEN["$ds"]}"
  local base_max="${MAX_K["$ds|$term"]}"
  local num=$(( ROW_BUDGET - ctx ))
  if (( num <= 0 )); then echo 1; return; fi
  local den=$(( ctx + pred ))
  local dyn=$(( num / den ))     # floor
  (( dyn < 1 )) && dyn=1
  if (( dyn > base_max )); then echo "$base_max"; else echo "$dyn"; fi
}

build_k_sweep() {
  local ds="$1" term="$2" pred="$3"
  local kmax; kmax=$(k_max_for "$ds" "$term" "$pred")
  local arr=()
  for c in "${TOPK_CANDIDATES[@]}"; do
    (( c <= kmax )) && arr+=("$c")
  done
  arr+=("$kmax")
  printf "%s\n" "${arr[@]}" | sort -n -u | tr '\n' ' '
}

for DS in "${DATASETS[@]}"; do
  TERMS="${TERMS_BY_DATASET[$DS]}"
  for TERM in $TERMS; do
    # Use CTX as proxy for pred in budget math (ctx + pred) per our setup.
    CTX="${CTX_LEN["$DS"]}"
    PRED="$CTX"
    K_SWEEP=$(build_k_sweep "$DS" "$TERM" "$PRED")
    for K in $K_SWEEP; do
      MODEL_NAME="${MODEL_PREFIX}-random-retrieval-top-k${K}-ctx${CTX}-dataset-${DS}-term-${TERM}"
      echo "=== Running: dataset=${DS} | term=${TERM} | context_length=${CTX} | top_k=${K} | model_name=${MODEL_NAME} ==="
      CUDA_VISIBLE_DEVICES="${GPUS}" \
        "${PYTHON_BIN}" evaluate.py \
          --dataset "${DS}" \
          --terms "${TERM}" \
          --batch_size "${BATCH_SIZE}" \
          --context_length "${CTX}" \
          # --debug \
          --few_shot_k "${K}" \
          --model_name "${MODEL_NAME}"
      echo
    done
  done
done

echo "All runs completed."
