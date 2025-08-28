#!/usr/bin/env bash
# run_moment_retrieval_evals.sh
# Evaluate MOMENT retrieval on selected datasets using existing persistent ChromaDB indices.
# Each run is pinned to one GPU to avoid collisions with TabPFN compute.

set -euo pipefail

# Comma-separated GPU list to shard jobs
GPUS_CSV="0,1,2,3"
BATCH_SIZE=2048
PYTHON_BIN="python"
MODEL_PREFIX="tabpfn-ts-moment"
ROW_BUDGET=10000

# Datasets with available MOMENT collections
DATASETS=(
  bizitobs_service
  "M_DENSE/D"
  "LOOP_SEATTLE/D"
  "hierarchical_sales/D"
  "SZ_TAXI/H"
  bizitobs_application
  "bizitobs_l2c/H"
)

declare -A CTX_LEN=(
  ["LOOP_SEATTLE/D"]=30
  ["M_DENSE/D"]=30
  ["SZ_TAXI/H"]=48
  ["bizitobs_application"]=60
  ["bizitobs_l2c/H"]=48
  ["bizitobs_service"]=60
  ["hierarchical_sales/D"]=30
)

# Conservative top-k sweep (adjust if desired). Must stay ≤ what the DB coverage can satisfy.
TOPK_CANDIDATES=(1 3 5 10 20 40)

# Split GPUs into an array
IFS="," read -ra GPU_LIST <<< "$GPUS_CSV"
NUM_GPUS=${#GPU_LIST[@]}

# Simple round-robin GPU scheduler
job_idx=0
run_job() {
  local gpu="$1" dataset="$2" term="$3" ctx="$4" k="$5"
  local model_name="${MODEL_PREFIX}-moment-top-k${k}-ctx${ctx}-dataset-${dataset}-term-${term}"

  echo "=== Running MOMENT: dataset=${dataset} | term=${term} | context_length=${ctx} | top_k=${k} | gpu=${gpu} | model_name=${model_name} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" evaluate.py \
      --dataset "${dataset}" \
      --terms "${term}" \
      --batch_size "${BATCH_SIZE}" \
      --context_length "${ctx}" \
      --few_shot_k "${k}" \
      --retrieval_mode moment \
      --moment_device cuda \
      --moment_ctx_len 0 \
      --moment_index_sample_size 0 \
      --model_name "${model_name}" &
}

for DS in "${DATASETS[@]}"; do
  TERMS="short"
  for TERM in $TERMS; do
    CTX="${CTX_LEN["$DS"]}"
    # Use ctx as proxy for pred in the budget math (ctx + pred) and build a small sweep
    for K in "${TOPK_CANDIDATES[@]}"; do
      gpu=${GPU_LIST[$(( job_idx % NUM_GPUS ))]}
      run_job "$gpu" "$DS" "$TERM" "$CTX" "$K"
      job_idx=$(( job_idx + 1 ))
    done
  done
done

wait
echo "All MOMENT runs completed."


