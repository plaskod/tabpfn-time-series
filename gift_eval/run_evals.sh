#!/usr/bin/env bash
# run_eval.sh
# Runs evaluate.py over multiple datasets and context lengths.

set -euo pipefail

# Visible GPUs and other fixed params
GPUS="0,1,2,3"
BATCH_SIZE=2048
PYTHON_BIN="python"
MODEL_PREFIX="tabpfn-ts"

# Datasets to iterate over
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

# SEEDS=(0 42 45)
# Context lengths to iterate over
CONTEXT_LENGTHS=(100 200 300 500 1000 2000 3000 4096 5000)

for CNTX in "${CONTEXT_LENGTHS[@]}"; do
  MODEL_NAME="${MODEL_PREFIX}-${CNTX}cntx-run4"
  for DS in "${DATASETS[@]}"; do
    echo "=== Running: dataset=${DS} | context_length=${CNTX} | model_name=${MODEL_NAME} ==="
    CUDA_VISIBLE_DEVICES="${GPUS}" \
      "${PYTHON_BIN}" evaluate.py \
        --dataset "${DS}" \
        --batch_size "${BATCH_SIZE}" \
        --context_length "${CNTX}" \
        --model_name "${MODEL_NAME}"
    echo
  done
done

echo "All runs completed."
