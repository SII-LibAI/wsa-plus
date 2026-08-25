#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CHECKPOINT="${CHECKPOINT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/wsa_diagnostic/$(date +'%Y_%m_%d_%H_%M_%S')}"
STATS_PATH="${STATS_PATH:-${DATASET_EXTERNAL_STATS_PATH:-}}"
STATS_ROOT="${STATS_ROOT:-${DATASET_EXTERNAL_STATS_ROOT:-}}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-${QWEN3_VL_PRETRAINED_PATH:-}}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${QWEN3_VL_PROCESSOR_PATH:-}}"
COSMOS_TOKENIZER_PATH="${COSMOS_TOKENIZER_PATH:-${COSMOS_TOKENIZER_PATH_OR_NAME:-}}"
ACTION_MODE="${ACTION_MODE:-}"
FRACTION="${FRACTION:-0.10}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_CURVE_SAMPLES="${NUM_CURVE_SAMPLES:-8}"
INFERENCE_STEPS="${INFERENCE_STEPS:-}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LOG_EVERY="${LOG_EVERY:-20}"

if [[ -z "${CHECKPOINT}" || -z "${DATASET_ROOT}" ]]; then
  echo "Usage:"
  echo "  CHECKPOINT=/path/to/checkpoint DATASET_ROOT=/path/to/collection bash evaluation/wsa_m/run.sh"
  echo "The checkpoint type is detected automatically (WSA-Base or WSA-Memory)."
  echo "Optional: STATS_PATH=/path/to/aggregated_stats.json OUTPUT_DIR=/path/to/output"
  exit 1
fi

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --fraction "${FRACTION}"
  --max-samples "${MAX_SAMPLES}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --num-curve-samples "${NUM_CURVE_SAMPLES}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${STATS_PATH}" ]]; then ARGS+=(--stats-path "${STATS_PATH}"); fi
if [[ -n "${STATS_ROOT}" ]]; then ARGS+=(--stats-root "${STATS_ROOT}"); fi
if [[ -n "${QWEN3_VL_PATH}" ]]; then ARGS+=(--qwen3-vl-path "${QWEN3_VL_PATH}"); fi
if [[ -n "${PROCESSOR_PATH}" ]]; then ARGS+=(--processor-path "${PROCESSOR_PATH}"); fi
if [[ -n "${COSMOS_TOKENIZER_PATH}" ]]; then ARGS+=(--cosmos-tokenizer-path "${COSMOS_TOKENIZER_PATH}"); fi
if [[ -n "${ACTION_MODE}" ]]; then ARGS+=(--action-mode "${ACTION_MODE}"); fi
if [[ -n "${INFERENCE_STEPS}" ]]; then ARGS+=(--inference-steps "${INFERENCE_STEPS}"); fi

cd "${PROJ_ROOT}"
python evaluation/wsa_m/evaluate.py "${ARGS[@]}"
