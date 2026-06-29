#!/usr/bin/env bash
set -euo pipefail

# OpenAI-compatible vLLM server for MotionAtlas-Bench.
# Override any variable below from the environment, for example:
# MODEL_PROFILE=qwen4b MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct DP_SIZE=8 PORT=8000 bash scripts/serve_qwen3vl_vllm.sh

MODEL_PROFILE="${MODEL_PROFILE:-qwen4b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_NAME_OVERRIDE="${SERVED_MODEL_NAME:-${MODEL_NAME:-}}"
TP_SIZE_OVERRIDE="${TP_SIZE:-${TENSOR_PARALLEL_SIZE:-}}"
PP_SIZE="${PP_SIZE:-${PIPELINE_PARALLEL_SIZE:-1}}"
DP_SIZE="${DP_SIZE:-${DATA_PARALLEL_SIZE:-1}}"
DP_BACKEND="${DP_BACKEND:-${DATA_PARALLEL_BACKEND:-mp}}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-${DATA_PARALLEL_SIZE_LOCAL:-}}"
DP_START_RANK="${DP_START_RANK:-${DATA_PARALLEL_START_RANK:-}}"
DP_ADDRESS="${DP_ADDRESS:-${DATA_PARALLEL_ADDRESS:-}}"
DP_RPC_PORT="${DP_RPC_PORT:-${DATA_PARALLEL_RPC_PORT:-}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-auto}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":128}}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

case "${MODEL_PROFILE}" in
  qwen4b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
    DEFAULT_SERVED_MODEL_NAME="qwen3-vl-4b"
    DEFAULT_TP_SIZE="1"
    ;;
  qwen8b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
    DEFAULT_SERVED_MODEL_NAME="qwen3-vl-8b"
    DEFAULT_TP_SIZE="1"
    ;;
  qwen32b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-32B-Instruct}"
    DEFAULT_SERVED_MODEL_NAME="qwen3-vl-32b"
    DEFAULT_TP_SIZE="4"
    ;;
  qwen235b)
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
    DEFAULT_SERVED_MODEL_NAME="qwen3-vl-235b-a22b"
    DEFAULT_TP_SIZE="8"
    ;;
  custom)
    if [[ -z "${MODEL_PATH:-}" ]]; then
      echo "MODEL_PATH is required when MODEL_PROFILE=custom" >&2
      exit 2
    fi
    DEFAULT_SERVED_MODEL_NAME="$(basename "${MODEL_PATH}")"
    DEFAULT_TP_SIZE="1"
    ;;
  *)
    echo "Unknown MODEL_PROFILE=${MODEL_PROFILE}. Use qwen4b, qwen8b, qwen32b, qwen235b, or custom." >&2
    exit 2
    ;;
esac

SERVED_MODEL_NAME="${MODEL_NAME_OVERRIDE:-${DEFAULT_SERVED_MODEL_NAME}}"
TP_SIZE="${TP_SIZE_OVERRIDE:-${DEFAULT_TP_SIZE}}"

args=(
  "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --dtype "${DTYPE}"
  --trust-remote-code
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi

if (( DP_SIZE > 1 )); then
  args+=(
    --data-parallel-size "${DP_SIZE}"
    --data-parallel-backend "${DP_BACKEND}"
  )
fi

if [[ -n "${DP_SIZE_LOCAL}" ]]; then
  args+=(--data-parallel-size-local "${DP_SIZE_LOCAL}")
fi

if [[ -n "${DP_START_RANK}" ]]; then
  args+=(--data-parallel-start-rank "${DP_START_RANK}")
fi

if [[ -n "${DP_ADDRESS}" ]]; then
  args+=(--data-parallel-address "${DP_ADDRESS}")
fi

if [[ -n "${DP_RPC_PORT}" ]]; then
  args+=(--data-parallel-rpc-port "${DP_RPC_PORT}")
fi

if [[ "${MODEL_PROFILE}" == "qwen235b" || "${ENABLE_EXPERT_PARALLEL:-0}" == "1" ]]; then
  args+=(--enable-expert-parallel)
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_VLLM_ARGS})
  args+=("${extra_args[@]}")
fi

vllm serve "${args[@]}"
