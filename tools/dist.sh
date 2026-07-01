#!/usr/bin/env bash
set -euo pipefail
set -x

FILE=${1:?Usage: bash tools/dist.sh train CONFIG GPUS [extra args]}
CONFIG=${2:?Usage: bash tools/dist.sh train CONFIG GPUS [extra args]}
GPUS=${3:?Usage: bash tools/dist.sh train CONFIG GPUS [extra args]}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((55500 + RANDOM % 2000))}
DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}
DIST_TIMEOUT_SECONDS=${DIST_TIMEOUT_SECONDS:-3600}
TORCHELASTIC_TIMEOUT=${TORCHELASTIC_TIMEOUT:-18000}

export DIST_TIMEOUT_SECONDS
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --rdzv_conf "timeout=${TORCHELASTIC_TIMEOUT}" \
  --nproc_per_node="${GPUS}" \
  "tools/${FILE}.py" "${CONFIG}" --launcher pytorch --deepspeed "${DEEPSPEED}" "${@:4}"

