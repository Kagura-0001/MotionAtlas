#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml}
GPUS=${2:-8}

bash tools/dist.sh train "${CONFIG}" "${GPUS}" "${@:3}"

