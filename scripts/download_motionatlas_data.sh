#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/motionatlas-data}"

hf download maxLWSv2/motionatlas-data \
  --repo-type dataset \
  --local-dir "${DATA_DIR}" \
  --include \
    "data/motionatlas_v1/*" \
    "data/motionatlas_v2/*" \
    "data/recipe/train.parquet.part-*"

RECIPE_DIR="${DATA_DIR}/data/recipe"
if compgen -G "${RECIPE_DIR}/train.parquet.part-*" > /dev/null; then
  cat "${RECIPE_DIR}"/train.parquet.part-* > "${RECIPE_DIR}/train.parquet"
  echo "Wrote ${RECIPE_DIR}/train.parquet"
else
  echo "No recipe parquet parts found under ${RECIPE_DIR}" >&2
fi
