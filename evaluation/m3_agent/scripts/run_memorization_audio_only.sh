#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_PY="${ENV_PY:-python}"
DATA_FILE="${DATA_FILE:?Set DATA_FILE to an M3-Agent JSONL manifest}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-_audio_only}
INTERMEDIATE_SUFFIX=${INTERMEDIATE_SUFFIX:-}

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${ENV_PY}" -m m3_agent.memorization_memory_graphs_audio_only \
  --data_file "${DATA_FILE}" \
  --output_suffix "${OUTPUT_SUFFIX}" \
  --intermediate_suffix "${INTERMEDIATE_SUFFIX}"
