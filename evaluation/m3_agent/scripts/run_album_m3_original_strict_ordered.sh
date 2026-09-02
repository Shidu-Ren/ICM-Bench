#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY_BIN="${PY_BIN:-python}"
GPU_LIST="${GPU_LIST:-0}"
BASE_ID="${BASE_ID:-icm_bench}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${BASE_ID}_m3_original_strict_ordered_${RUN_TS}}"

ICM_BENCH_ROOT="${ICM_BENCH_ROOT:?Set ICM_BENCH_ROOT to the extracted Hugging Face dataset root}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/data/icm_bench_runs}"
SOURCE_DIR="${SOURCE_DIR:-${ICM_BENCH_ROOT}/videos}"
MEM_PATH="${MEM_PATH:-${WORK_DIR}/memory_graphs/${BASE_ID}_m3_original.pkl}"
INTERMEDIATE_DIR="${INTERMEDIATE_DIR:-${WORK_DIR}/intermediate_outputs/${BASE_ID}_m3_original}"
DATA_JSONL="${DATA_JSONL:-${WORK_DIR}/${BASE_ID}_m3_original.jsonl}"
ANNOTATION_IN="${ANNOTATION_IN:-${ICM_BENCH_ROOT}/annotations/qa_test.json}"
ANNOTATION_OUT="${ANNOTATION_OUT:-${WORK_DIR}/${BASE_ID}_m3_original_qa.json}"
RESULT_NAME="${RESULT_NAME:-${BASE_ID}_m3_original}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/${BASE_ID}_m3_original_strict_ordered_latest.txt}"
FRESH="${FRESH:-0}"
RUN_EVAL="${RUN_EVAL:-1}"

cd "${REPO_ROOT}"
if ! compgen -G "${SOURCE_DIR}/*.mp4" > /dev/null; then
  echo "No MP4 files found under ${SOURCE_DIR}. Extract videos.tar into ICM_BENCH_ROOT first." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "$(dirname "${MEM_PATH}")" "$(dirname "${DATA_JSONL}")" "${INTERMEDIATE_DIR}" "$(dirname "${ANNOTATION_OUT}")"

# Only keep the path setup needed to import local M3Agent modules and SpeakerLab.
# Deliberately unset every optional compatibility override so Qwen/video/audio
# behavior falls back to the original code defaults.
SPEAKERLAB_ROOT="${SPEAKERLAB_ROOT:-${REPO_ROOT}/external/3D-Speaker}"
export PYTHONPATH="${SPEAKERLAB_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset M3_QWEN_USE_AUDIO_IN_VIDEO || true
unset M3_QWEN_MAX_NEW_TOKENS || true
unset M3_QWEN_VIDEO_MODE || true
unset M3_MEMORY_GENERATION_BACKEND || true
unset M3_VOICE_METADATA_PATH || true
unset M3_MAX_CLIPS_PER_RUN || true
unset M3_PROCESS_VIDEO_FPS || true
unset M3_INCLUDE_BASE64_VIDEO || true

if [[ "${FRESH}" == "1" ]]; then
  rm -f "${MEM_PATH}"
fi

"${PY_BIN}" - <<PY
import json
sample = {
    "id": "${BASE_ID}_fullvideo_m3_original_strict_ordered",
    "video_path": "${SOURCE_DIR}",
    "clip_path": "${SOURCE_DIR}",
    "mem_path": "${MEM_PATH}",
    "intermediate_outputs": "${INTERMEDIATE_DIR}",
}
with open("${DATA_JSONL}", "w", encoding="utf-8") as f:
    f.write(json.dumps(sample, ensure_ascii=False) + "\\n")
print(json.dumps(sample, ensure_ascii=False, indent=2))
PY

{
  echo "run_name=${RUN_NAME}"
  echo "started_at=$(date -Is)"
  echo "repo=${REPO_ROOT}"
  echo "gpus_visible=${GPU_LIST}"
  echo "mode=m3_original_strict_ordered_single_graph"
  echo "qwen_use_audio_in_video=original_default"
  echo "source_dir=${SOURCE_DIR}"
  echo "data_jsonl=${DATA_JSONL}"
  echo "intermediate_dir=${INTERMEDIATE_DIR}"
  echo "mem_path=${MEM_PATH}"
  echo "annotation_out=${ANNOTATION_OUT}"
  echo "result=data/results/${RESULT_NAME}.jsonl"
} > "${STATUS_FILE}"

echo "[master] Strict original ordered M3 memorization starts: ${SOURCE_DIR} -> ${MEM_PATH}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PY_BIN}" -u m3_agent/memorization_memory_graphs.py \
  --data_file "${DATA_JSONL}"

echo "memorization_finished_at=$(date -Is)" >> "${STATUS_FILE}"

"${PY_BIN}" scripts/retarget_album_annotation_mem_path.py \
  --input "${ANNOTATION_IN}" \
  --output "${ANNOTATION_OUT}" \
  --mem-path "${MEM_PATH}"

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "status=memorization_complete_eval_skipped" >> "${STATUS_FILE}"
  echo "[master] Memorization complete; RUN_EVAL=${RUN_EVAL}, skipping control/eval"
  exit 0
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
TP_SIZE="${TP_SIZE:-${#GPUS[@]}}"
if (( TP_SIZE < 1 )); then
  TP_SIZE=1
fi

echo "[master] Strict original graph complete. Running M3 control/eval with TP=${TP_SIZE}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
M3_EVAL_MODEL="${M3_EVAL_MODEL:-models/gemini-3-flash-preview}" \
"${PY_BIN}" -u m3_agent/control.py \
  --data_file "${ANNOTATION_OUT}" \
  --tensor_parallel_size "${TP_SIZE}" \
  --output_name "${RESULT_NAME}"

echo "finished_at=$(date -Is)" >> "${STATUS_FILE}"
echo "status=complete" >> "${STATUS_FILE}"
echo "[master] Complete"
