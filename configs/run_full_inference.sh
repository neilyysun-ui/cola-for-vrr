#!/usr/bin/env bash
set -euo pipefail

: "${QUESTIONS_JSON:?set QUESTIONS_JSON}"
: "${VIDEO_DIR:?set VIDEO_DIR}"
: "${OUTPUT_JSON:?set OUTPUT_JSON}"

python run_final.py \
  --questions-json "${QUESTIONS_JSON}" \
  --video-dir "${VIDEO_DIR}" \
  --output-json "${OUTPUT_JSON}"
