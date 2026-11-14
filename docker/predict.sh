#!/usr/bin/env bash
# Organizer interface. Artifacts must be present in the image, never downloaded.
set -euo pipefail
export DATA_ROOT=/data
if [[ -z "${SPLIT:-}" ]]; then
  SPLIT="$(python3 - <<'PY'
from pathlib import Path
root = Path('/data')
splits = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / (p.name + '.jsonl')).is_file() and (p / (p.name + '_tools_catalog.json')).is_file())
if len(splits) != 1:
    raise SystemExit('Expected exactly one dataset split under /data; set SPLIT explicitly otherwise')
print(splits[0])
PY
)"
fi
export SPLIT
export MODEL_MODE="${MODEL_MODE:-lora}" MODEL_NAME="${MODEL_NAME:-/models/base}" CHECKPOINT="${CHECKPOINT:-/models/adapter}" ASR_MODEL_DIR="${ASR_MODEL_DIR:-/models/asr}"
export OUTPUT="${RAW_OUTPUT:-/result/.lifemate_raw.jsonl}"
FINAL_OUTPUT="${FINAL_OUTPUT:-/result/submission.jsonl}"
mkdir -p /result
/app/docker/entrypoint.sh
PYTHONPATH=/app/src python3 /code/export_timed_submission.py "$OUTPUT" "$OUTPUT.progress.json" "$FINAL_OUTPUT"
