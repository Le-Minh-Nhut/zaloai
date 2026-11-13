#!/usr/bin/env bash
# LifeMate inference entrypoint.
#
# Runs exactly one offline inference job. No pip install, no network access, no
# shell interpolation of untrusted data. Every value can be overridden by an
# environment variable so the same image serves base/full/lora checkpoints.
#
#   SPLIT        dataset split        (default: public_test)
#   DATA_ROOT    dataset root        (default: /data)
#   MODEL_MODE   base|full|lora      (default: full)
#   CHECKPOINT   checkpoint dir      (default: /models/ckpt)
#   OUTPUT       submission path     (default: /outputs/submission.jsonl)
#   ASR          gipformer|whisper|none|oracle (default: gipformer; oracle needs ALLOW_ORACLE=1)
#   DEVICE       cuda|cpu            (default: cuda; use cpu for CPU image builds)
#   CONFIG       YAML config path    (optional)
#   HISTORY_TRANSCRIPTS
#                ASR transcripts of history turns (JSONL). A test split carries
#                history AUDIO only, so without this every prior turn renders
#                blank: measured on public_test, 419 of 634 samples lose their
#                history without it. Unset = blank history, which `infer.py`
#                refuses by default (exit 2); a diagnostic run opts out with
#                EXTRA_ARGS=--allow-missing-history.
#                Mount it read-only; a set-but-unreadable path exits 2.
#   EXTRA_ARGS   additional CLI args (optional, word-split on purpose)
set -euo pipefail

SPLIT="${SPLIT:-public_test}"
DATA_ROOT="${DATA_ROOT:-/data}"
MODEL_MODE="${MODEL_MODE:-full}"
CHECKPOINT="${CHECKPOINT:-/models/ckpt}"
OUTPUT="${OUTPUT:-/outputs/submission.jsonl}"
ASR="${ASR:-gipformer}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-}"
HISTORY_TRANSCRIPTS="${HISTORY_TRANSCRIPTS:-}"
# Offline by default: the container must not reach the network at run time.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# An explicit command wins over the built-in inference default, so the same image
# also runs training (``docker run IMAGE python scripts/train.py --config ...``,
# which is exactly the command cloud/sandbox_runner.py submits). Without this the
# container would silently run inference and ignore the requested job.
if [[ "$#" -gt 0 ]]; then
    echo "lifemate: running the command given on the command line"
    exec "$@"
fi

cmd=(python /app/scripts/infer.py
    --split "$SPLIT"
    --data-root "$DATA_ROOT"
    --model-mode "$MODEL_MODE"
    --checkpoint "$CHECKPOINT"
    --output "$OUTPUT"
    --asr "$ASR"
    --device "$DEVICE")
if [[ -n "$CONFIG" ]]; then
    cmd+=(--config "$CONFIG")
fi

if [[ -d "/models/asr" && -f "/models/asr/tokens.txt" ]]; then
    cmd+=(--override "asr.model_dir=/models/asr")
fi
# History transcripts: an explicitly configured file must be usable. Degrading to
# blank history would silently drop 66.1% of public_test prompts' prior turns, so a
# set-but-unreadable path is a hard error rather than a warning.
if [[ -n "$HISTORY_TRANSCRIPTS" && -r "$HISTORY_TRANSCRIPTS" ]]; then
    cmd+=(--history-transcripts "$HISTORY_TRANSCRIPTS")
elif [[ -n "$HISTORY_TRANSCRIPTS" ]]; then
    echo "ERROR: HISTORY_TRANSCRIPTS set but not readable: $HISTORY_TRANSCRIPTS" >&2
    exit 2
fi
if [[ "$ASR" == "oracle" ]]; then
    # Oracle transcripts are evaluation-only; require an explicit opt-in AND a
    # labeled split, otherwise refuse before the model is ever loaded.
    if [[ "${ALLOW_ORACLE:-0}" != "1" ]]; then
        echo "ERROR: ASR=oracle requires ALLOW_ORACLE=1 (evaluation only, never a submission)" >&2
        exit 2
    fi
    if [[ ! -r "${TRANSCRIPTS:-/nonexistent}" ]]; then
        echo "ERROR: ASR=oracle requires TRANSCRIPTS pointing at an existing file" >&2
        exit 2
    fi
    cmd+=(--allow-oracle --transcripts "$TRANSCRIPTS")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # Intentional word-splitting: EXTRA_ARGS is a list of CLI flags.
    read -r -a extra <<< "$EXTRA_ARGS"
    cmd+=("${extra[@]}")
fi

echo "lifemate: split=$SPLIT model_mode=$MODEL_MODE asr=$ASR device=$DEVICE output=$OUTPUT"
echo "lifemate: offline (HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE)"
exec "${cmd[@]}"
