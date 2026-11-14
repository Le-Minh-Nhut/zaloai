#!/usr/bin/env bash
# LifeMate inference entrypoint.
#
# Runs exactly one offline inference job. No pip install, no network access, no
# shell interpolation of untrusted data. Every value can be overridden by an
# environment variable so the same image serves base/full/lora checkpoints.
#
#   SPLIT        dataset split        (default: public_test)
#   DATA_ROOT    dataset root        (default: /data)
#   MODEL_NAME   base-model IDENTITY (default: openbmb/MiniCPM5-2B; the supported
#                alternative is Qwen/Qwen3.5-4B). Identity only: it selects the
#                chat template, terminator, stop ids and token budget, and it is
#                the weight source for `base` (no checkpoint) and `lora` (the
#                frozen base the adapter is applied to). It is NOT the loading
#                mode and NOT the fine-tune location -- see MODEL_MODE/CHECKPOINT.
#   MODEL_MODE   base|full|lora      (default: full)
#   CHECKPOINT   checkpoint dir      (default: /models/ckpt)
#   MODEL_ROOT   mount root for pre-mounted weights (default: /models)
#   HF_HOME      pre-mounted Hugging Face cache (default: /models/hf-cache)
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
MODEL_NAME="${MODEL_NAME:-openbmb/MiniCPM5-2B}"
CHECKPOINT="${CHECKPOINT:-/models/ckpt}"
OUTPUT="${OUTPUT:-/outputs/submission.jsonl}"
ASR="${ASR:-gipformer}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${CONFIG:-}"
HISTORY_TRANSCRIPTS="${HISTORY_TRANSCRIPTS:-}"
# Offline by default: the container must not reach the network at run time.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Hub-id resolution for MODEL_NAME reads this pre-mounted cache (HF_HOME/hub).
# Pinned so a hub id never silently resolves against an unrelated ~/.cache: with
# HF_HUB_OFFLINE=1 the loader passes local_files_only=True, so a cache miss is a
# hard error, never a download.
MODEL_ROOT="${MODEL_ROOT:-/models}"
export HF_HOME="${HF_HOME:-$MODEL_ROOT/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# An explicit command wins over the built-in inference default, so the same image
# also runs training (``docker run IMAGE python scripts/train.py --config ...``,
# which is exactly the command cloud/sandbox_runner.py submits). Without this the
# container would silently run inference and ignore the requested job.
if [[ "$#" -gt 0 ]]; then
    echo "lifemate: running the command given on the command line"
    exec "$@"
fi

# --- offline weight preflight ------------------------------------------------- #
# The scoring host has no network, so every weight source must already be mounted.
# Resolve it here and exit 2 with the mount to add, rather than letting
# `from_pretrained` fail deep inside the run (HF_HUB_OFFLINE=1 makes the loader
# pass local_files_only=True, so a miss is a hard failure, never a download).
#
# Gated on the mount root existing: the image always creates /models, so a real
# container run always enforces it; a bare host checkout (no /models) skips it and
# says so, which keeps the entrypoint usable outside a container.
resolve_local_model() {
    # A directory as-is, else the HF cache snapshot for a hub id.
    local ref="$1" slug cand
    if [[ -d "$ref" ]]; then printf '%s\n' "$ref"; return 0; fi
    slug="models--${ref//\//--}"
    for cand in "$HF_HUB_CACHE/$slug"/snapshots/*; do
        if [[ -d "$cand" ]]; then printf '%s\n' "$cand"; return 0; fi
    done
    return 1
}

has_tokenizer_files() {
    local dir="$1"
    [[ -f "$dir/tokenizer_config.json" || -f "$dir/tokenizer.json" \
        || -f "$dir/vocab.json" || -f "$dir/spiece.model" ]]
}

has_weights() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    compgen -G "$dir/*.safetensors" >/dev/null && return 0
    compgen -G "$dir/pytorch_model*.bin" >/dev/null && return 0
    return 1
}

ckpt_dir=""
if [[ -n "$CHECKPOINT" && -d "$CHECKPOINT" ]]; then
    ckpt_dir="$CHECKPOINT"
fi

# Called last, right before exec: the argument-only checks (history transcripts,
# oracle opt-in) keep their documented error precedence over resource checks.
check_local_weights() {
    # Mirrors scripts/infer.py: `base` loads CHECKPOINT when one exists (else the
    # MODEL_NAME weights), `lora` loads the MODEL_NAME base + the adapter in
    # CHECKPOINT, `full` loads CHECKPOINT. The tokenizer comes from CHECKPOINT when
    # it carries one, else from MODEL_NAME.
    case "$MODEL_MODE" in
        base) if [[ -n "$ckpt_dir" ]]; then weights_ref="$ckpt_dir"; else weights_ref="$MODEL_NAME"; fi ;;
        lora) weights_ref="$MODEL_NAME" ;;
        full) weights_ref="$ckpt_dir" ;;
        *)
            echo "ERROR: MODEL_MODE must be base|full|lora, got '$MODEL_MODE'" >&2
            exit 2
            ;;
    esac
    if [[ -n "$ckpt_dir" ]] && has_tokenizer_files "$ckpt_dir"; then
        tokenizer_ref="$ckpt_dir"
    else
        tokenizer_ref="$MODEL_NAME"
    fi

    if [[ ! -d "$MODEL_ROOT" ]]; then
        # The image always creates $MODEL_ROOT, so this only skips on a bare host
        # checkout. A container run can never take this branch.
        echo "lifemate: WARNING: $MODEL_ROOT absent -- skipping the offline weights preflight" >&2
        return 0
    fi
    if [[ -z "$weights_ref" ]]; then
        echo "ERROR: MODEL_MODE=$MODEL_MODE needs a checkpoint directory but CHECKPOINT=$CHECKPOINT does not exist." >&2
        echo "       Mount one read-only: -v <ckpt-dir>:/models/ckpt:ro (or set -e CHECKPOINT=/models/<name>)." >&2
        exit 2
    fi
    if ! weights_dir="$(resolve_local_model "$weights_ref")"; then
        echo "ERROR: no local weights for '$weights_ref' (MODEL_MODE=$MODEL_MODE, MODEL_NAME=$MODEL_NAME)." >&2
        echo "       Nothing is downloaded at scoring time. Provide the weights by mounting either:" >&2
        echo "         a) a local directory:  -v <weights-dir>:/models/<name>:ro -e MODEL_NAME=/models/<name>" >&2
        echo "         b) the HF cache:       -v <hf-cache>:/models/hf-cache:ro  (expects hub/models--${MODEL_NAME//\//--}/snapshots/<rev>/)" >&2
        exit 2
    fi
    if ! has_weights "$weights_dir"; then
        echo "ERROR: '$weights_ref' resolved to '$weights_dir' but it holds no config.json + *.safetensors/pytorch_model*.bin." >&2
        echo "       The weights are incomplete or the mount points at the wrong level." >&2
        exit 2
    fi
    if [[ "$MODEL_MODE" == "lora" && ! -f "$ckpt_dir/adapter_config.json" ]]; then
        echo "ERROR: MODEL_MODE=lora needs a PEFT adapter in CHECKPOINT=$CHECKPOINT (adapter_config.json absent)." >&2
        exit 2
    fi
    if ! tokenizer_dir="$(resolve_local_model "$tokenizer_ref")" || ! has_tokenizer_files "$tokenizer_dir"; then
        echo "ERROR: no local tokenizer for '$tokenizer_ref' (MODEL_NAME=$MODEL_NAME, MODEL_MODE=$MODEL_MODE)." >&2
        echo "       Mount the tokenizer files next to the weights (tokenizer_config.json/tokenizer.json)," >&2
        echo "       or pass -e MODEL_NAME=/models/<name> pointing at a complete directory." >&2
        exit 2
    fi
}

cmd=(python /app/scripts/infer.py
    --split "$SPLIT"
    --data-root "$DATA_ROOT"
    --model "$MODEL_NAME"
    --model-mode "$MODEL_MODE"
    --output "$OUTPUT"
    --asr "$ASR"
    --device "$DEVICE")
# CHECKPOINT is a location, not an identity. For `base` it wins over MODEL_NAME
# when the directory exists (scripts/infer.py: `model_source = checkpoint if
# (mode == "base" and checkpoint) else cfg.model.name`), so a base run with
# nothing mounted must NOT pass the default path: it would try to load
# /models/ckpt instead of the MODEL_NAME weights.
if [[ -n "$ckpt_dir" ]]; then
    cmd+=(--checkpoint "$CHECKPOINT")
elif [[ "$MODEL_MODE" == "base" ]]; then
    echo "lifemate: CHECKPOINT=$CHECKPOINT absent; base mode loads MODEL_NAME weights" >&2
else
    cmd+=(--checkpoint "$CHECKPOINT")
fi
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

echo "lifemate: split=$SPLIT model_mode=$MODEL_MODE model=$MODEL_NAME asr=$ASR device=$DEVICE output=$OUTPUT"
echo "lifemate: offline (HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE)"
check_local_weights
exec "${cmd[@]}"
