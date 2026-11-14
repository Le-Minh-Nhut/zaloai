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
#   ASR_MODEL_ID gipformer checkpoint id (default: g-group-ai-lab/gipformer-68M-rnnt).
#                Any other value must come with asr.gipformer_v15 enabled: it is
#                forwarded as `--override asr.model_id=<id>`, and refusing the pair
#                keeps a non-default ASR checkpoint from silently moving the baseline.
#   ASR_MODEL_DIR
#                ONNX directory for the gipformer backend (default: /models/asr).
#                Forwarded as `--override asr.model_dir=<dir>` only when the
#                directory holds tokens.txt, as before; an EXPLICIT value without
#                one exits 2 instead of silently loading something else.
#   EXTRA_ARGS   additional CLI args (optional, word-split on purpose)
#
# Optional features. Each switch below is OFF unless its variable is set to a truthy
# value. The entrypoint never enables, rewrites, unsets or case-folds one: `python`
# reads the same names straight from os.environ
# (src/lifemate/features/overrides.py). They are validated here so a typo fails
# before any weight is touched:
#
#   LIFEMATE_FEATURE_ASR_GIPFORMER_V15              asr.gipformer_v15
#   LIFEMATE_FEATURE_ASR_MULTI_ASR                  asr.multi_asr
#   LIFEMATE_FEATURE_ASR_NBEST                      asr.nbest
#   LIFEMATE_FEATURE_AUDIO_RMS_NORMALIZATION        audio.rms_normalization
#   LIFEMATE_FEATURE_AUDIO_VAD                      audio.vad
#   LIFEMATE_FEATURE_DECODING_CONSTRAINED           decoding.constrained
#   LIFEMATE_FEATURE_POSTPROCESSING_ENHANCED_SCHEMA_GUARDRAIL
#                                                   postprocessing.enhanced_schema_guardrail
#   LIFEMATE_FEATURE_PROMPT_ITN                     prompt.itn
#   LIFEMATE_FEATURE_PROMPT_NBEST_FORMAT            prompt.nbest_format
#   LIFEMATE_FEATURE_TRAINING_NOISY_ASR             training.noisy_asr
#   LIFEMATE_FEATURE_TRAINING_SLOT_MASKING          training.slot_masking
#   LIFEMATE_FEATURE_TRAINING_TOOL_DROPOUT          training.tool_dropout
#
# Truthy: 1|true|yes|on. Falsy: 0|false|no|off. Case-insensitive, exactly the
# TRUE_TEXT/FALSE_TEXT sets in overrides.py. Any other value, or any other name
# under the prefix, exits 2 (python would raise on both).
#
# Requires bash >= 4 for ${!prefix@} and ${var,,}; the image ships bash 5.1
# (ubuntu22.04).
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

# ASR checkpoint knobs. The default is the baseline the image was measured with
# (`GIPFORMER_SPECS[DEFAULT_SPEC_KEY]` in src/lifemate/asr/model_specs.py).
ASR_DEFAULT_MODEL_ID="g-group-ai-lab/gipformer-68M-rnnt"
ASR_MODEL_ID="${ASR_MODEL_ID:-$ASR_DEFAULT_MODEL_ID}"
# The default path is a convention, not a promise: an absent /models/asr is normal
# (ASR=none, or a host checkout) and only skips the override. An EXPLICIT path that
# holds no tokens.txt is a mount mistake and fails below.
asr_model_dir_explicit=0
if [[ -n "${ASR_MODEL_DIR-}" ]]; then
    asr_model_dir_explicit=1
fi
ASR_MODEL_DIR="${ASR_MODEL_DIR:-/models/asr}"

# Every optional feature, in alphabetical order (so the summary is sorted without
# `sort`). This list is the entrypoint's declaration of the ids it advertises;
# tests/test_docker_entrypoint.py pins it to lifemate.features.registry.FEATURES.
FEATURE_VARS=(
    LIFEMATE_FEATURE_ASR_GIPFORMER_V15
    LIFEMATE_FEATURE_ASR_MULTI_ASR
    LIFEMATE_FEATURE_ASR_NBEST
    LIFEMATE_FEATURE_AUDIO_RMS_NORMALIZATION
    LIFEMATE_FEATURE_AUDIO_VAD
    LIFEMATE_FEATURE_DECODING_CONSTRAINED
    LIFEMATE_FEATURE_POSTPROCESSING_ENHANCED_SCHEMA_GUARDRAIL
    LIFEMATE_FEATURE_PROMPT_ITN
    LIFEMATE_FEATURE_PROMPT_NBEST_FORMAT
    LIFEMATE_FEATURE_TRAINING_NOISY_ASR
    LIFEMATE_FEATURE_TRAINING_SLOT_MASKING
    LIFEMATE_FEATURE_TRAINING_TOOL_DROPOUT
)

feature_truthy() {
    # Same truth set as overrides.TRUE_TEXT; case-insensitive like `_read_bool`.
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
    esac
    return 1
}

validate_feature_env() {
    # Read-only pass over LIFEMATE_FEATURE_*: nothing is exported, unset or
    # case-folded, so `python` sees exactly what the operator passed. An unknown
    # name or an unreadable value is refused here (exit 2) rather than surfacing
    # as a traceback after the weights are loaded -- `parse_env_overrides` raises
    # on both.
    local name value lowered
    for name in "${!LIFEMATE_FEATURE_@}"; do
        value="${!name}"
        lowered="${value,,}"
        case "$lowered" in
            1|true|yes|on|0|false|no|off) ;;
            *)
                echo "ERROR: $name=$value is not a feature boolean." >&2
                echo "       Accepted values: 1|0|true|false|yes|no|on|off (case-insensitive)." >&2
                exit 2
                ;;
        esac
        if [[ " ${FEATURE_VARS[*]} " != *" $name "* ]]; then
            echo "ERROR: $name does not name a known feature." >&2
            echo "       Known feature variables: ${FEATURE_VARS[*]}" >&2
            exit 2
        fi
    done
}

feature_summary() {
    # One stderr line naming the enabled features; NONE when nothing is set.
    local name enabled=()
    for name in "${FEATURE_VARS[@]}"; do
        [[ -n "${!name-}" ]] || continue
        if feature_truthy "${!name}"; then
            enabled+=("$name")
        fi
    done
    if [[ "${#enabled[@]}" -eq 0 ]]; then
        echo "lifemate: features=NONE" >&2
    else
        local IFS=,
        echo "lifemate: features=${enabled[*]}" >&2
    fi
}

# Runs before the explicit-command branch: `python scripts/train.py` reads the same
# variables, so an invalid one must fail the same way for training and inference.
validate_feature_env

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

if [[ "$ASR_MODEL_ID" != "$ASR_DEFAULT_MODEL_ID" ]]; then
    # A non-default gipformer checkpoint is a different artifact set, not a
    # different path: selecting it is exactly what asr.gipformer_v15 means
    # (registry: "Decode with the Gipformer-1.5 checkpoint instead of
    # gipformer-68M-rnnt"). Refuse the id alone so an operator cannot move the
    # baseline by editing one variable.
    if ! feature_truthy "${LIFEMATE_FEATURE_ASR_GIPFORMER_V15:-0}"; then
        echo "ERROR: ASR_MODEL_ID=$ASR_MODEL_ID differs from the baseline $ASR_DEFAULT_MODEL_ID" >&2
        echo "       but LIFEMATE_FEATURE_ASR_GIPFORMER_V15 is not enabled." >&2
        echo "       A non-default ASR checkpoint silently changes the baseline; enable the" >&2
        echo "       feature with it: -e LIFEMATE_FEATURE_ASR_GIPFORMER_V15=1" >&2
        echo "       (or set features.asr.gipformer_v15_enabled=true in a --config YAML)." >&2
        exit 2
    fi
    cmd+=(--override "asr.model_id=$ASR_MODEL_ID")
fi

if [[ -d "$ASR_MODEL_DIR" && -f "$ASR_MODEL_DIR/tokens.txt" ]]; then
    cmd+=(--override "asr.model_dir=$ASR_MODEL_DIR")
elif [[ "$asr_model_dir_explicit" -eq 1 ]]; then
    echo "ERROR: ASR_MODEL_DIR=$ASR_MODEL_DIR holds no tokens.txt." >&2
    echo "       Mount the gipformer ONNX directory there (-v <dir>:/models/asr:ro) or unset" >&2
    echo "       ASR_MODEL_DIR to keep the /models/asr default." >&2
    exit 2
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
feature_summary
check_local_weights
exec "${cmd[@]}"
