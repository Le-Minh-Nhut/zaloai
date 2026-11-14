# LifeMate inference image (CPU-extensible, GPU runtime tag).
#
# Tag choice: nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
#   * CUDA 12.8 + cuDNN 9 runtime is the oldest major line whose torch wheels
#     (`TORCH_INDEX=.../whl/cu128`) still carry `sm_120` kernels. The target card
#     here is an RTX 5070 Ti = compute capability (12, 0) = `sm_120` (Blackwell);
#     `cu124` wheels contain no `sm_120` kernel at all, so an image pinned to
#     cu124 builds fine and then cannot launch a single GPU kernel on this host
#     (`torch.cuda.get_arch_list()` omits `sm_120`). The local `/tmp/cudaenv`
#     runs `torch 2.15.0+cu132`, which does include it.
#   * the *runtime* flavour (not devel) keeps the image ~1 GB smaller and we never
#     compile CUDA code.
#   * ubuntu22.04 has glibc 2.35, which satisfies the manylinux_2_28 wheels used
#     by torch/transformers and the Python 3.10+ interpreter of the base image.
#   * The `runtime` (not `base`) flavour is required because cuDNN is needed by
#     the ASR/whisper path.
#
# Design rules enforced here:
#   * two stages: `build` installs Python deps with a compiler toolchain, the
#     final stage copies only the resulting site-packages (no compilers, smaller
#     layer surface).
#   * NO pip install at run time and no network access in the entrypoint: the
#     container is meant to run with `--network none`.
#   * the base-model IDENTITY is explicit (MODEL_NAME, default openbmb/MiniCPM5-2B)
#     and independent of how it is loaded (MODEL_MODE) and where a fine-tune lives
#     (CHECKPOINT). Its weights are pre-mounted, never fetched.
#   * runs as an unprivileged user (uid 10001).
#   * HEALTHCHECK is intentionally absent: the workload is a one-shot batch job
#     with no listening port and no network.

# --------------------------------------------------------------------------- #
# Stage 1: build wheels for the CPU-side Python dependencies
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /build/pyproject.toml
COPY src /build/src

WORKDIR /build
# Optional extras: EMPTY by default, so the default image is the baseline and
# `pip install -e .` pulls in nothing an optional feature would need.
#   * A disabled feature must not be installed. Every optional backend is imported
#     lazily (src/lifemate/features/registry.py probes only enabled features), so
#     omitting a group breaks nothing until the operator turns the flag on -- and
#     turning it on without the group fails validation with a named reason instead
#     of importing something half-installed.
#   * No weights are downloaded at build time. `pip install` fetches wheels, never
#     model artifacts, and HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are runtime ENV here;
#     the model, ASR and feature artifacts are mounted at run time (docker/README.md).
#   * Groups are comma-separated pip extras from pyproject.toml, e.g.
#     --build-arg LIFEMATE_EXTRAS=asr-whisper or LIFEMATE_EXTRAS="asr,onnx".
#     Empty (the default) installs the baseline. Note: no feature flag needs an
#     extra of its own -- the 12 flags are satisfied by the baseline stack, the
#     `asr`/`asr-whisper` groups, or mounted artifacts (see pyproject.toml).
#   * Non-editable on purpose, like the baseline line: only /opt/venv is copied
#     into the runtime stage, so an editable install would point at /build/src,
#     which does not exist there (the runtime sets PYTHONPATH=/app/src instead).
ARG LIFEMATE_EXTRAS=""
# Coupling: base image tag nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 <-> torch cu128 index.
# For CPU-only image builds, pass: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
# Floors are LOAD-BEARING — do not relax them back to 4.4x/2.4 or the image builds fine and then
# cannot load either supported model:
#   * openbmb/MiniCPM5-2B (the default MODEL_NAME) ships `transformers_version: 5.6.2` in its
#     config.json;
#   * Qwen/Qwen3.5-4B needs the `qwen3_5` model classes, which simply do not exist at 4.44-4.5x:
#     `from_pretrained` fails on `model_type qwen3_5` before any weight is read.
# Verified interpreter of record: torch 2.15.0+cu132 / transformers 5.22.2 / peft 0.21.2.
# No bitsandbytes here on purpose: this image runs inference, not QLoRA training, and nothing on
# the inference path requests `quantization` (see scripts/infer.py / docker/entrypoint.sh), so a
# 4-bit training dependency would only add an unused CUDA extension.
# Note: sherpa-onnx wheels provide manylinux x86_64 and aarch64 binaries.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url "${TORCH_INDEX}" \
        "torch>=2.15" \
    && /opt/venv/bin/pip install \
        "transformers>=5.22" \
        "peft>=0.21" \
        "accelerate>=0.33" \
        "soundfile>=0.12" \
        "sherpa-onnx>=1.10" \
        "sherpa-onnx-core>=1.10" \
        "huggingface-hub>=0.20" \
        "pyyaml>=6" \
        "numpy>=1.26" \
    && if [ -n "${LIFEMATE_EXTRAS}" ]; then \
        /opt/venv/bin/pip install ".[${LIFEMATE_EXTRAS}]"; \
    else \
        /opt/venv/bin/pip install --no-deps .; \
    fi
# --------------------------------------------------------------------------- #
# Stage 2: runtime image
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/venv/bin:/usr/local/bin:/usr/bin:/bin" \
    PYTHONPATH=/app/src \
    SPLIT=public_test \
    MODEL_MODE=full \
    MODEL_NAME=openbmb/MiniCPM5-2B \
    MODEL_ROOT=/models \
    HF_HOME=/models/hf-cache \
    ASR=gipformer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 lifemate \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin lifemate

COPY --from=build /opt/venv /opt/venv
COPY --chown=lifemate:lifemate src /app/src
COPY --chown=lifemate:lifemate scripts /app/scripts
COPY --chown=lifemate:lifemate configs /app/configs
COPY --chown=lifemate:lifemate docker/entrypoint.sh /app/docker/entrypoint.sh

RUN chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /data /models /models/asr /models/hf-cache /history /outputs \
    && chown -R lifemate:lifemate /data /models /history /outputs /app

USER lifemate
WORKDIR /app

# Mount points (documented, not declared as VOLUME so `docker run -v` stays explicit):
#   /data        dataset root (read-only)
#   /models      checkpoint directory (read-only)
#   /models/asr  Gipformer ONNX model directory (read-only; whisper is optional host extra)
#   /models/hf-cache
#                Hugging Face cache holding the BASE model weights named by MODEL_NAME
#                (read-only; HF_HOME, so hub ids resolve to hub/models--<org>--<name>/
#                snapshots/<rev>/). Alternative: mount a plain weights directory and
#                point MODEL_NAME at it (-e MODEL_NAME=/models/MiniCPM5-2B).
#                Either way the weights are mounted, never downloaded: HF_HUB_OFFLINE=1
#                makes the loader pass local_files_only=True.
#   /history     history ASR transcripts FILE (read-only; mount the single jsonl, e.g.
#                -v "$PWD/outputs/asr/history_transcripts.jsonl":/history/history_transcripts.jsonl:ro
#                then pass -e HISTORY_TRANSCRIPTS=/history/history_transcripts.jsonl).
#                Deliberately NOT baked into the image: the file is regenerated per
#                dataset release (4790 rows / 1.4 MB today), so baking it would freeze
#                an artifact that must track the split it describes.
#   /outputs     submission + reports (read-write)

# Default job: the offline inference entrypoint. An explicit command
# (`docker run IMAGE python scripts/train.py ...`) replaces it, which is what the
# cloud job spec submits. Without an ENTRYPOINT the image had none, so the
# documented `docker run` started nothing.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
