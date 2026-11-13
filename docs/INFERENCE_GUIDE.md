# Inference guide

How to produce `submission.jsonl` for Zalo AI Challenge 2026 Track 1 (LifeMate).

> Model weights, a CUDA GPU and W&B credentials are **not present in this
> environment**; the real dataset (`./data`: `train` 1545 / `public_test` 634) and
> its per-split `zaic_tools` SDK **are**. The commands below are the exact ones to
> run on a scoring machine; the mock variants at the end are what has been
> exercised end to end here.

## 0. Submission format

One JSON object per line, one line per sample, in the dataset's order:

```json
{"id": "sample-1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}], "state_hash": "9f2c…"}
```

* `tool_calls` — the environment's full call log, **including failed calls**
  (`{"name": …, "arguments": …, "ok": false, "error": …}` entries stay in).
* `state_hash` — `env.get_state_hash()` after the last call.
* Both fields are copied from the environment, never reconstructed.

`scripts/infer.py` validates its own output and exits `4` if anything is wrong.

## 1. Base model (no fine-tuning)

```bash
python scripts/infer.py \
    --split public_test --data-root ./data \
    --output submission.jsonl \
    --history-transcripts outputs/asr/history_transcripts.jsonl \
    --model-mode base \
    --asr gipformer
```

## 2. Full fine-tuned checkpoint (competition setup)

```bash
python scripts/infer.py \
    --split public_test --data-root ./data \
    --output submission.jsonl \
    --history-transcripts outputs/asr/history_transcripts.jsonl \
    --model-mode full \
    --checkpoint outputs/checkpoints/full/step-<n> \
    --asr gipformer
```

`--checkpoint` takes the `step-<n>` directory that `scripts/train.py` writes
(`<output_dir>/<run_name>/step-<n>`, run name = tuning mode). The run root
(`outputs/checkpoints/full`) holds no weights, so loading it fails.

Optional merge/sanity export first (`export_model.py` does accept the run root
and resolves the newest `step-<n>` inside it):

```bash
python scripts/export_model.py --ckpt outputs/checkpoints/full --output outputs/export
```

## 3. LoRA adapter

```bash
# run directly against the adapter (base model + adapter at load time)
python scripts/infer.py --split public_test --data-root ./data --output submission.jsonl \
    --history-transcripts outputs/asr/history_transcripts.jsonl \
    --model-mode lora --checkpoint outputs/checkpoints/lora/step-<n> --asr gipformer

# or merge first, then run as a full checkpoint
python scripts/export_model.py --ckpt outputs/checkpoints/lora/step-<n> --output outputs/export_lora --merge-lora
python scripts/infer.py --split public_test --data-root ./data --output submission.jsonl \
    --history-transcripts outputs/asr/history_transcripts.jsonl \
    --model-mode full --checkpoint outputs/export_lora --asr gipformer
```

## 4. Useful flags

| Flag | Effect |
|---|---|
| `--limit N` | first N samples only (smoke runs) |
| `--max-steps N` | agent step budget (default from config: 8) |
| `--time-budget S` | per-sample wall clock budget (default 120 s) |
| `--debug-dump DIR` | per-sample JSON (prompts/observations, PII stripped) |
| `--resume` | skip ids already present in `--output`; keeps existing rows |
| `--config FILE` | YAML config; `--override key=value` for single values |
| `--device cuda:1` | torch device |
| `--history-transcripts FILE` | ASR transcripts of history user turns; a test split stores history audio only, so without it those prior turns stay blank — and the run then **exits 2** rather than degrading |
| `--allow-missing-history` | diagnostic only: accept blank history turns (never for a submission) |
| `--allow-empty-transcripts` | diagnostic only; never for a competition submission |

`--asr` defaults to `gipformer` (`configs/inference.yaml`, `asr.backend`), so the
examples above can drop the flag — it is shown only to be explicit.

### Optional comparison backend (not the default)

PhoWhisper is kept for comparison only; the default and competition backend is
Gipformer. It needs `pip install -e '.[asr-whisper]'` and is never used unless
asked for explicitly:

```bash
python scripts/infer.py --split public_test --data-root ./data \
    --output /tmp/submission_whisper.jsonl \
    --history-transcripts outputs/asr/history_transcripts.jsonl \
    --model-mode full --checkpoint outputs/checkpoints/full/step-<n> \
    --asr whisper          # comparison run only; do not submit these numbers
```

Exit codes: `0` ok · `2` dataset/environment missing · `3` model load failure ·
`4` submission validation failure.

## 5. Offline / air-gapped runs

No network call is made by inference. Set the standard HF switches (they are
honoured, and `HF_HUB_OFFLINE=1` also sets `local_files_only=True` in the loader):

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1            # transformers' own switch; not read by us
export HF_DATASETS_OFFLINE=1
```

`TRANSFORMERS_OFFLINE` is a transformers-library variable, not one this repo
reads — set it anyway so no library tries a hub call. We read `HF_HUB_OFFLINE`
in `lifemate.models.loader.is_local_or_offline()`, `lifemate.asr.gipformer_backend`, and in the whisper backend.

## 6. Docker

Build CUDA 12.4 default image:
```bash
docker build -f docker/Dockerfile -t lifemate-infer .
```

Build CPU-only image:
```bash
docker build -f docker/Dockerfile \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
    --platform linux/amd64 \
    -t lifemate-infer-cpu .
```

Run with GPU (CUDA default):
```bash
docker run --rm --gpus all --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/full/step-<n>":/models/ckpt:ro \
    -v "$PWD/outputs":/outputs \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -e SPLIT=public_test -e MODEL_MODE=full -e ASR=gipformer -e DEVICE=cuda \
```

Run with CPU:
```bash
docker run --rm --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/full/step-<n>":/models/ckpt:ro \
    -v "$PWD/outputs":/outputs \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -e SPLIT=public_test -e MODEL_MODE=full -e ASR=gipformer -e DEVICE=cpu \
```

`--network none` is the intended mode: the image installs nothing at run time and relies on baked-in offline environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
## 7. Mock self-test (no dataset, no model, no GPU)

The mock environment proves the plumbing only. It is refused on the official
output path and prints a banner:

```bash
python scripts/infer.py --split public_test --data-root tests/fixtures/synthetic_lifemate \
    --use-mock-env --generator dumpscript:/tmp/script.json \
    --limit 3 --output /tmp/mock_submission.jsonl --asr none
```

`/tmp/script.json` maps sample ids to raw model outputs (a documented test hook):

```json
{"test_single_tool": ["<action>{\"type\": \"tool_call\", \"name\": \"smart_home.device_set\", \"arguments\": {\"device_id\": \"light1\", \"on\": false}}</action>",
                      "<action>{\"type\": \"finish\"}</action>"]}
```

`--generator` without `--use-mock-env` is refused; `--use-mock-env` is refused
whenever the output resolves to `outputs/submission.jsonl` — with the default
`--output` and with an explicit one (exit `4`).

## 8. Oracle ASR (evaluation only)

`--asr oracle` requires `--allow-oracle` **and** a labeled split
(`train`, `validation`, `val`, `dev`). On `public_test`/`private_test` it is an
error: those splits have no gold transcript, so oracle would fabricate input.

```bash
python scripts/infer.py --split train --data-root ./data --asr oracle \
    --allow-oracle --transcripts outputs/transcripts/train.jsonl \
    --output /tmp/train_oracle.jsonl
```

## 9. Evaluation

```bash
# teacher-forced over SFT validation records (from scripts/prepare_dataset.py)
python scripts/evaluate.py --validation-only --model-mode full \
    --checkpoint outputs/checkpoints/full/step-<n> --predictions outputs/datasets/train/val.jsonl

# closed loop against the official simulator
python scripts/evaluate.py --closed-loop --split train --limit 20

# pipeline self-test (marked MOCK ENV / SYNTHETIC in every artifact)
python scripts/evaluate.py --closed-loop --split train --use-mock-env --limit 5
```

Artifacts: `outputs/evaluation/{metrics.json,metrics.md,predictions.jsonl,error_analysis.json,comparisons.md}`.
`final_state_correctness` is always `{"available": false}` — the official
scoring tooling is not available here.

### ASR evaluation

ASR scoring is reachable from three commands. None of them fabricates numbers:
with no predictions file the ASR block stays `available: false`.

```bash
# 1. transcribe a split, write predictions + a metrics sidecar
python3 scripts/transcribe_split.py --split train --data-root ./data \
    --output outputs/asr/train_predictions.jsonl

# 2. score any prediction JSONL inside the EDA report
python3 scripts/run_eda.py --data-root ./data --split train \
    --asr-predictions outputs/asr/train_predictions.jsonl --output outputs/eda

# 3. append an `asr` block to the evaluation artifacts
python3 scripts/evaluate.py --validation-only --predictions outputs/datasets/train/val.jsonl \
    --asr-predictions outputs/asr/train_predictions.jsonl
```

Prediction JSONL row: `{"id", "reference", "hypothesis", "latency_s", "error"}`
(`ref`/`hyp`/`text` are accepted as aliases). `scripts/transcribe_split.py`
writes `raw_hypothesis` (verbatim recognizer text) and `hypothesis` (the
LLM-facing `to_llm_text` view), plus `<output>.metrics.json`.
Exit codes: `0` ok · `2` dataset or backend unavailable (no file written) ·
`3` every transcription failed.

`transcribe_split.py` flags: `--split` `--data-root` `--output` `--config`
`--limit` `--device` `--quantization {int8,fp32}` `--model-dir` `--overwrite`.

Both reports separate the two normalization paths: WER/CER are shown raw and
after `for_wer`, together with per-class critical-token diagnostics
(numbers, times, money, entities, addresses, English loanwords) and a magnitude
drift count. **ASR WER/CER is independent of final-state correctness**: a
transcript score says nothing about whether the agent's tool calls were right,
and `final_state_correctness` still has to come from the official scorer.

Predictions passed with `--asr-predictions` are *file-supplied* and are not a
benchmark result; references must come from a labeled split.

### CPU tiny-model validation

To verify the full generation and evaluation pipeline on CPU with a local random causal LM (no HF download, no GPU):

```bash
# 0. Build the tiny model and tokenizer in /tmp/tiny_lifemate (run once)
python3 -c "from pathlib import Path; from tests.test_tiny_e2e import make_tiny_model_and_tokenizer; make_tiny_model_and_tokenizer(Path('/tmp/tiny_lifemate'))"

# 1. End-to-end inference against mock environment (2 samples)
python scripts/infer.py --split public_test --data-root tests/fixtures/synthetic_lifemate \
    --use-mock-env --model-mode base --checkpoint /tmp/tiny_lifemate --asr none --limit 2 \
    --output /tmp/tiny_submission.jsonl --device cpu

# 2. Teacher-forced evaluation over validation records
python scripts/evaluate.py --validation-only --predictions /tmp/sft_tiny/train/val.jsonl \
    --checkpoint /tmp/tiny_lifemate --limit 2 --output /tmp/tiny_eval --device cpu
```

Results: `tiny_submission.jsonl` passes `validate_submission`; `metrics.json` records honest evaluation metrics with `final_state_correctness.available: false`.
*Coverage limit*: Proves runtime plumbing, action parsing, stopping policies, submission validation and metrics calculation work on CPU. Does not evaluate task performance or task accuracy.

## 10. Verification gate

```bash
python scripts/verify_all.py            # table of PASS/FAIL/SKIP/BLOCKED
python scripts/verify_all.py --gates A,B,J
```

Writes `outputs/verification/verify_all.json` and `.md`. Exit `1` on any FAIL.
BLOCKED = cannot be checked here (GPU, cloud submit, docker, official final-state scorer);
the real dataset and the per-split `zaic_tools` SDK are present and probe as verified.

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Dataset not found.` + exit 2 | split directory/JSONL missing | extract the dataset under `--data-root` |
| `zaic_tools not installed` + exit 2 | official simulator absent | install the organizers' SDK; use `--use-mock-env` only for tests |
| `requires --checkpoint` + exit 3 | `--model-mode full/lora` without a path | pass `--checkpoint` |
| `load_model_and_tokenizer requires torch and transformers` + exit 3 | CPU-only box | `pip install -e '.[train]'` |
| `submission validation: …` + exit 4 | row/order/type problem | see the printed line; re-run without `--resume` for a clean file |
| `--asr oracle is forbidden for split …` | oracle on a test split | use `--asr gipformer` |
| empty `tool_calls` for every sample | model never emitted a valid action | inspect `--debug-dump` output; check the chat template |
| `--generator is a TEST HOOK` | scripted generator outside a mock run | add `--use-mock-env` (tests only) |
