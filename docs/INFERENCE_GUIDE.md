# Inference guide

How to produce `submission.jsonl` for Zalo AI Challenge 2026 Track 1 (LifeMate),
and how to run a single adapter or checkpoint outside the competition path.

The default model is `openbmb/MiniCPM5-2B` (weights present at
`models/MiniCPM5-2B`, 5,043,806,458 B). `Qwen/Qwen3.5-4B` is the supported
alternative. The real dataset (`./data`: `train` 1545 / `public_test` 634) and its
per-split `zaic_tools` SDK are present; no competition submission has been scored.

## 0. Submission format

One JSON object per line, one line per sample, in the dataset's order:

```json
{"id": "sample-1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}], "state_hash": "9f2c…"}
```

* `tool_calls` — the environment's full call log, **including failed calls**
  (`{"name": …, "arguments": …, "ok": false, "error": …}` entries stay in).
* `state_hash` — `env.get_state_hash()` after the last call.
* Both are copied from the environment, never reconstructed.

`scripts/infer.py` validates its own output and exits `4` if anything is wrong.

## 1. Choosing the model

`--model` overrides `model.name` (base model id **or** local path). The default
comes from `configs/inference.yaml`:

| Model | Selected by | Notes |
|---|---|---|
| `openbmb/MiniCPM5-2B` (default) | nothing, or `--model openbmb/MiniCPM5-2B`, or `--model models/MiniCPM5-2B` | 2,516,756,480 params, one 5.03 GB bf16 shard, local copy present |
| `Qwen/Qwen3.5-4B` | `--model Qwen/Qwen3.5-4B` | the alternative; needs its own weights and its own adapter |

The model name is also what picks the profile (prompt template kwargs,
supervised/stop terminator, decode defaults), so **always pass the base model
name that trained the adapter** — a checkpoint directory carries no family name
in its tokenizer metadata, and `infer.py` reads identity from `cfg.model.name`,
not from the checkpoint.

```bash
# default MiniCPM5-2B from the local directory (offline-friendly)
python3 scripts/infer.py --split public_test --data-root ./data \
    --output submission.jsonl --model models/MiniCPM5-2B --asr gipformer

# the Qwen alternative
python3 scripts/infer.py --split public_test --data-root ./data \
    --output submission.jsonl --model Qwen/Qwen3.5-4B --asr gipformer
```

## 2. Checkpoint modes

`--model-mode {base,full,lora}` (`inference.model_mode`):

| Mode | `--checkpoint` | What loads |
|---|---|---|
| `base` | omitted | the base model, no adapter (untrained policy) |
| `base` | given | that directory **is** the model (e.g. an exported/merged directory) |
| `full` | required | checkpoint directory holding full weights; a missing `--checkpoint` is a hard error |
| `lora` | required | LoRA adapter directory; the base comes from `model.name` |

```bash
# base model, no adapter
python3 scripts/infer.py --split public_test --data-root ./data \
    --output submission.jsonl --model-mode base --asr gipformer

# full fine-tuned checkpoint
python3 scripts/infer.py --split public_test --data-root ./data \
    --output submission.jsonl --model-mode full \
    --checkpoint outputs/checkpoints/full/step-<n> --asr gipformer

# LoRA adapter (base model + adapter at load time)
python3 scripts/infer.py --split public_test --data-root ./data \
    --output submission.jsonl --model-mode lora \
    --checkpoint outputs/checkpoints/lora/step-<n> --asr gipformer

# merge an adapter, then treat the result as a full checkpoint
python3 scripts/export_model.py --ckpt outputs/checkpoints/lora/step-<n> \
    --merge-lora --output outputs/export_lora
python3 scripts/infer.py --split public_test --data-root ./data \
    --output submission.jsonl --model-mode full --checkpoint outputs/export_lora --asr gipformer
```

`--checkpoint` wants the `step-<n>` directory that `scripts/train.py` writes
(`<output_dir>/<run_name>/step-<n>`, run name = tuning mode). The run root
(`outputs/checkpoints/lora`) holds no weights. `export_model.py --ckpt` accepts
either — `resolve_checkpoint` walks into the newest `step-*` inside a run root.

An adapter recorded against a different family's base is **refused before any
weight is touched** (`lifemate_meta.json` / `adapter_config.json` →
profile comparison), so `--model` must name the training base.

## 3. Decoding: two EOS ids and the stop marker

MiniCPM5-2B declares **two** eos ids — `</s>` = 1 and `<|im_end|>` = 130073 — and
its chat template terminates an assistant turn with **130073**. Decoding stops on
the union of every declared eos id (tokenizer, then the checkpoint's
`generation_config`/`config`, then the profile's defensive extras), measured as
`[1, 130073]`. Stopping on one id only would let the other decode into visible
text and derail parsing.

Separately, generation aborts at `</action>` via a `StoppingCriteria`, so no
tokens are spent after the decision. If the tokenizer cannot stop mid-sequence,
the same truncation is applied post-hoc and flagged in `stats`.

### `max_new_tokens`: two layers, and the config wins

| Layer | MiniCPM5-2B | Qwen3.5-4B | Where |
|---|---|---|---|
| Profile fallback (`HFGenerator` only when the caller passes **no** value) | **192** | **512** | `models/profiles.py` |
| Shipped inference config (`agent.max_new_tokens`) | **512** | **512** | `configs/base.yaml`, `configs/inference.yaml` |

Precisely: 192 is the `HFGenerator` fallback, **not** the runtime default of the
shipped scripts. `scripts/infer.py` and `scripts/evaluate.py` both pass
`cfg.agent.max_new_tokens` explicitly, and the shipped configs set that to **512
for both models**, so a default run decodes up to 512 new tokens on MiniCPM as
well.

Where 192 comes from: measured action targets over 4368 records — p50 33, p90 58,
p99 113, **max 165 tokens** (terminator included). A thinking-disabled turn emits
one short block, so 192 would cover every measured target; that is a statement
about the observed targets, not about what the scripts use. It is **not** enough
for think mode — see the next subsection.

`evaluate.py` additionally accepts `--max-new-tokens N` for teacher-forced
scoring. Changing `agent.max_new_tokens` changes the decode budget only — it does
not change the profile, the prompt, the stop ids or the terminator.

### Think vs no-think: two separate layers, only one of them settled

`MINICPM5_PROFILE.chat_template_kwargs = {"enable_thinking": False}` is often
described as simply "the correct branch". That conflates two very different
claims. They are separated here because only the first one is a fact about bytes.

**Layer 1 — prompt format (measured, token ids).** With `enable_thinking=False`
the template renders a *closed empty* think block, `…<|im_start|>assistant\n\x3cthink\x3e\n\n\x3c/think\x3e\n\n`
(escaped here so the tags stay string data, as in `models/profiles.py`), which
is byte-identical to the serialization the template produces for a training assistant turn that has no `reasoning_content`. The
inference prompt ids for a real SFT record are an exact prefix of that record's
training ids. Re-measured by `tests/test_thinking_modes.py`: the two branches
differ by exactly the trailing `\n\x3c/think\x3e\n\n` and agree everywhere before it
(the think-mode *text* is a strict prefix of the no-think text; the two **id**
lists are not prefix related — 18 vs 20 ids — because BPE merges across the
newline). Layer 1 says the template branch exists and renders as intended. It
says **nothing** about what the model then does.

**Layer 2 — model behaviour (measured on real weights, 2 arms, same prompts).**

```bash
WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src python3 scripts/compare_thinking_modes.py \
    --model models/MiniCPM5-2B --n 20 --seed 0 --max-new-tokens 512 \
    --out outputs/reports/thinking_ab.json
```

Both arms saw the same 20 real `train` records (first decision point each; the
record sample is seeded, decode is **greedy** so the arms are deterministic),
same prompt messages, same 512-token budget, `models/MiniCPM5-2B` bf16 on one
RTX 5070 Ti, no adapter. Only `enable_thinking` differed.

| metric | `enable_thinking=True` | `enable_thinking=False` (shipped) |
|---|---|---|
| n | 20 | 20 |
| valid action (`parse_model_output` → `tool_call`) | **0.15** | **0.20** |
| valid action, untruncated cases only | 0.30 (n=10) | 0.20 (n=20) |
| `finish` | 0.05 | 0.00 |
| parse failure | 0.80 | 0.80 |
| truncated by the 512 budget | **0.50** | **0.00** |
| self-repeating loop | 0.00 | 0.00 |
| `<|im_end|>` emitted | 0.35 | 0.85 |
| emitted `<function name=` instead of `<action>` | 0.25 | 0.10 |
| mean / median / max generated tokens | 399.1 / 512 / 512 | **32.4 / 29 / 62** |
| mean seconds per decision | 5.241 | **0.802** |

A second seed (`--seed 7`, different 20 records) reproduced the ordering:
valid action 0.15 vs 0.20, untruncated 0.25 vs 0.20, truncation 0.40 vs 0.00,
`<|im_end|>` 0.45 vs 0.80, `<function name=` 0.45 vs 0.25, mean tokens 345.9 vs
40.6, mean seconds 4.763 vs 1.008. Raw rows: `outputs/reports/thinking_ab.json`,
`outputs/reports/thinking_ab_seed7.json`.

What that means, honestly:

* **Neither mode is good on the base checkpoint** — 0.15–0.20 valid actions with
  no adapter. These numbers are a *relative* comparison of the two branches, not
  an accuracy claim about a fine-tuned model.
* **Think mode is not more accurate here, and it is ~12× more expensive.**
  Its 50% truncation rate at 512 exists precisely because reasoning tokens are
  spent before the decision; a truncated decode raises (§ above), so those
  samples contribute no action at all.
* **The failure mode of no-think is a missing `</action>`, not reasoning.** In
  seed 0, 14/20 no-think outputs opened `<action>{…}` and stopped at `<|im_end|>`
  without closing the block; 14/14 of those would have parsed as a valid
  `tool_call` had the tag been closed. (Seed 7: 11/20 unclosed, 8/11 would have
  parsed.) That is a base-model instruction-following gap and the natural target
  for the adapter — not evidence for turning thinking on.
* **The parser can consume reasoning when reasoning is actually finished.**
  Every one of the 4/20 think-mode outputs that reached a complete `<action>`
  block was parsed; 3 became `tool_call`, 1 `finish`. No think-mode failure was
  "action inside unterminated reasoning" — all 16 were "no complete
  `<action>…</action>` block found". So the parser's reasoning guard was not the
  bottleneck; the token budget was.
* **Both modes sometimes emit the template's own `<function name=…>` syntax
  instead of the `<action>` protocol** (0.25 vs 0.10 here). Those count as parse
  failures; a branch that emits `<function>` has not solved the task. Note this
  is invisible in `HFGenerator`'s decoded text, because `<function` (id 18) and
  `<param` (id 20) are *special* tokens and `skip_special_tokens=True` erases
  them — the harness decodes the raw ids separately to count them.

**Limitation of the no-think path, stated plainly.** MiniCPM5-2B was post-trained
with deep-thinking SFT and RL, and the model card's official Transformers example
(`models/MiniCPM5-2B/README.md`, "### Transformers" under Quickstart) passes
`enable_thinking=True`. The no-think branch is a template branch that exists and
renders correctly; it is not a documented behaviour guarantee from the author.
Successful rendering ≠ validated model behaviour, and only the numbers above
support keeping it.

**Decision.** Keep `enable_thinking=False` as the shipped default. The evidence
is (a) equal-or-better valid-action rate, (b) 12.3× fewer generated tokens,
(c) 0% vs 50% truncation at the same budget, (d) 6.5× lower latency, and (e) a
parser that never had to reject a reasoning-tainted decision. Revisit after
fine-tuning: re-run `scripts/compare_thinking_modes.py` against the adapter, and
flip only if think mode wins on valid-action rate.

**How to switch to think mode** (the old path stays reachable — nothing below is
the default):

* Override the template kwarg without touching the profile, e.g. in a config or
  harness: `dataclasses.replace(MINICPM5_PROFILE, chat_template_kwargs={**MINICPM5_PROFILE.chat_template_kwargs, "enable_thinking": True})`
  and pass that profile to `HFGenerator`/`render_prompt_text`.
* Budget: the profile's 192 and the shipped 512 are both too small. Think-mode
  decisions in the run above used mean 399 / median 512 tokens and half hit the
  512 ceiling; a truncated decode **raises**, so raise
  `agent.max_new_tokens` (≥1024 is the honest starting point) before enabling it.
  Note the memory ceiling: 1024+ was **NOT RUN** here (14.37/15.17 GB measured
  peak at seq 8192 on a 16.57 GB card).
* Parser: think-mode output carries reasoning before the decision. `parse_action`
  takes the *last* complete `<action>…</action>` block and rejects a block that
  sits inside an unterminated `think`, so reasoning is ignored when it is closed
  and refused when it is not — but the agent pays for those tokens either way.
  The prompt will also no longer be byte-identical to the training
  serialization, which invalidates the Layer-1 parity argument.

### A truncated decode raises — hard rule

`HFGenerator.generate` raises `LifeMateError` when the decode hit
`max_new_tokens` without emitting `</action>`:

```
generation hit the <N>-token budget without emitting </action> (<n> new tokens);
refusing to parse a truncated decision
```

A half-written JSON body can still look like a valid action to a lenient reader,
and executing it is worse than a visible error. Inside the agent loop the failure
is caught per step, recorded as `generator_error`, and the episode stops — the
sample keeps the calls it already made, with `state_hash` from the environment.
`stop_on_action=False` (deliberately unbounded decode) is exempt.

## 4. Useful flags

| Flag | Effect |
|---|---|
| `--split NAME` | split to run (default from config: `public_test`) |
| `--data-root DIR` | dataset root (default: config `data.root`, `./data`) |
| `--output PATH` | submission path (default: config `inference.output`) |
| `--limit N` | first N samples only (smoke runs) |
| `--max-steps N` | agent step budget (default 8) |
| `--time-budget S` | per-sample wall clock budget (default 120 s) |
| `--debug-dump DIR` | per-sample JSON (prompts/observations, PII stripped) |
| `--resume` | skip ids already present in `--output`; keeps existing rows |
| `--config FILE` | YAML config |
| `--override KEY=VALUE` | dotted config override, repeatable (e.g. `agent.max_new_tokens=256` sets the decode budget; it does not switch the model profile) |
| `--device cuda:1` | torch device |
| `--history-transcripts FILE` | ASR transcripts of history user turns; a test split stores history audio only, so without it those prior turns stay blank — and the run then **exits 2** rather than degrading |
| `--asr {gipformer,oracle,whisper,none}` | ASR backend (default from config: `gipformer`) |
| `--allow-missing-history` | diagnostic only: accept blank history turns (never for a submission) |
| `--allow-empty-transcripts` | diagnostic only; never for a competition submission |
| `--quiet` | only errors and the summary line |

`--batch-size` and `--seed` exist and are marked **reserved** in `--help`; they
do not change decoding.

## 5. Offline / air-gapped runs

Inference makes no network call when the model resolves locally. Set the standard
switches (honoured by the loader):

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

`lifemate.models.loader.is_local_or_offline()` reads `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` and turns them into `local_files_only=True`. A model path
that exists on disk is treated as local automatically.

## 6. Docker

Docker is **NOT RUN / UNVERIFIED** here (no daemon in the execution container);
the image was never built.

```bash
docker build -f docker/Dockerfile -t lifemate-infer .

docker run --rm --gpus all --network none \
    -v "$PWD/data":/data:ro \
    -v "$PWD/outputs/checkpoints/full/step-<n>":/models/ckpt:ro \
    -v "$PWD/outputs":/outputs \
    -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
    -e SPLIT=public_test -e MODEL_MODE=full -e ASR=gipformer -e DEVICE=cuda \
    lifemate-infer
```

`--network none` is the intended mode: nothing is installed at run time and the
offline environment variables are baked into the image.

## 7. Mock self-test (no dataset, no model, no GPU)

The mock environment proves the plumbing only. It is refused on the official
output path and prints a banner:

```bash
python3 scripts/infer.py --split public_test --data-root tests/fixtures/synthetic_lifemate \
    --use-mock-env --generator dumpscript:/tmp/script.json \
    --limit 3 --output /tmp/mock_submission.jsonl --asr none
```

`/tmp/script.json` maps sample ids to raw model outputs (a documented test hook):

```json
{"test_single_tool": ["<action>{\"type\": \"tool_call\", \"name\": \"smart_home.device_set\", \"arguments\": {\"device_id\": \"light1\", \"on\": false}}</action>",
                      "<action>{\"type\": \"finish\"}</action>"]}
```

`--generator` without `--use-mock-env` is refused; `--use-mock-env` is refused
whenever the output resolves to `outputs/submission.jsonl`.

## 8. Oracle ASR (evaluation only)

`--asr oracle` requires `--allow-oracle` **and** a labeled split (`train`,
`validation`, `val`, `dev`). On `public_test`/`private_test` it is an error: those
splits have no gold transcript, so oracle would fabricate input.

```bash
python3 scripts/infer.py --split train --data-root ./data --asr oracle \
    --allow-oracle --transcripts outputs/transcripts/train.jsonl \
    --output /tmp/train_oracle.jsonl
```

## 9. Evaluation

```bash
# teacher-forced over SFT validation records
python3 scripts/evaluate.py --validation-only --model-mode lora \
    --checkpoint outputs/checkpoints/lora/step-<n> --predictions outputs/datasets/train/val.jsonl

# closed loop against the official simulator
python3 scripts/evaluate.py --closed-loop --split train --limit 20

# pipeline self-test (marked MOCK ENV / SYNTHETIC in every artifact)
python3 scripts/evaluate.py --closed-loop --split train --use-mock-env --limit 5
```

Artifacts: `outputs/evaluation/{metrics.json,metrics.md,predictions.jsonl,error_analysis.json,comparisons.md}`.
`final_state_correctness` is always `{"available": false}` — the official
scoring tooling is not available here.

`evaluate.py` takes `--model`, `--model-mode`, `--checkpoint`, `--config`,
`--device`, `--predictions`, `--output`, `--max-new-tokens`, `--limit`,
`--quiet` — the same identity rules as `infer.py` apply.

### ASR evaluation

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
(`ref`/`hyp`/`text` are accepted aliases). **ASR WER/CER is independent of
final-state correctness**: a transcript score says nothing about whether the
agent's tool calls were right.

## 10. Verification gate

```bash
python3 scripts/verify_all.py            # table of PASS/FAIL/SKIP/BLOCKED
python3 scripts/verify_all.py --gates A,B,J
```

Writes `outputs/verification/verify_all.json` and `.md`. Exit `1` on any FAIL.
BLOCKED = cannot be checked here (GPU, cloud submit, docker, official
final-state scorer).

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Dataset not found.` + exit 2 | split directory/JSONL missing | extract the dataset under `--data-root` |
| `zaic_tools not installed` + exit 2 | official simulator absent | install the organizers' SDK; `--use-mock-env` is tests only |
| `--model-mode full requires --checkpoint` + exit 3 | no checkpoint path | pass `--checkpoint <step-dir>` |
| `load_model_and_tokenizer requires torch and transformers` + exit 3 | CPU-only box | `pip install -e '.[train]'` |
| `refusing to parse a truncated decision` | decode hit `max_new_tokens` before `</action>` | raise the budget (`--override agent.max_new_tokens=…`); a **base** model can do this — a fine-tune is not proven to fix it |
| `... was trained for model ... but base ... was requested` | adapter from another family | pass the matching `--model` |
| `submission validation: …` + exit 4 | row/order/type problem | see the printed line; re-run without `--resume` for a clean file |
| `--asr oracle is forbidden for split …` | oracle on a test split | use `--asr gipformer` |
| empty `tool_calls` for every sample | model never emitted a valid action | inspect `--debug-dump`. Measured on the base checkpoint: the usual cause is an `<action>{\u2026}` block left **unclosed** before `<|im_end|>` (14/20 and 11/20 cases in the A/B above), or the template's native `<function name=\u2026>` syntax instead of the `<action>` protocol (2/20 and 5/20) |
| `--generator is a TEST HOOK` | scripted generator outside a mock run | add `--use-mock-env` (tests only) |

Exit codes: `0` ok · `2` dataset/environment missing · `3` model load failure ·
`4` submission validation failure.

## 12. CPU tiny-model validation

```bash
python3 -c "from pathlib import Path; from tests.test_tiny_e2e import make_tiny_model_and_tokenizer; make_tiny_model_and_tokenizer(Path('/tmp/tiny_lifemate'))"

python3 scripts/infer.py --split public_test --data-root tests/fixtures/synthetic_lifemate \
    --use-mock-env --model-mode base --checkpoint /tmp/tiny_lifemate --asr none --limit 2 \
    --output /tmp/tiny_submission.jsonl --device cpu

python3 scripts/evaluate.py --validation-only --predictions /tmp/sft_tiny/train/val.jsonl \
    --checkpoint /tmp/tiny_lifemate --limit 2 --output /tmp/tiny_eval --device cpu
```

*Coverage limit*: proves runtime plumbing, action parsing, stopping policies,
submission validation and metrics on CPU. Says nothing about task accuracy.
