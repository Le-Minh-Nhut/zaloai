# Architecture

LifeMate Track 1 (Zalo AI Challenge 2026): Vietnamese voice input → a multi-step
tool-calling agent whose output for one sample is exactly

```json
{"id": "sample-1", "tool_calls": [{"name": "alarm.set", "arguments": {"time": "06:30"}}], "state_hash": "…"}
```

with `tool_calls` and `state_hash` taken verbatim from the official ToolEnv.

> Status: the real dataset **is present** (`./data`: `train` 1545 samples,
> `public_test` 634) and the per-split `zaic_tools` SDK imports via
> `lifemate.env.official.check_official_sdk`; model weights, a CUDA torch build
> and cloud credentials are **absent**. Data-side code has been exercised on the
> real splits, model-side code only on synthetic fixtures — real-model
> validation has not been run.

## End-to-end dataflow

```mermaid
flowchart TD
    A[data/&lt;split&gt;/&lt;split&gt;.jsonl + audio/] -->|discover_splits, load_samples| B[Sample<br/>audio · context · history]
    C["&lt;split&gt;_tools_catalog.json"] -->|load_catalog| D[ToolCatalog]
    B -->|audio_path| E[ASR backend<br/>gipformer · whisper · oracle(eval only)]
    E -->|transcript| F[ToolAgent loop]
    D -->|ToolRegistry.for_sample<br/>installed apps only| F
    G[generator: HFGenerator<br/>base · full · lora] -->|next action| F
    F -->|call name + args| H[create_environment<br/>use_mock=False → OfficialToolEnv]
    H -->|observation ok/result| F
    H -->|get_tool_calls / get_state_hash| I[SubmissionWriter]
    I -->|atomic rename| J[submission.jsonl]
    J -->|validate_submission| K[exit 0 / 4]

    L[SFT records: prompt · target] -->|evaluate_teacher_forced| M[Metrics]
    F -->|closed-loop rows| M
    M --> N[metrics.json/.md · predictions.jsonl ·<br/>error_analysis.json · comparisons.md]
```

## Module map

| Area | Module | Responsibility |
|---|---|---|
| config | `lifemate/config.py` | dataclass config, YAML/JSON load, dotted overrides |
| guards | `lifemate/errors.py`, `lifemate/compliance.py` | error hierarchy; `*_environment` path deny-list |
| data | `lifemate/data/*` | split discovery, tolerant schemas, catalog, loader, validator |
| env | `lifemate/env/*` | `OfficialToolEnv` (zaic_tools adapter) and `MockToolEnv` |
| agent | `lifemate/agent/*` | protocol, prompt builder, registry, executor, stop policy, loop |
| models | `lifemate/models/*` | model/tokenizer loading, HF generator, checkpoint IO |
| asr | `lifemate/asr/*` | backend protocol, gipformer (default)/whisper/oracle backends, WER/CER |
| training | `lifemate/training/*` | SFT collator/trainer, full + LoRA, memory preflight |
| evaluation | `lifemate/evaluation/metrics.py` | pure-stdlib decision metrics |
| evaluation | `lifemate/evaluation/evaluator.py` | teacher-forced + closed-loop drivers, artifacts, provenance |
| evaluation | `lifemate/evaluation/error_analysis.py` | failure buckets, dominant-cause recommendations |
| evaluation | `lifemate/evaluation/submission.py` | atomic writer, row validation, env-provenance assert |
| evaluation | `lifemate/evaluation/submission_validator.py` | structural validation of a submission file |
| tracking | `lifemate/tracking/wandb_tracker.py` | optional W&B logging with redaction, no-op fallback |
| tracking | `lifemate/tracking/artifact_manager.py` | checkpoint upload/download, cheap manifests |
| entrypoints | `scripts/*.py`, `lifemate/cli.py` | CLI per stage; `python -m lifemate <command>` dispatcher |

## Contract table

| Producer | Symbol | Consumer |
|---|---|---|
| `env.factory` | `create_environment(split, sample_id, data_root, use_mock, catalog, mock_state, apps)` | `infer.py`, `evaluate.py` |
| `env.base.ToolEnvironment` | `validate / call / get_tool_calls / get_state_hash / available_tools` | `ToolAgent`, submission writer |
| `agent.agent` | `AgentResult{status, steps, tool_calls, state_hash, transcript, n_llm_calls, elapsed_seconds, errors, finish_reason}` | `evaluate_closed_loop`, debug dumps |
| `agent.prompt_builder` | `build_decision_prompt`, `to_messages`, `StepRecord` | agent loop, SFT builder |
| `models.generation` | `.generate(messages) -> str` (duck type) | `ToolAgent`, `evaluate_teacher_forced` |
| `evaluation.submission` | `SubmissionWriter.add/finalize`, `assert_env_sourced` | `infer.py` |
| `evaluation.metrics` | `aggregate_teacher_forced`, per-metric functions | `evaluate_split` reports |
| `tracking.wandb_tracker` | `WandbTracker.log/log_metrics/log_summary/log_config/finish/url` | training, evaluation |

## Design decisions

**Own training loop instead of `Trainer`.** The loss mask must keep the prompt
tokens out of the loss and the action tokens in it — a detail that is easy to
get wrong through a generic collator. Owning ~300 lines of loop also keeps the
package importable without `accelerate`/`datasets`, which makes `--dry-run`,
memory preflight and CI possible on a CPU-only box. Cost: we maintain the loop.

**Frozen one-action protocol.** `<action>{...}</action>` is parsed by
`agent.protocol` only. Training targets, inference parsing and evaluation all
route through the same functions, so the model cannot be trained on one format
and decoded with another. A malformed action is a first-class value
(`Action(type="invalid")`) that is fed back to the model, not an exception.

**Environment abstraction, mock behind three barriers.** `ToolEnvironment` is a
5-method protocol; `MockToolEnv` exists so the whole pipeline is testable
without the organizers' simulator. Three independent barriers keep the mock out
of a submission: (1) `--use-mock-env` is an explicit CLI flag and `--generator`
refuses to work without it, (2) `create_environment(use_mock=False)` is the
default so a forgotten flag cannot silently mock, (3) a loud banner plus an
absolute refusal to write `outputs/submission.jsonl` (the official submission path)
under mock mode.
**Rows come from the environment.** The submission is built from
`env.get_tool_calls()` / `env.get_state_hash()`, and `assert_env_sourced()`
fails the row if it disagrees with the env. The agent's own `AgentResult` copy
is used for diagnostics only.

**LCS for sequence scores.** Calls are a sequence (a write may consume a read's
output), so a multiset would score a reordering as correct. LCS tolerates one
inserted call without penalising every later call, which is the failure mode
we actually want to measure. Exact match is reported separately.

**Unavailable is a value.** Any metric without an applicable sample returns
`{"available": false, "reason": …}` rather than `0.0`. The hidden-state metric
is permanently unavailable: `final_state_correctness` is
`{"available": False, "reason": "official scoring tooling unavailable"}`.

**Cheap manifests.** `checkpoint_manifest()` hashes only config/meta files; a
multi-GB `*.safetensors` gets a size entry. The artifact store already
content-addresses weights, so hashing them again would only cost minutes.

## Failure modes and their symptom

| Failure | Symptom | Where handled |
|---|---|---|
| dataset absent | canonical `Dataset not found.` text, exit 2, no traceback | `require_split`, script `main` |
| `zaic_tools` absent | `LifeMateError` naming the missing SDK, exit 2 | `env.official` |
| model/torch absent | actionable pip message, exit 3 | `infer.py`, `export_model.py` |
| model emits garbage | `Action(type="invalid")`, loop continues or stops | `agent.protocol`, `tool_parser` |
| env rejects a call | `ok:false` entry kept in the log and submission | `MockToolEnv._record`, writer |
| unfinished submission | structural validation errors, exit 4 | `submission_validator` |
| W&B offline/missing | warning logged once, run continues | `WandbTracker` |
| deadline exceeded | `status="timeout"`, bucket `runtime` | `StoppingPolicy`, `error_analysis` |
| prompt overflow | truncated prompt, bucket `prompt_construction` | `prompt_builder`, `error_analysis` |
