# Final Pre-training Readiness Audit

> ## ⚠ SUPERSEDED SNAPSHOT — dated 2026-11-08. The numbers in this document are NOT current.
>
> This file is kept for the **defect inventory (§3), the provenance labelling convention (§2)
> and the reasoning about what a tiny-model CPU pass can and cannot establish** — that prose
> still describes how the workstreams were audited. Do not quote its counters.
>
> Everything quantitative here predates the SFT rebuild, the collator/catalog fixes and the
> 2026-11-13 session. The historical values appear unmarked in §6, §8 and §9; only this notice
> marks them stale.
>
> **Live documents (use these for numbers):**
>
> - `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` — experiment readiness, in Vietnamese;
> - `docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md` — dataset and defect audit, in Vietnamese;
> - `README.md` status table — one status line per dimension.
>
> **Figure families that changed (old here → current):**
>
> - test suites, stdlib / `.venv` pytest / torch venv: `450 / 450 / 455` with 6 / 6 / 1 skips
>   → **`479 / 485 / 491`** (skips `13 / 7 / 1`, measured 2026-11-13). The `455` figure came
>   from `/tmp/torchenv`, a CPU venv **deleted during the 2026-11-13 session**; `/tmp/cudaenv`
>   (`torch 2.15.0+cu132`, `transformers 5.22.2`) is the torch interpreter of record now.
>   The intermediate `468 / 468 / 473` triple is also historical;
> - SFT build: `4385` records → **4850** (**4368** train + **482** val decision points, 0 sample drops);
> - truncation at `max_seq_length` 8192: `46/4385 = 1.05 %` → **47/4850 = 0.97 %**
>   (47/4368 = 1.08 % of train, 0/482 val);
> - truncation at 4096: `3299/4385 = 75.23 %` → **3520/4850 = 72.58 %**;
> - `tool_call` schema-block loss: `27.7 % (921/3324)` at 4096 → **0/3305** after the catalog
>   was pinned to the prompt tail (the `3/3324` residue at 8192 is likewise now `0/3305`);
> - LoRA memory: the `11.31 GB` figure / `4.14 GB` headroom were the **0.5 % placeholder**
>   estimate → measured **11.49 GB** (32,464,896 trainable params, `trainable_source=measured`);
> - `/tmp/torchenv` no longer exists (`ls -d /tmp/torchenv` → `No such file or directory`);
>   every command line below that names it must be read as `/tmp/cudaenv`.

Audit of the five implementation workstreams (data/SFT, agent, ASR/EDA, training, eval/cloud/docker) plus their defect-remediation pass, executed on 2026-11-08 in this repository's verification environment. Every claim below is labelled by provenance and, where it came from the repo, carries the `file:line` it was read from. No real dataset, no real Qwen3.5-4B weight bytes, no GPU training step, no cloud job and no docker build has been executed in this environment — the only model-side execution was meta-device and tiny-model CPU work.

## 1. Verdict

**Is the repository ready to receive the official dataset and begin real integration testing?**

**YES, with the five prerequisites in §7.**

That answer says nothing about competition readiness: nothing here is competition-validated, gate Q is BLOCKED, and no score, WER, or loss has ever been measured on real inputs.

## 2. Method and provenance labels

Four labels are used throughout this document, verbatim:

- `OBSERVED (executed)` — a command was run in this environment and its output is quoted.
- `OBSERVED (source)` — read from a named `file:line`.
- `OBSERVED (network metadata)` — fetched from a named URL (config/index JSON only; never weights).
- `NOT VERIFIED` / `NOT RUN` — with the missing prerequisite named.

A tiny-model CPU pass validates the training **loop** (collator, masking, optimizer, scheduler, checkpoint save/resume, RNG restore) — it never validates the **architecture** (Qwen3.5 hybrid attention, real gradient numerics, real memory).

## 3. Confirmed defects fixed (19)

| ID | severity | file:line | trigger | why wrong | fix | regression |
| --- | --- | --- | --- | --- | --- | --- |
| EVAL-A | critical | `src/lifemate/evaluation/evaluator.py:160` `_messages_of` | a record carrying `prompt`/`input`/`body` string | the gold `target` was appended as an assistant turn, so the gold answer sat in model-visible context and an echoing model scored 1.0 | never append the target (`evaluator.py:166-170`) | `tests/test_eval.py::test_teacher_forced_never_scores_gold_against_itself` |
| EVAL-B | critical | `src/lifemate/evaluation/evaluator.py:200` `_expected_calls` | a submission/`predictions.jsonl` file passed as `--predictions` | `tool_calls` is the *prediction* key of both formats; reading it as gold scored the model against itself on an empty prompt | `_GOLD_KEYS = ("gold_tool_calls", "expected_tool_calls")` (`:185`), `_is_gradeable` requiring prompt AND gold (`:187`), `n_ungradeable` reported (`:296`, `:368`), `scripts/evaluate.py:194-199` exits 2 before writing any artifact | `tests/test_eval.py` (EVAL-A/B tests) |
| EVAL-C | critical | `src/lifemate/evaluation/metrics.py:374` `decision_emitted`, `:387` `aggregate_teacher_forced` | crashed generator or output with no complete `<action>` block | empty `tool_calls` equalled an empty gold and scored 1.0 — a CUDA OOM reported as a perfect "do nothing" | gate applied to exact-match (`:404-406`), P/R/F1 (`:408-414`), arguments (`:417-427`), selection/stop (`:429-444`), next-action (`:445-452`), missing-info and out-of-scope via `_UNMEASURED` poisoning (`:454-462`); `n_unmeasured` exposed (`:401-402`, `:466`) | `tests/test_eval.py` metrics tests |
| EVAL-D | high | `src/lifemate/evaluation/evaluator.py:350` per-record row | same crash/parse-error case | per-record `exact_match` said `True` while the aggregate said `0.0` in the same `metrics.json` | same gate: `metrics.decision_emitted(pred) and metrics.tool_call_exact_match(...)` | `tests/test_eval.py` |
| ASR-A | high | `src/lifemate/asr/gipformer_backend.py:239` `_load_audio` (read at `:262`) | a 10-minute file | `sf.read()` had no `frames=`, so the whole file was decoded and `tolist()`-ed (~73 MB of Python floats) before the `max_audio_seconds` slice | `sf.info(...).samplerate` (`:259-261`) then `sf.read(..., frames=frame_budget)` (`:262`) | `tests/test_gipformer.py::test_soundfile_read_is_bounded_by_max_audio_seconds` |
| ASR-B | critical | `src/lifemate/asr/interface.py:172-177`, `src/lifemate/asr/gipformer_backend.py:241-244` | symlink with clean path components pointing into `*_environment/` | the compliance guard was lexical only, so the banned file was opened | `assert_path_allowed(path.resolve())` in addition to the lexical check | `tests/test_gipformer.py::test_symlink_into_environment_dir_is_refused` (incl. `transcribe(lenient=True)`) |
| ASR-C | high | `scripts/download_asr_model.py:53-96` (`_TREE_API`, `fetch_pinned_hashes`), used by `--verify` | same-size tampering, or Gipformer 1.5's byte-identical file sizes | size-only comparison cannot distinguish repos that publish identical sizes, so only the repo-id string blocked the swap | per-file digests from the pinned revision's tree API — `lfs.oid` = sha256 for LFS files, plain `oid` = git blob sha1 — never hardcoded; offline degrades to sizes with an explicit message (`:120`) | `test_verify_detects_same_size_tampering`, `test_verify_offline_degrades_to_sizes_only`, `test_fetch_pinned_hashes_returns_none_offline` |
| SFT-A | critical | `src/lifemate/data/trajectory_builder.py:143` `align_calls_and_results`, `:446-449`, `:459-466` | result whose explicit index contradicts its position | `result_out_of_order` was reported but `build_decision_points` consumed `alignment.steps` unconditionally, so step k's prompt showed a result produced by another call | `Alignment.misaligned` (`:99-112`, out-of-order only) → return no decision points; `misaligned_results` drop reason (`:461-466`, `:633`) | `tests/test_trajectory.py::test_misaligned_results_emit_no_decision_points` |
| SFT-B | critical | `src/lifemate/data/trajectory_builder.py:566` `build_sft_dataset` (`:621-625`) | gold call missing a required argument | only tool *availability* was validated, never *executability*, so `<action>…"arguments": {}</action>` was supervised — an action the loop rejects before reaching the env | `registry.validate` gate, drop reason `unexecutable_gold_target` | `tests/test_trajectory.py::test_unexecutable_gold_target_is_dropped` |
| SFT-C | critical | `src/lifemate/agent/tool_executor.py:48`, `scripts/infer.py:316`, `:619` (also `:322`) | `ProhibitedAccessError` during a tool call or transcription | blanket `except Exception` downgraded a compliance violation into an observation, i.e. data the model could learn to work around | `except ProhibitedAccessError: raise` at all three sites | `tests/test_agent.py::test_prohibited_access_is_never_captured_as_an_observation`, `tests/test_inference.py::test_transcribe_propagates_backend_prohibited_access` |
| SFT-D | critical | `src/lifemate/training/collator.py:96` `REQUEST_MARKER`, `:99` `_split_at_request`, `:192` | long trajectory: the steps section grows | middle truncation cut a fixed head fraction of the whole prompt, so the `# Yêu cầu` (spoken request) section was deleted and the target had no request to condition on | keep the system+request prefix and truncate only what follows it | `tests/test_training.py::test_collator_truncation_keeps_the_request_section` |
| SFT-E | high | `src/lifemate/data/trajectory_builder.py:338` `_is_meta_call`, `:532` | catalog spells the meta tool `ask_missing_info` under the `sys` app | terminality was decided by literal name comparison, so a `finish` point was emitted right after an unanswered question | resolve `kind == "meta"` through the catalog spec instead of the name | `tests/test_trajectory.py::test_ask_missing_alias_is_terminal` |
| TRAIN-A1 | critical | `src/lifemate/training/trainer.py:871` `_load_state_dict_into`, `:908` `_without_tied` (used at `:782`) | resumed run on a hybrid checkpoint this trainer had just written | raw `load_state_dict` bypassed transformers' weight conversions and tied-weight handling, so every converted key was reported missing (measured: 55 missing + 55 unexpected for 56 tensors, `tests/test_training.py:1983`) | apply the model's checkpoint conversions, then drop tied names from the load report | `tests/test_training.py::test_hybrid_checkpoint_reloads_through_weight_conversions` |
| TRAIN-A2 | critical | `src/lifemate/training/trainer.py:798` `_load_adapter_weights` (called at `:768`) | LoRA resume | `PeftModel.load_adapter` takes a positional `adapter_name` in peft ≥0.21 (verified signature in `peft 0.21.2`), so resume raised `TypeError`; loading under a new name would have left the active adapter at fresh init | load into the active adapter name and assert the name did not change | `tests/test_training.py::test_lora_resume_restores_adapter_and_enables_checkpointing` |
| TRAIN-A3 | critical | `src/lifemate/training/trainer.py:425-428` `run()` | resumed run | the restored `LambdaLR` had been built for the *interrupted* run's horizon and was already at it, so every remaining step ran at LR 0.0: full compute, no learning (`OBSERVED (executed)` reproduction: `[(5,0.0),(6,0.0),(7,0.0),(8,0.0)]`) | rebuild the schedule on the restored optimizer for the real horizon via `build_scheduler(..., last_epoch=start_step)`; AdamW moments untouched | `tests/test_training.py::test_resumed_run_keeps_a_real_learning_rate` |
| TRAIN-A4 | high | `src/lifemate/training/lora_finetuning.py:143-160` | LoRA run with `training.gradient_checkpointing: true` (both shipped configs set it) | `configure_lora` ignored the flag, so checkpointing stayed off (`use_cache` stayed on) and no `enable_input_require_grads` was installed — adapter gradients would vanish through checkpointed blocks | `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`, `enable_input_require_grads()`, `use_cache = False`, all reported in the returned dict (`:190`) | `tests/test_training.py::test_lora_resume_restores_adapter_and_enables_checkpointing` |
| TRAIN-A5 | medium | `src/lifemate/training/trainer.py:354-369` `_prepare_examples` | split whose last record is falsy | the per-split kept/dropped metrics entry was silently skipped | append unconditionally — the counters are the diagnosis when the "no usable examples" error fires | `tests/test_training.py::test_trainer_drop_accounting_and_empty_guard` |
| CLOUD-B1 | high | `cloud/sandbox_runner.py:207` | training command inside the image | passed `--config /workspace/configs/training_full.yaml`, a path that does not exist in the image (`docker/Dockerfile:94` copies configs to `/app/configs`) | `/app/configs/training_full.yaml` | `tests/test_cloud_spec.py` |
| CLOUD-B2 | critical | `cloud/sandbox_runner.py:186`, `:122` `uncovered_container_paths`, `:411` | inference command + model mount | `--checkpoint /models/ckpt` while the mount's container was `/models`, so the container path resolved to `<host>/ckpt`; and a run root is not loadable (`infer.py` passes it straight to `from_pretrained`) | mount container `/models/ckpt`, host resolved to the newest `step-<n>` via `resolve_checkpoint` (`:191-193`), plus a static coverage gate that fails the dry run and blocks submit | `tests/test_cloud_spec.py` (3 tests: `test_spec_has_no_uncovered_container_paths`, `test_uncovered_detector_flags_legacy_layout`, `test_plan_exits_zero_and_submit_refuses_unmounted`) |

### Documentation defects fixed (8)

- `--checkpoint` must point at a `step-<n>` directory, not a run root (`docs/INFERENCE_GUIDE.md:46-48`, `docs/GIPFORMER_INTEGRATION_AUDIT.md:160`, `docs/IMPLEMENTATION_READINESS_REPORT.md:247`).
- `--output-dir outputs/checkpoints` double-nests the run directory (`docs/IMPLEMENTATION_READINESS_REPORT.md:240`).
- docker mount of a run root instead of `step-<n>` (`docker/README.md:31` now documents `…/full/step-<n>`).
- `README.md`'s `--generator dumpscript:` pointed at a file no command created.
- `--tuning` invalid documented as exit 1 where argparse yields 2 (`docs/TRAINING_GUIDE.md:259` now states 2; measured: `train.py: error: argument --tuning: invalid choice`, exit 2).
- the claim that an explicit `--output` bypasses the mock refusal — the refusal is unconditional and exits 4 (`scripts/infer.py:425-452`; reproduced against `outputs/submission.jsonl`, exit 4 even with `--output` given).
- `docs/VERIFICATION_REPORT.md` gate table stale against the artifact.
- a documented `LIFEMATE_DATA_ROOT` environment variable that no code reads (deleted; `grep -rn LIFEMATE_DATA_ROOT` returns nothing).

## 4. Suspicions investigated and DISPROVEN

- "`AutoModelForCausalLM` is the wrong class for a hybrid Gated-DeltaNet model" — `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3_5']` resolves to `Qwen3_5ForCausalLM` (`OBSERVED (local library)`, transformers 5.22.2 in `/tmp/torchenv`).
- "A multimodal checkpoint silent-random-initialises the LM" — the conversion path's load returned an empty missing/unexpected set (through `_load_state_dict_into`) and the library's resolved conversion mapping contains `PrefixChange(['^model\.language_model\.(.+)'] → ['model.\1'])` (`transformers.conversion_mapping.get_model_conversion_mapping`, `OBSERVED (local library)`); the real index lists all 426 `model.language_model.*` tensors (`OBSERVED (network metadata)`); the tensors themselves are `inference`.
- "`--predictions` may be misread as model output, or the checkpoint may never be executed" — `val.jsonl` is written by `prepare_dataset` with keys `{sample_id, step_index, messages, target, meta}` (`scripts/prepare_dataset.py:13-14`, `:183`), i.e. validation examples with gold; `scripts/evaluate.py:140-155` loads the checkpoint (`load_model_and_tokenizer` → `HFGenerator`) and generates per record (`OBSERVED (source)`), and the suite verifies it with a stubbed loader that records the call.
- "Teacher-forced and closed-loop metrics are conflated" — two distinct functions (`metrics.aggregate_teacher_forced` vs `evaluator`'s closed-loop path), and no perplexity metric exists anywhere in the repo (`grep -rni perplexity src/ scripts/ tests/` returns nothing); token accuracy exists only as a *training* validation number (`trainer.py:619`), never as task success; `final_state_correctness` is hardwired `{"available": False}` (`evaluator.py:485-488`).
- "The collator can emit an all-`-100` row and NaN the loss" — the model does return NaN for such a row, but `build_supervised_example` guarantees ≥1 supervised token for every `kept` example, so it is unreachable (`OBSERVED (source)` + masking tests).
- "Gipformer silently falls back to PhoWhisper" — the whisper branch is reached only when `name == "whisper"` is requested (`src/lifemate/asr/__init__.py:112`); any unusable name raises `AsrError` (`:126-133`), and `transcribe_split` returns exit 2 (`EXIT_DATA`, `scripts/transcribe_split.py:33`, `:151`) writing no output file.
- "Inference reconstructs `tool_calls`/`state_hash` from the transcript" — both are read from the env (`scripts/infer.py:628-629`) and `assert_env_sourced` compares canonical JSON plus hash against the same env (`:635`).
- "`assert_no_label_leak` never runs" — it runs per decision point (`trajectory_builder.py:537`); its scope is label-only key literals plus `final_state` containers ≥30 chars (`_MIN_NEEDLE_CHARS = 30`, `:268-272`), deliberately not leaf values.
- "`name_mismatch` proves mispairing, so those samples must be dropped" — result names are free-form in real data; a name difference is a flag (`:459-460`), not proof. Only an explicit index contradicting its position discards a sample (`:99-112`).
- "History turns with audio cannot render, so train and inference disagree" — both paths call the same `_render_history` (`src/lifemate/agent/prompt_builder.py:152-171`) on the same objects and emit nothing for such turns (`content` only): consistent but lossy, so a capability ceiling rather than a defect. Residual risk: if the official history carries audio-only turns, the request may be unresolvable from text; the cheapest detection is a prepare-dataset counter for history turns with an audio key and empty content — not implemented.

## 5. Unverified real-model behaviour

| item | status | missing prerequisite |
| --- | --- | --- |
| Real Qwen3.5-4B one-step training (loss, grad norms, peak GPU memory, checkpoint reload on real weights) | NOT RUN | no weight bytes locally; downloads forbidden by policy |
| GPU execution | NOT RUN | GPU **present** (see below), but the only installed torch build is `2.15.0+cpu` with `cuda = None` |
| Real Gipformer inference, WER, CER, latency | NOT RUN | no `sherpa_onnx`, no Gipformer ONNX assets, no `soundfile` (all verified absent from `/tmp/torchenv`) |
| Real vision-tower parameter count of the released checkpoint | NOT VERIFIED | index tensor counts are known (297 `model.visual.*`); a parameter count was not computed |
| Real checkpoint tensor values/dtypes on disk | NOT VERIFIED | no weight bytes fetched |

### GPU availability

`OBSERVED (executed)` — the hardware is present and the software is not:

- `/proc/driver/nvidia/gpus/0000:02:00.0/information` → `Model: NVIDIA GeForce RTX 5070 Ti`, `GPU Firmware: 580.173.02`, `GPU Excluded: No`; `nvidia-smi --query-gpu=name,memory.total,driver_version` → `NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 580.173.02`.
- `/tmp/torchenv/lib/python3.12/site-packages/torch/version.py:4,6` → `__version__ = '2.15.0+cpu'`, `cuda: Optional[str] = None`; consequently `torch.cuda.is_available()` is `False` and `torch.cuda.device_count()` is `0`.

So GPU present (RTX 5070 Ti, 16303 MiB = **17.09 GB decimal / 15.92 GiB** total, 14733 MiB = 15.45 GB decimal free when idle); the only installed torch build is `2.15.0+cpu` with `cuda = None`, so `torch.cuda.is_available()` is `False`. Real one-step training is therefore NOT RUN for two independent reasons: no CUDA torch build and no Qwen3.5-4B weights locally. This is an **environment prerequisite, not a repo defect**.

The repo's own estimator (`src/lifemate/training/memory_preflight.py:104-165`) makes the path explicit. Two provenance notes on the table below: the "shipped config" column is printed by `scripts/train.py --dry-run` (`OBSERVED (executed)`); the "real config" column is arithmetic from `run_memory_preflight` with `param_count=4,205,751,296` and a 32-layer/2560-hidden model config injected (`OBSERVED (executed)`, but it is the estimator's formula, not a memory measurement). The capacity figure is **not** from `_gpu_info()` — that returns `unavailable` on this CPU-only torch build — it is `nvidia-smi`'s `16303 MiB × 1024²` converted to decimal GB = **17.09 GB** (15.92 GiB); it was passed in as `gpu_info={'total_bytes': …}`. An earlier revision of this document said 16.29 GB, which corresponds to 15535 MiB and is wrong.

| mode | estimator with the shipped config defaults (4.0B assumed params, 32 layers, seq 8192, microbatch 1) | estimator with the real config (4,205,751,296 params, 32 layers, seq 8192, microbatch 1) | verdict at 17.09 GB capacity / 15.45 GB free |
| --- | --- | --- | --- |
| LoRA | 10.88 GB (`train.py --dry-run --tuning lora`) | 11.31 GB | fits (0.73x free) |
| full FT | 66.68 GB (`train.py --dry-run --tuning full`) | 69.98 GB | does_not_fit (4.53x free) |

Gradient checkpointing is modelled in both columns (`training.gradient_checkpointing: true` → `k = 2` instead of `10`), so these are the checkpointed figures; with it off, LoRA at 8192 rises to 35.47 GB and full FT to 80.71 GB. LoRA at 11.31 GB leaves **4.14 GB headroom** below the 15.45 GB free figure. Sequence length is 8192 in both configs since the truncation measurement below.

Estimator arithmetic — `run_memory_preflight`'s formula — says LoRA fits this card and full-parameter SFT does not. Nothing here is a measured peak allocation.

**Sequence budget (`max_seq_length: 8192`, raised from 4096).** `OBSERVED (executed)` with the real Qwen3.5 tokenizer over the 4385-record SFT build: at 4096 the collator middle-truncates **3299/4385 = 75.23 %** of records (at 8192: **46/4385 = 1.05 %**), and the sacrificed middle is where `# Lịch sử hội thoại` and `# Công cụ khả dụng` live — **27.7 % (921/3324)** of `tool_call` decision points lose the parameter-schema block of the tool they are asked to call, the block being the literal `### <tool> (kind=` string in the decoded, truncated prompt, over train+val `tool_call` targets; the whole `# Công cụ khả dụng` section is lost in **4.0 % (133/3324)** (176/4385 = 4.01 % over all records). At 8192 those fall to **0.09 % (3/3324)** and **0**; since the 2026-11-08 collator change that pins `# Công cụ khả dụng` to the preserved tail, schema-block loss is **0/3324** with truncation unchanged (47/4385 under the current D-24 rendering, 46 when the pre-D-24 prompt suffix is used). Largest record needs 9014 tokens; targets are never truncated. Both shipped configs and `TrainingConfig.max_seq_length` now say 8192, which costs 1.35 GB of total estimate (LoRA 11.31 GB at 8192 versus 9.96 GB at 4096, microbatch 1 — the activation term doubling from 1.34 to 2.68 GB) and is covered by the margin above.

**Precision (`full_finetuning.master_dtype`).** `OBSERVED (executed)`: AdamW writes its update into the parameter tensor, so bf16 master weights quantise a `lr = 5e-6` step away — measured on `N(0, 0.02)` weights with grad scale 0.01 over 20 steps, only **7.4 %** of elements moved in bf16 versus **100 %** in fp32 (`moved=0.073 / mean|delta|=1.33e-06` vs `moved=1.000 / 2.15e-05`). Full fine-tuning therefore loads fp32 master weights (requested by `scripts/train.py`, which is also why the weight/gradient terms in the table above are 4 bytes per parameter) and takes its bf16 speed from `torch.autocast`; LoRA keeps the frozen base in bf16, as the LoRA column shows.

What *is* known about the architecture, with provenance:

| fact | value | provenance |
| --- | --- | --- |
| text config | 32 layers, 24 `linear_attention` + 8 `full_attention`, hidden 2560, head_dim 256, `attn_output_gate: true`, `tie_word_embeddings: true`, vocab 248320, `mtp_num_hidden_layers: 1`, vision `depth: 24` | `OBSERVED (network metadata)` — released `config.json` |
| index | 738 tensors / 9.32 GB (`total_size` 9,319,737,856): 426 `model.language_model.*`, 297 `model.visual.*`, 15 `mtp.*`; no `lm_head.weight` | `OBSERVED (network metadata)` — released `model.safetensors.index.json` |
| meta-device `Qwen3_5ForCausalLM` built from that real text config | `sum(named_parameters())` = `sum(parameters())` = **4,205,751,296** (426 tensors, no `lm_head`); `sum(state_dict().values())` = **4,841,450,496** (427 tensors, adds `lm_head.weight`); difference = 635,699,200 = `vocab × hidden` = 248320 × 2560, exactly the tied tensor counted twice | `OBSERVED (executed, no weight bytes)` |

Consequence: text-only SFT trains ~4.21B parameters (the tied case), and the vision+MTP weights in the download are unusable dead weight for this task.

## 6. Dataset-dependent and cloud-dependent checks

| check | status | what would settle it |
| --- | --- | --- |
| Official schema/split layout | RUN on the extracted splits (`train` 1545, `public_test` 634): `scripts/inspect_data.py`, `scripts/run_eda.py` and `scripts/prepare_dataset.py` (4385 SFT records) completed | model weights / GPU still absent, so no training ran on it |
| The parser's tolerance of unknown fields | designed (`unknown fields kept, never rejected`, `src/lifemate/data/schemas.py:1-7`), unverified on real data | one real split through `inspect_data.py`, comparing `raw` retention and validator findings |
| History-audio-in-turn frequency | unknown | the prepare-dataset counter described in §4 |
| `zaic_tools.ToolEnv` behaviour, `get_tool_calls()`/`get_state_hash()` contract | NOT RUN (SDK absent) | install the organizers' wheel and run `create_environment('public_test', …)` |
| Cloud submit | dry run only, not cloud validation | `wandb` absent (verified), `preflight` blocks on missing credentials; the spec now passes the mount-coverage gate |
| Docker build/runtime | NOT RUN (no daemon) | `docker build` + `docker run --network none` on a host with a daemon |

## 7. Prerequisites before real integration (the five)

1. Official dataset extracted at the configured root (`inspect_data.py` → `run_eda.py` → `prepare_dataset.py`; treat a schema mismatch as a parser defect in `src/lifemate/data/schemas.py`, not as a data problem).
2. `sherpa_onnx` + Gipformer-68M-RNNT assets (`download_asr_model.py --verify` now hash-checks against the pinned revision) → `transcribe_split.py` → real WER/CER.
3. Install a CUDA-enabled torch build (cu128 for RTX 5070 Ti / sm_120) — the hardware is already present. LoRA (`11.31 GB` here was the 0.5 % placeholder estimate; measured figure is **11.49 GB**, seq 8192, microbatch 1) fits the 17.09 GB / 15.45 GB-free card with 4.14 GB headroom; full FT (69.98 GB est.) does not, and needs an 80 GB-class or sharded target. Estimator arithmetic, not a measurement. **Superseded by `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` §8.3.**
4. `zaic_tools` installed from the organizers.
5. W&B credentials for the cloud path (dry run only until then).

## 8. Reproduction commands

```bash
# RUN in this environment (historical figures from 2026-11-08)
python3 tests/run_tests.py                 # then: passed=450 failed=0 skipped(needs fixtures)=6; now: passed=479 … =13
.venv/bin/python -m pytest -q              # then: 450 passed, 6 skipped; now: 485 passed, 7 skipped
PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q   # then: 455 passed, 1 skipped; now: 491 passed, 1 skipped
PYTHONPATH=src python3 scripts/verify_all.py   # PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 1

# NOT RUN in this environment (need the official dataset)
python3 scripts/inspect_data.py --data-root ./data
python3 scripts/run_eda.py --data-root ./data
python3 scripts/prepare_dataset.py --data-root ./data --split train --output outputs/datasets
```

## 9. Evidence summary

| metric | value | source |
| --- | --- | --- |
| stdlib runner | `passed=450 failed=0 skipped(needs fixtures)=6`, skipped modules `test_tiny_e2e.py`, `test_tiny_generation.py` | `python3 tests/run_tests.py` (`OBSERVED (executed)`) |
| venv pytest | `450 passed, 6 skipped` | `.venv/bin/python -m pytest -q` (`OBSERVED (executed)`) |
| torch pytest (**historical**) | `455 passed, 1 skipped` | `/tmp/torchenv/bin/python -m pytest -q` (`OBSERVED (executed)` before that venv was deleted 2026-11-13; `/tmp/cudaenv` now reports `491 passed, 1 skipped`) |
| `verify_all` totals | `PASS 17, FAIL 0, SKIP 0, BLOCKED 1` | `PYTHONPATH=src python3 scripts/verify_all.py`; artifact `outputs/verification/verify_all.json`, `meta.timestamp = 2026-11-08T` (this run; see the artifact for the exact value) |
| gate J | `test_agent.py: 56 passed` | artifact `results` / `outputs/verification/verify_all.md` |
| gate N | `311 documented flag(s) across 16 docs verified in --help` | artifact `results` |
| gate P | `run_tests: passed=450 failed=0 skipped(needs fixtures)=6`; `pytest: 450 passed, 6 skipped` | artifact `results` |
| gate Q | `BLOCKED — GPU training/inference, cloud submit (wandb missing), docker build/runtime, official final-state scorer; verified here: real dataset (public_test 634, train 1545), official ToolEnv (zaic_tools)` | artifact `results` |

What this audit did NOT establish: no real model weight was loaded, no GPU step ran, no cloud job ran, no docker image was built, and no example was scored with the official final-state scorer. The real dataset *was* parsed (`train` 1545 / `public_test` 634; 4385 SFT records) and the per-split `zaic_tools` SDK *does* import.
