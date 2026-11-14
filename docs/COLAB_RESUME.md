# Resume, verification and cleanup — the procedure behind notebook section 6d/13

This is the standalone version of what `notebooks/finetune_qwen35_colab.ipynb` does in cells
**6d** (resume target), **8** (training) and **13** (cleanup). Use it when you resume a Colab run by
hand, or when you need to explain why a checkpoint was refused.

Everything below was exercised on a **local filesystem** by `tests/test_colab_notebook.py`. Nothing
here was executed on a Colab runtime, against a real Google Drive mount, or with Qwen3.5 weights:
those paths are marked `[UNVERIFIED]` and carry the command an operator must run.

---

## 1. What a resumable checkpoint contains

A `step-<n>` directory produced by this trainer:

| file | why it matters for resume |
| --- | --- |
| `adapter_model.safetensors` (or `pytorch_model*.bin`) | the weights; without it there is nothing to continue |
| `training_state.json` | `step`, `epoch`, `seed`, `config_hash`, `dataset_manifest_hash`, `is_lora`, `precision`, torch/python/cuda RNG states, and (schema 2) a `progress` block |
| `optimizer_state.pt` | AdamW moments + `scheduler` + `grad_scaler`; **resume without it restarts the LR schedule and the moments from zero** |
| `lifemate_meta.json` | model identity (`model_name`, `model_profile`, `tuning_mode`) — how a wrong-base resume is caught |
| `SYNC_COMPLETE.json` | **only on a Drive copy**: per-file `size` + `sha256`, `total_bytes`, `source`, `rename_mode`, `schema` |

Two schemas of `training_state.json` exist and **both are accepted**:

* **schema 1** — no `progress` block. Resumable, but `exact=False`: the sampler permutation and
  micro-batch cursor were never recorded, so the trainer continues in `best-effort` mode.
* **schema 2** — `progress` with `global_step`, `epoch_index`, `next_micro_index`, `permutation`,
  `sampler_rng_state`, `planned_total_steps`, `dataset_order_hash`. `exact=True`.

A schema-1 checkpoint is **not** corrupt. The reader reports the missing block as the reason for
`exact=False` and proceeds. The notebook never writes schema 2 itself — that is the trainer's job.

---

## 2. Choosing the resume target

`RESUME_MODE` in notebook section 2:

| mode | behaviour (enforced in code, not prose) |
| --- | --- |
| `"auto"` (default) | Drive first, then local. Picks the **highest step that verifies**. Every rejected candidate is printed with its reason. If candidates exist but none verifies → **raises**; it never silently starts over. If the run has no `step-*` at all → first run, printed as such with the roots searched. |
| `"fresh"` | new experiment. A populated `checkpoints/<RUN_NAME>/step-*` is never reused: the run name gains a UTC stamp (`<run>-2026-11-14T101500Z`) and the change is printed loudly. |
| `"explicit"` | `RESUME_FROM_CHECKPOINT`, verified; any failure raises. |

### 2.1 The verification order (each step yields its own reason)

1. the run directory is the requested one;
2. enumerate `step-*` generations;
3. exclude `.incomplete-*` (interrupted staging) and `.prev-*` (superseded rollback);
4. the marker's per-file `sha256` re-hashes cleanly (Drive copies **require** a marker; a local
   trainer-written directory does not);
5. model identity / profile matches (`lifemate_meta.json`, plus the marker when present);
6. `config_hash` matches — **both values are printed**;
7. `dataset_manifest_hash` matches (a warning, not a refusal, when one side never recorded it);
8. `optimizer_state.pt` exists **and loads**: a valid `torch.load`, an `optimizer` key, and the
   `scheduler` / `grad_scaler` keys present. A 0-byte or truncated file is refused — including on a
   machine without torch, where the `torch.save` zip container is checked structurally;
9. exact-resume metadata (`progress` / `permutation` / RNG states) — **absence is `exact=False`,
   never corruption**;
10. the highest step passing 1-8 is selected.

When the newest generation is corrupt, an older verified one is selected and the rejected newer ones
are printed with their reasons. A candidate that fails 4-8 is never returned.

### 2.2 The report

On resume the notebook prints exactly these lines; the four booleans are derived from the artifact
contents (`optimizer_state.pt` loaded with a non-null `scheduler`, a recorded `permutation`, recorded
torch+python RNG states), never from a notebook variable:

```text
Resume checkpoint:
Previous global step:
Previous epoch:
Next batch:
Optimizer restored:
Scheduler restored:
Sampler restored:
RNG restored:
Expected total steps:
```

`exact` vs `best-effort` comes from `scripts/train.py`'s own `resume mode :` line when it prints one
(current checkout does); otherwise the notebook recomputes it from the `progress` block and states
which source it used. Two independent verdicts that can disagree are worse than one.

### 2.3 Manual equivalent

```bash
# What section 6d does, by hand: pick the newest verified step dir, then let the trainer verify again.
python scripts/train.py --config <cfg.yaml> --dataset-dir <dir> --model Qwen/Qwen3.5-4B \
    --output-dir <ckpt-root> --run-name <run> --tuning lora --seed 42 --seq-len 8192 \
    --save-steps 50 --eval-steps 50 --planned-steps <total> \
    --resume <ckpt-root>/<run>/step-<n>
```

`scripts/train.py` prints `resumed`, `resume mode : exact|best-effort`, `epoch`, `scheduler` and
`rng`; quote those lines rather than recomputing them.

### 2.4 Training-feature fingerprint (cell 2a)

Feature flags are part of a checkpoint's identity, not a run-time preference. The notebook builds
one `FeatureConfig` from the `#@param` block in cell **2a**, validates it with
`REGISTRY.validate_or_raise`, and prints the report:

```text
Enabled features:
  training.slot_masking
...
Mode:
  EXPERIMENTAL
```

`training_feature_fingerprint` (the `training` + `both` scopes only) goes into the generated YAML,
into `RUN["features"]`, into the run report JSON, and into the trainer's own
`training_state.json`. On resume the report prints both values:

```text
  training features: checkpoint <fp> | this session <fp> | MATCH|MISMATCH
  EXACT RESUME REFUSED: the training-feature fingerprint differs (differing flags: [...])
```

A mismatch is not a warning to click through: training features change the supervised data
distribution, so the AdamW trajectory of the checkpoint is not reproducible under new flags. The
trainer refuses it (`_check_feature_identity`); the notebook reports the same verdict instead of
papering over it. To continue the experiment, re-enable the checkpoint's flags; to change them,
start a fresh run.

Three asymmetries worth knowing:

* an **inference-only** id (`asr.multi_asr`, `decoding.constrained`,
  `postprocessing.enhanced_schema_guardrail`, `asr.gipformer_v15`) never moves the training
  fingerprint and is never passed to `scripts/train.py` — the script refuses it, and passing it
  would promise an experiment the run cannot perform;
* a **parameter of a disabled feature** is inert: it is accepted, printed as `INERT`, and excluded
  from both fingerprints, so editing it cannot look like a new experiment;
* a **schema-1 checkpoint** records no fingerprint at all. It is treated as feature-free, so the
  comparison is against `""` and no flag can differ — not as corruption.

---
## 3. Drive sync semantics

```text
copy src -> <dst>.incomplete-<ts>       # never mistaken for a checkpoint
     -> fsync every file
     -> SYNC_COMPLETE.json              # written LAST; per-file size + sha256, rename_mode
     -> <dst> -> <dst>.prev-<n>         # rename, never delete
     -> install the new tree            # os.replace (atomic) or copy-then-mark (copy_only)
     -> verify <dst>, then prune generations beyond KEEP_PREVIOUS
```

* **Directories are versioned and immutable**: `checkpoints/<run>/step-000100/`. The step is part of
  the name, so a published generation is not rewritten in place; only a re-sync of the *same* step
  replaces one.
* **The marker is the contract.** A missing or unparsable marker makes the directory non-selectable.
* **Directory rename is probed, not assumed.** `rename_mode_for()` tests `os.replace` on two
  throw-away sibling directories, caches the answer per parent, and `atomic_sync` records
  `rename_mode: "atomic" | "copy_only"` in the marker. In `copy_only` mode both generations are kept
  for the whole copy, so a crash leaves a `dst` without a marker (refused by every reader) plus an
  intact rollback copy.
  `[UNVERIFIED]` on a real Drive mount — operator: mount Drive, run section 4b, and read the printed
  `rename mode :` line; `copy_only` there is the honest answer, not a failure.

---

## 4. Cleanup of local checkpoints

A local `step-<n>` directory is deleted **only when, for that directory**:

1. a Drive counterpart exists under `checkpoints/<run>/<same name>`;
2. the Drive marker verifies (sha256 of every file);
3. the marker's `run_name`, `step` and `model_id` match this session;
4. the Drive copy is resumable, and so is the local one.

Additional rules:

* a non-empty local set with an **empty** Drive set is an explicit **refusal** — this is the bug
  this procedure replaced (`all([]) is True` deleted everything);
* one missing/unverified counterpart refuses the *whole* cleanup, because a partial delete destroys
  the run's only complete history;
* the **newest** local generation is never removed (it is the one a resume would use);
* `DELETE_LOCAL_AFTER_VERIFIED_SYNC` defaults to `False`: the cell prints what *would* be deleted and
  deletes nothing.

There is deliberately **no** shell equivalent offered here: the check is pairwise (one directory
against one Drive directory, with the marker re-hashed), and the original bug was a glob plus
`all([])`. Run notebook cell 13 — it prints one `[cleanup] <step>: delete|keep — <reason>` line per
local generation, so a refusal is explainable instead of silent.

---

## 5. Worst-case memory probe (section 7c)

The 2-step probe of section 7b only measures the padded length of *the first batch it happened to
get*. Section 7c forces the configured ceiling instead:

forward → chunked CE → backward → accumulate `GRADIENT_ACCUMULATION_STEPS` micro-batches → gradient
clipping → optimizer step → one eval batch → one checkpoint save, with
`reset_peak_memory_stats()` immediately before, so `max_memory_allocated()` /
`max_memory_reserved()` describe exactly that sequence.

Two batches are probed and reported **separately**, never averaged: the configured
`MAX_SEQ_LENGTH` (`source="configured_max"`) and the longest real prepared sample
(`source="longest_sample"`). The report prints 15 labels — `GPU`, `CUDA`, `PyTorch`, `Model`,
`Precision`, `Quantization`, `LoRA rank`, `Sequence length`, `Microbatch size`,
`Gradient accumulation`, `Peak allocated VRAM`, `Peak reserved VRAM`, `Free VRAM`,
`Estimated VRAM`, `Observed VRAM` — with `Estimated VRAM` (heuristic) and `Observed VRAM` (measured)
as different quantities.

A probe that does not fit **stops** with numbered options. The notebook never lowers
`MAX_SEQ_LENGTH` on its own.

`[BLOCKED]` — this machine has no CUDA device, so the probe is a no-op that prints its reason and
every VRAM figure in the notebook remains an ESTIMATE. Operator command:

```bash
# on a GPU runtime, after section 6 and before section 8
jupyter nbconvert --to notebook --execute notebooks/finetune_qwen35_colab.ipynb  # or run 7c in Colab
```

---

## 6. What the tests prove, and what they cannot

`tests/test_colab_notebook.py` extracts these helpers from the `.ipynb` with `ast` and drives them
against temp directories, so the shipped code is what runs:

* `chunk_equivalence` with a 4096-token batch whose first 256 positions are all `-100` (the fixed
  first-256 window would have compared nothing and passed on a NaN), an all-masked batch, NaN/Inf in
  both the single-shot and the chunked value, a single supervised token, and two deliberately wrong
  chunked losses;
* discovery/selection with a 0-byte `optimizer_state.pt`, one flipped adapter byte, a deleted and a
  corrupted marker, `.incomplete-*` and `.prev-*` siblings, and a corrupt newest generation with an
  older valid one — each with the accepting path asserted on the same fixture;
* cleanup with an empty Drive set, one of three missing on Drive, and all three verified;
* all three `RESUME_MODE` branches, plus the labelled report lines;
* the feature switchboard: the shipped cell 5a/6c argv builders produce byte-identical commands with
  every flag off, each flag adds exactly its own args and id, a ratio with its flag off is inert
  and does not move the fingerprint, a bad ratio and `NBEST_TRAINING_FORMAT` without `ASR_NBEST`
  both raise naming the flag, the report prints `Mode: BASELINE`/`EXPERIMENTAL`, and a
  training-fingerprint mismatch prints `EXACT RESUME REFUSED`.

Not provable here: real Colab, real Drive rename semantics, GPU VRAM measurement, bitsandbytes/QLoRA
and Qwen3.5 weights. Those are `[BLOCKED]` above, each with the command an operator would run.
