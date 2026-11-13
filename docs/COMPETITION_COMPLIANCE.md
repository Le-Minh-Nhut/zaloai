# Competition compliance

Rules: no reading of the organizers' `*_environment` simulator directories, no
test-label leakage, one environment per sample, and a submission built only from
environment outputs.

## 1. No `*_environment` access

**Mechanics.** `lifemate.compliance` denies any path whose components end with
`_environment` (`train_environment`, `public_test_environment`,
`private_test_environment`, nested or globbed):

```python
from lifemate.compliance import assert_path_allowed, is_prohibited_path, safe_open, read_json
assert_path_allowed("data/train/train.jsonl")   # ok
read_json("data/train_environment/state.json")  # raises ProhibitedAccessError
```

Every file read in `lifemate.data.loader` goes through `compliance.safe_open`
(both `iter_raw_records` and `read_sample_audio`). Discovery only ever calls
`Path.is_dir()` on such a directory to record a boolean
(`SplitPaths.has_environment_dir`); it never lists or opens it.

**Static scan.** `scan_source_for_prohibited_access(root, dirs)` walks source
files and reports any *string literal* that looks like a `*_environment` path
inside a `/`-separated path. `scripts/verify_all.py --gates M` runs it over
`src scripts cloud tests docker` and fails on any finding. Self-referential
mentions (the guard module, this doc, the compliance test) are allowlisted, and
a line may opt out with a `compliance-scan: allow` marker (used once, in the
gate that deliberately constructs such a path).

**Limits of a static scan.** It cannot see paths assembled at run time from
pieces (`"train" + "_" + "environment"`), nor access through a C extension or a
subprocess. It is a review aid plus a regression net, not a sandbox. The real
guarantee is structural: `OfficialToolEnv` delegates to the organizers' SDK and
this repo never opens simulator files itself.

**Test that proves the guard works.** `tests/test_compliance.py` and gate `M`
build a temporary `train_environment/state.json` and assert `read_json()` and
`read_jsonl()` raise `ProhibitedAccessError` on it.

## 2. No test-label leakage

* `Sample.to_model_input()` excludes the label block, and
  `Label.is_model_input = False` is a checkable marker.
* `load_samples(..., with_labels=False)` is what `infer.py` uses.
* The oracle ASR backend is refused on `public_test`/`private_test` even with
  `--allow-oracle` (`_build_asr` in `scripts/infer.py`), because those splits
  have no gold transcript and an oracle there could only produce fabricated
  input.
* SFT record builders (`scripts/prepare_dataset.py`, worker D) keep later
  observations out of step-`k` prompts; `verify_all` gate `F` re-checks this
  independently by scanning each record's prompt for a later step's target or
  observation.

## 3. Submission sourced from the environment only

```python
env = create_environment(split, sample_id, data_root, use_mock=False, catalog=catalog)
result = agent.run(sample, transcript=text)
writer.add(sample_id, env.get_tool_calls(), env.get_state_hash())   # ← env, not result
```

`assert_env_sourced(row, env)` is the defensive hook: it compares the row with
`env.get_tool_calls()` / `env.get_state_hash()` and raises `SubmissionError` on
any difference, so a reconstructed call list cannot reach the file. Failed calls
are kept verbatim — the official scorer replays the whole history.

## 4. One environment per sample

`infer.py` creates a fresh environment inside the per-sample loop; nothing is
reused across ids, so state cannot leak between samples. The state hash is
therefore per-sample.

## 5. The mock is never in the official path — three barriers

1. **CLI flag.** `--use-mock-env` must be passed explicitly; `--generator`
   (the scripted test double) is refused unless that flag is present.
2. **Factory default.** `create_environment(..., use_mock=False)` is the
   default, so omitting the argument yields `OfficialToolEnv`. `OfficialToolEnv`
   has no mock fallback: a missing SDK raises.
3. **Banner + output guard.** A mock run prints
   `MOCK ENVIRONMENT — NOT VALID FOR SUBMISSION` to stderr and refuses any write
   to `outputs/submission.jsonl`, including an explicit `--output
   outputs/submission.jsonl` (that run exits `4`). Pass a different path
   (e.g. `--output /tmp/mock_submission.jsonl`).

`tests/test_inference.py` asserts all three, plus the oracle refusal.

## 6. Data handling and PII

* Reports and debug dumps drop `user_profile`, `messages`, `audio_text`,
  `transcript`, `content`, `history`, `phone`, `email`, `address`
  (`scripts/infer.py::_sanitize`).
* `error_analysis.analyze()` stores sample **ids** only (≤64 chars) and never a
  message body.
* `WandbTracker` redacts any key matching `api_key|token|secret|password|
  passwd|authorization|credential` to `[REDACTED]` and replaces the PII keys
  with `[DROPPED_PII]` unless `allow_pii=True`.
* No credential is ever printed: `cloud/sandbox_runner.py` reports environment
  variable **names** and `set`/`absent`, never values.

## 7. What this cannot guarantee

* That our guesses about the official record schema are correct — the real
  dataset is not available here, so the parsers are tolerant by design.
* That `zaic_tools` behaves like `MockToolEnv` on edge cases.
* Real ASR quality, real model quality, real GPU feasibility: none of these have
  been measured. `docs/VERIFICATION_REPORT.md` records them as BLOCKED, never as
  passed — cite it for that compliance reasoning only. Its counters are a **dated
  snapshot** (`2026-11-08T10:58:58+0700`, `450/455` suites, gate N `311` flags /
  16 docs); the current counts live in
  `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md` §9 and
  `docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md` §I
  (`479/485/491` measured 2026-11-13, gate N `350` flags / 17 docs, gate P `479/491`). The
  `468/468/473` triple is historical: it was measured on `/tmp/torchenv`, a CPU venv deleted
  during the 2026-11-13 session — `/tmp/cudaenv` is the torch interpreter of record now.
