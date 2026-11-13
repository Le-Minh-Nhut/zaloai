# Gipformer-68M-RNNT ASR Integration Audit

## 1. Executive Summary & Original Mismatch

In the initial repository state, PhoWhisper-small (`vinai/PhoWhisper-small`, ~244M parameters, encoder-decoder Transformer via Hugging Face `transformers` + PyTorch) was wired as the default ASR backend in `configs/inference.yaml` and referenced throughout documentation. However, the architectural target for the Zalo AI Challenge 2026 Track 1 ("LifeMate") speech-to-text pipeline is **Gipformer-68M-RNNT** (`g-group-ai-lab/gipformer-68M-rnnt`), a ~68M parameter Zipformer-transducer model running via `sherpa-onnx` runtime.

This architectural correction replaces PhoWhisper with Gipformer-68M-RNNT as the **default** ASR backend across the codebase:
- Pinned model repo: `g-group-ai-lab/gipformer-68M-rnnt`
- Pinned commit revision: `318322b009200f8e77a3751fa64638e5cc85a7dc`
- Precision: INT8 quantized ONNX models (default) and FP32 ONNX models supported without silent fallback
- Runtime: `sherpa-onnx` offline transducer recognizer
- PhoWhisper remains available as an optional comparison backend via `--asr whisper`.

Worker W4 owns Vietnamese normalization (`to_llm_text` / `for_wer` in `src/lifemate/asr/normalization.py`). `GipformerBackend` returns the recognizer output verbatim (`TranscriptResult.text == raw_text`); the LLM-facing `to_llm_text` pass happens once, backend-agnostically, in `scripts/infer.py::_transcribe`. Because normalization is backend-agnostic, the gold-vs-ASR comparison is not confounded by the backend choice.

---

## 2. Verification Gate Separation

| Level | Status | Detail |
|---|---|---|
| **Code Implementation** | COMPLETE | `GipformerBackend`, factory default, `AsrConfig`, YAML configs, download CLI, Docker wiring |
| **Mock / Synthetic Tested** | PASS | 22 test functions in `tests/test_gipformer.py` covering the integration requirements listed below (1–21) (mock recognizer, synthetic audio; item 4 and 5 share one function) |
| **Real Model Tested** | NO | Model weights (~343 MB / ~80 MB int8) absent in test environment; real weight inference not claimed |
| **Real Dataset Tested** | NO | Official Zalo AI dataset absent; synthetic fixtures used for all testing |
| **Docker Tested** | NO | Docker daemon unavailable in test environment; Dockerfile structural validation only |
| **Competition Validated** | NO | Official simulator (`zaic_tools`) and submission portal unavailable |

### 2b. Defect fixes landed after the first integration pass

| ID | Defect | Fix | Covering test |
|---|---|---|---|
| A1 | Unknown `--asr` name fell through and produced empty transcripts | Factory raises `AsrError` naming the valid choices; CLI rejects the typo as an invalid choice | `test_inference.py::test_unknown_asr_backend_raises_instead_of_returning_none` |
| A2 | A `*_environment` audio path could be read as a soft "audio missing" error | Three guards, all propagating `ProhibitedAccessError`: pipeline resolver (`infer.py::_transcribe`, `transcribe_split.py` → `read_sample_audio`), `load_audio_16k_mono` before `wave.open`, and `GipformerBackend._load_audio` before `soundfile`/`wave` (the last two also cover callers that bypass the resolver) | `test_inference.py::test_transcribe_propagates_prohibited_audio_access`, `test_gipformer.py::test_load_audio_16k_mono_refuses_prohibited_path`, `::test_gipformer_load_audio_refuses_prohibited_path` |
| A3 | A broken ASR backend still wrote a submission | ASR failures counted; when nothing usable remains (or >50% failed) the run returns exit 3 before `finalize()`, so no file is created | `test_inference.py::test_asr_failing_every_sample_exits_three_without_submission`, `::test_asr_empty_transcripts_without_error_exits_three` |
| A4 | `to_llm_text` was applied at more than one layer | Normalization happens once, in `scripts/infer.py::_transcribe`, for every backend; `raw_transcript` preserved | `test_inference.py::test_transcribe_normalizes_once_and_keeps_raw` |
| A5 | `decoding_method` default was `greedy_search`, not the published WER configuration | Default is `modified_beam_search` in `AsrConfig` and both configs; `ASR_DECODINGS` validates it | `test_asr.py::test_default_decoding_method_is_modified_beam_search` |
| A6 | `ASRBackend.transcribe` protocol lacked the `sample_id`/`lenient` kwargs every caller used | Protocol declares `*, sample_id: str \| None = None, lenient: bool = False`; all backends bind that shape | `test_asr.py::test_protocol_signature_matches_backend_implementations` |
| B1 | ASR metrics were unreachable from the EDA/evaluation commands | `--asr-predictions` wired through `transcribe_split.py` → `run_eda.py` → `evaluate.py`; file-supplied caveat and `n_missing`/`n_extra` reported | `test_eda.py::test_eda_asr_predictions_reach_the_report`, `::test_transcribe_split_writes_predictions_and_metrics` |
| B2 | Docs still presented PhoWhisper as the primary ASR | Gipformer is the documented default; `--asr whisper` appears only inside blocks labelled as comparison | `test_eda.py::test_docs_never_present_whisper_as_primary_asr`, `::test_primary_inference_example_uses_gipformer_default` |
| B3 | Docs claimed the backend applies `to_llm_text` | Claim removed; the single backend-agnostic call site in `scripts/infer.py` is named instead | `test_eda.py::test_docs_do_not_claim_backend_side_llm_normalization` |

---

## 3. Files Modified & Added

| File | Status | Description |
|---|---|---|
| `src/lifemate/asr/gipformer_backend.py` | New | `GipformerBackend` implementing `ASRBackend` via `sherpa-onnx` transducer; returns recognizer text verbatim (`text == raw_text`); validates `decoding_method` against `ASR_DECODINGS`; `_load_audio` guards `*_environment` paths and `transcribe` re-raises `ProhibitedAccessError` even in lenient mode |
| `src/lifemate/asr/__init__.py` | Modified | Added `GipformerBackend` export; set `"gipformer"` as default backend in `create_asr_backend` |
| `src/lifemate/asr/interface.py` | Modified | `AsrError` derives from `LifeMateError` and `RuntimeError`; `ASRBackend.transcribe` protocol declares `*, sample_id: str \| None = None, lenient: bool = False`; `load_audio_16k_mono` and `GipformerBackend._load_audio` call `assert_path_allowed` before opening audio, so the `*_environment` prohibition holds even for callers that bypass `read_sample_audio` |
| `src/lifemate/config.py` | Modified | `AsrConfig` fields (`backend="gipformer"`, `model_id`, `revision`, `runtime="onnx"`, `quantization="int8"`, `offline=True`) + validation; `ASR_DECODINGS` validation; `decoding_method` default `modified_beam_search` (matches upstream `infer_onnx.py` and the published WER configuration) |
| `configs/base.yaml` | Modified | Updated `asr:` section to `backend: gipformer`, `runtime: onnx`, `quantization: int8`, `device: cpu` |
| `configs/inference.yaml` | Modified | Updated `asr:` section to `backend: gipformer`, `runtime: onnx`, `quantization: int8`, `device: cpu` |
| `pyproject.toml` | Modified | `asr` extra with `sherpa-onnx>=1.10`, `sherpa-onnx-core>=1.10`, `soundfile`, `huggingface-hub`, `numpy`; `onnx` extra; `asr-whisper` extra |
| `scripts/download_asr_model.py` | New | Reproducible download script for pinned Gipformer ONNX models with size/path validation |
| `scripts/infer.py` | Modified | Added `--asr gipformer` option; default resolves to `gipformer`; unknown names fail loudly via the factory; audio resolved through `read_sample_audio` with `ProhibitedAccessError` propagating; ASR failures counted and a broken backend exits 3 before `finalize()`; `--allow-empty-transcripts` (diagnostic only); `to_llm_text` applied once for every backend with `info["raw_transcript"]` preserved; `--debug-dump` records both `transcript` and `raw_transcript` |
| `docker/Dockerfile` | Modified | Installed `sherpa-onnx` and `sherpa-onnx-core`; baked `/models/asr` mount point; `ASR=gipformer` default |
| `docker/entrypoint.sh` | Modified | Default `ASR="${ASR:-gipformer}"`; auto-binds `/models/asr` when mounted |
| `docker/README.md` | Modified | Documented `ASR=gipformer` default and `/models/asr` volume mount |
| `docs/INFERENCE_GUIDE.md` | Modified | Gipformer documented as the default `--asr` (host mount instructions, troubleshooting); whisper demoted to a labelled comparison subsection; `--allow-empty-transcripts` documented as diagnostic only; `transcribe_split.py` flags and exit codes listed |
| `README.md` | Modified | Gipformer documented as the default ASR backend with whisper as an optional comparison; `.[asr]` install dependencies and repo tree architecture notes updated |
| `tests/test_gipformer.py` | New | 22 test functions (`grep -c "^def test_"`): items 4 and 5 share one function, plus two defence-in-depth guard tests (`test_load_audio_16k_mono_refuses_prohibited_path`, `test_gipformer_load_audio_refuses_prohibited_path`) covering the integration requirements listed below (1–21) (mock recognizer, synthetic audio) |
| `tests/test_asr.py` | Modified | Updated default backend assertion to `cfg.asr.backend == "gipformer"` |
| `scripts/transcribe_split.py` | New | Transcribes a split with the configured backend, writes predictions JSONL + `<output>.metrics.json`; audio resolved through `read_sample_audio` (guarded, `ProhibitedAccessError` re-raised); exit 2 when the backend is unavailable (nothing written), 3 when every transcription failed |
| `scripts/run_eda.py` | Modified | `--asr-predictions FILE` feeds `evaluate_asr` + `critical_token_diagnostics` into the EDA report; missing/unusable path stays explicit (`available: false`, exit 2); reports `n_missing`/`n_extra` and skipped rows |
| `scripts/evaluate.py` | Modified | `--asr-predictions FILE` appends an `asr` block/section; final-state correctness stays `available: false` |
| `src/lifemate/eda/report.py` | Modified | Renders the ASR section (raw vs normalized WER/CER, latency p50/p95, 6-class critical-token table, magnitude-error count, file-supplied caveat) |
| `tests/test_gipformer.py`, `tests/test_asr.py`, `tests/test_inference.py`, `tests/test_eda.py` | Modified | Regression coverage for the above (unknown backend, guarded audio, ASR-failure refusal, single normalization point, decoding default, protocol signature, ASR-eval wiring, doc-drift guard) |
| `docs/DATA_ANALYSIS.md` | Modified | ASR-evaluation workflow (`transcribe_split.py` → `run_eda.py --asr-predictions` → `evaluate.py`), prediction-row schema, `n_missing`/`n_extra` reporting, file-supplied caveat |
| `docs/IMPLEMENTATION_READINESS_REPORT.md` | Modified | Backend table updated (Gipformer default, whisper comparison); states that LLM normalization is backend-agnostic in `scripts/infer.py::_transcribe`, correcting the stale backend-side claim |

---

## 4. Exact Model Identity & Pinned Assets

- **Repository**: `g-group-ai-lab/gipformer-68M-rnnt`
- **Revision**: `318322b009200f8e77a3751fa64638e5cc85a7dc`
- **Architecture**: `ZipformerRnntModel` (`zipformer-transducer`)
- **Version 1.5 Restriction**: Upstream default `DEFAULT_VERSION = "1.5"` resolves `g-group-ai-lab/gipformer1.5-68M-rnnt`, a different model. The codebase hard-pins `g-group-ai-lab/gipformer-68M-rnnt` at the exact commit above and explicitly rejects any model ID referencing `1.5`.

### File Sizes & Quantization Flavour Mapping

| File | Size (Bytes) | Role | Quantization |
|---|---|---|---|
| `encoder.int8.onnx` | 70,876,359 | Transducer Encoder | int8 (default) |
| `decoder.int8.onnx` | 1,308,690 | Transducer Decoder (Predictor) | int8 (default) |
| `joiner.int8.onnx` | 1,033,417 | Transducer Joiner | int8 (default) |
| `encoder.onnx` | 261,057,847 | Transducer Encoder | fp32 |
| `decoder.onnx` | 5,165,084 | Transducer Decoder (Predictor) | fp32 |
| `joiner.onnx` | 4,104,465 | Transducer Joiner | fp32 |
| `tokens.txt` | 25,847 bytes | BPE Token Vocabulary (2000 tokens) | shared |
| `bpe.model` | 270,695 | SentencePiece Model | shared |
| `config.json` | 615 | Model Configuration | shared |

Total storage:
- **INT8 path (default)**: ~73.5 MB (70.1 MiB)
- **FP32 path**: ~270.6 MB (258.1 MiB)
- **Full repository**: ~343.8 MB (327.9 MiB)

---

## 5. Runtime Architecture: Why `sherpa-onnx` and Not Raw `onnxruntime`

Gipformer is a Zipformer transducer model. A transducer consists of three interconnected neural networks evaluated autoregressively during beam/greedy search:
1. **Encoder**: Converts audio filterbanks into acoustic frames.
2. **Decoder (Predictor)**: Evaluates preceding predicted tokens (RNN-T history).
3. **Joiner**: Combines encoder and decoder representations into a probability distribution over the vocabulary.

Implementing transducer search with raw `onnxruntime` requires hand-rolling the RNN-T greedy or beam search loop in Python (managing hidden states, blank token emission loops, maximum symbol per frame limits, and search trees), adding latency and potential decoding bugs.

`sherpa-onnx` provides:
- C++ optimized transducer decoding engine (`OfflineRecognizer.from_transducer`).
- Exact alignment with upstream Zipformer / Icefall models.
- Native multithreaded greedy search and modified beam search.
- Minimal Python overhead via direct C++ stream buffers.

### The `sherpa-onnx-core` Pin Requirement
Starting with `sherpa-onnx` 1.13+, the native compiled library `libonnxruntime.so` was split into a separate distribution package: `sherpa-onnx-core`. Installing `sherpa-onnx` alone without `sherpa-onnx-core` causes an immediate `ImportError: cannot open shared object file: libonnxruntime.so`. Both packages (`sherpa-onnx>=1.10` and `sherpa-onnx-core>=1.10`) are strictly declared in `pyproject.toml` and Docker configurations.

---

## 6. Device, Quantization, and Offline Strategy

### Precision / Quantization
- Config field `quantization: "int8"` (default) maps to `*.int8.onnx`.
- `quantization: "fp32"` maps to `*.onnx`.
- **No silent substitution**: If FP32 files are requested when only INT8 files exist on disk (or vice versa), `GipformerBackend` raises an explicit `AsrError`. It never silently switches precision.

### Device Handling
- `device: "cpu"` (default): Uses CPU inference with `num_threads=4`.
- `device: "cuda"`: Passes `provider="cuda"` to `sherpa-onnx.OfflineRecognizer.from_transducer` when supported by the installed wheel. If the installed build lacks CUDA provider support, an actionable `AsrError` is raised with installation instructions.
- `device: "auto"`: Checks `torch.cuda.is_available()`, selecting `"cuda"` if available, else falling back to `"cpu"`.

### Offline Mode
- `offline: True` (default in base/inference config): Model files are resolved strictly from local disk (`model_dir` or `models/gipformer-68M-rnnt`). Missing files raise `AsrError` with the exact download command. Zero network calls are made.
- `offline: False` (explicit opt-in): Permitted for development; uses `huggingface_hub.hf_hub_download` at the pinned revision.

---

## 7. Exact Commands

### Download Model Assets
```bash
# Download default INT8 weights (approx 73.5 MB)
python3 scripts/download_asr_model.py --output-dir models/gipformer-68M-rnnt --quantization int8

# Dry run (prints exact file list and byte counts without downloading)
python3 scripts/download_asr_model.py --dry-run --quantization int8
```

### Run Competition Inference
```bash
# Default inference (automatically uses Gipformer ASR from configs/inference.yaml)
python3 scripts/infer.py \
    --split public_test \
    --data-root ./data \
    --output outputs/submission.jsonl \
    --model-mode full \
    --checkpoint outputs/checkpoints/full/step-<n>

# Explicitly specifying Gipformer ASR backend
python3 scripts/infer.py \
    --split public_test \
    --data-root ./data \
    --output outputs/submission.jsonl \
    --model-mode full \
    --checkpoint outputs/checkpoints/full/step-<n> \
    --asr gipformer
```

---

## 8. Verification Results

### Test Suite Execution
- `python3 tests/run_tests.py`: **passed=380 failed=0 skipped(needs fixtures)=6**; skipped modules `test_tiny_e2e.py`, `test_tiny_generation.py` (Wall time: 3.27s)
- `.venv/bin/python tests/run_tests.py`: **passed=380 failed=0 skipped(needs fixtures)=6** (Wall time: 6.86s)
- `.venv/bin/python -m pytest -q`: **380 passed, 6 skipped** (Wall time: 6.92s)
- `/tmp/torchenv/bin/python -m pytest -q`: **455 passed, 1 skipped** (superseded an earlier 385/410 count; this is the current figure)
- `PYTHONPATH=src python3 scripts/verify_all.py`: **17 PASS + 1 BLOCKED across gates A–R** — every gate passes except `Q`, which stays BLOCKED by design for GPU training/inference, cloud submit, docker and the official final-state scorer. Gate Q is probe-driven: the real dataset and the per-split `zaic_tools` SDK report as verified. Footer verdict as printed by the gate: `CODE COMPLETE (this repo) / SYNTHETIC TESTED (fixtures) / REAL DATASET TESTED: yes (public_test 634, train 1545) / OFFICIAL SDK: yes (public_test, train) / REAL MODEL TESTED: no / GPU TRAINING TESTED: no / COMPETITION VALIDATED: no` (the footer is derived from Gate Q's own probe results in `real_environment_summary()`, so the two cannot disagree).

Counts measured 2026-11-08 on this host; re-run the four commands to refresh.

### Test Items Verified (tests/test_gipformer.py)
1. Default backend is `gipformer` in `AppConfig`, `base.yaml`, `inference.yaml`, and factory.
2. Exact model ID (`g-group-ai-lab/gipformer-68M-rnnt`) and revision (`318322b...`) pinned.
3. Version 1.5 rejected by config validation and backend constructor.
4. Recognizer constructed with exact INT8 and FP32 filenames via mocked `sherpa_onnx`.
5. INT8 configuration propagation verified.
6. Missing precision files raise explicit `AsrError` without silent fallback.
7. CPU device path passes `num_threads` and omits CUDA provider.
8. Audio loading handles 16 kHz mono and resamples 8 kHz to 16 kHz.
9. Empty audio raises `AsrError` or yields error result in lenient mode.
10. Corrupt/invalid WAV headers raise `AsrError`.
11. Vietnamese text with diacritics and numbers ("500.000đ") preserved in `raw_text` and `text`.
12. Missing model files produce actionable error with `download_asr_model.py` invocation.
13. `offline=True` verified to make zero network calls (raising monkeypatch).
14. CLI integration: `scripts/infer.py --help` shows `--asr gipformer`; missing weights exits code 2 without traceback.
15. Closed-loop agent integration with `MockToolEnv` and ASR stub succeeds.
16. Oracle restriction unchanged: forbidden on test splits without `--allow-oracle`.
17. Submission format compatibility verified against environment provenance.
18. Docker structural checks: `sherpa-onnx` declared, `/models/asr` created, no whisper weights.
19. Download script `--dry-run` displays file list and byte size.
20. Download script compliance guard refuses `*_environment` target directories.
21. Parameter class (68M) and on-disk size accounting verified.

---

## 9. Remaining Limitations & Pending Real Evals

1. **Real-Weight Latency & WER Benchmark**: Not executed in this environment due to absence of GPU and real weights. Benchmarking on official validation audio with real Gipformer weights remains pending once official evaluation hardware is available.
2. **Docker Build Execution**: Docker daemon is absent in this test environment. Structural Dockerfile inspection passed, but full image build and runtime test (`docker run`) must be executed on a Docker-capable host.
3. **Worker W4 Normalization Integration**: `GipformerBackend` deliberately does **not** call `lifemate.asr.normalization.to_llm_text`; it returns verbatim recognizer output. The LLM-facing normalization is applied once in `scripts/infer.py::_transcribe`, identically for every backend, so the gold-vs-ASR comparison is not confounded.
4. **No Real Weights / No `sherpa-onnx`**: real Gipformer weights and the `sherpa-onnx` runtime are absent on this host, so `scripts/transcribe_split.py` exits 2 with the actionable backend-unavailable message and writes no file. Real WER/CER therefore remains NOT RUN; every number reported here is synthetic-fixture only.
