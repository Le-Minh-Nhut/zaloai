# Data Analysis

> **Section scope: synthetic fixtures only.** This section documents an EDA run
> over `tests/fixtures/synthetic_lifemate/`, the hand-built synthetic fixtures.
> No official LifeMate data was available. **Nothing here is a competition
> statistic.**

## Running the EDA

```bash
python3 scripts/run_eda.py \
  --data-root tests/fixtures/synthetic_lifemate \
  --split train \
  --output /tmp/eda_synthetic
```

Missing dataset (exit code 2, exact message, no traceback):

```bash
mkdir -p /tmp/empty_root_c
python3 scripts/run_eda.py --data-root /tmp/empty_root_c --split train --output /tmp/eda_empty
# Dataset not found.
#
# Expected directory:
# /tmp/empty_root_c/train
# ...
```

## Artifacts

| Path | Content |
|---|---|
| `<out>/summary.json` | every analysis dict (dataset, audio, tools, trajectory, meta) |
| `<out>/report.md` | findings, quality issues, missing fields, visibility limitations, recommendations |
| `<out>/charts/*.png` | up to 10 charts + `charts_manifest.json` (path, series, data_hash) |
| `<out>/tables/*.csv` | `tools.csv`, `tool_usage.csv`, `transitions.csv`, `per_app.csv`, `audio_per_file.csv` |

Matplotlib is **optional**. Without it the report states
`charts: skipped (matplotlib unavailable)` and `charts_manifest.json` records
`"available": false`; every other artifact is still produced.

## Synthetic run result (fixtures, 14 train samples)

Installed environment: Python 3.12, stdlib only (no numpy / matplotlib / torch).

- samples 14; audio 14/14 readable, all 16 kHz mono 16-bit, flags: none
- duration mean 1.31 s, min 0.50 s, max 2.13 s (synthetic clips, not real speech)
- catalog: 13 tools / 5 apps, kinds {meta 1, read 6, write 6}, 1 x-ref edge, 1 x-one-of tool
- gold tool calls: zero 2, single 7, multi 5; `sys.ask_missing_info` 1
- out-of-scope (0 calls, no ask) 2
- failed gold tool results 4 / 17
- charts: skipped in the stdlib environment; verified separately with matplotlib Agg (10 PNGs)

These counts exercise the analysis code paths; they say nothing about the real
dataset distribution.

## Contract notes (EDA / ASR)

- `lifemate.eda` and `scripts/run_eda.py` never name or open a simulator
  (`*_environment`) directory; all file access goes through
  `lifemate.compliance.assert_path_allowed`.
- `user_profile` values are never copied into any artifact — only key names and counts.
- `label.final_state` is counted (apps changed) but is never model input.
- ASR: `OracleTranscriptBackend` is marked `is_oracle = True` and is rejected by
  `scripts/infer.py` for test splits; `WhisperBackend` fails with an actionable
  `AsrError` when torch/transformers are absent and never downloads weights when
  `HF_HUB_OFFLINE=1`.

## ASR Normalization & Diagnostic Metrics (Synthetic-Only)

> **Synthetic-only notice**: No predictions or real test speech transcripts exist yet.
> Therefore, no empirical WER/CER numbers are reported. Metrics and rules below
> document the normalization contract and diagnostic infrastructure.

### Two Distinct Normalization Paths

LifeMate speech contains Vietnamese personal names, addresses, currency values,
calendar dates, and times. Standard benchmark ASR normalization is **destructive**
for downstream tool-calling agent ingestion (e.g. converting `500.000đ` to `500 nghìn`
or altering numeric magnitudes must never corrupt arguments to financial/alarm tools).
The pipeline enforces two strictly separated paths:

1. **`for_wer(text, *, fold_diacritics=False, expand_numbers=True)` (Evaluation Only)**:
   - Unicode NFC canonical composition.
   - Lowercased (`str.lower()`).
   - Conservative number-to-words verbalization for faithful integers ($0$ to $10^{12}$)
     and thousand-separated amounts (e.g. `1.500.000` $\rightarrow$ `một triệu năm trăm nghìn`,
     `21` $\rightarrow$ `hai mươi mốt`).
   - Ambiguous or formatted tokens that cannot be faithfully verbalized without magnitude risk
     (e.g. time `15h30`, date `20/11`, international phone `+84900000000`) are left as digits.
   - Punctuation stripped (Unicode categories `L*` and `N*` preserved; other characters become space).
   - Whitespace collapsed.
   - Whole-token abbreviations expanded (chat abbreviations like `ko` $\rightarrow$ `không`,
     and units like `kg` $\rightarrow$ `ki lô gam`).
   - Vietnamese diacritics **preserved by default**. An explicit `fold_diacritics=True` option
     provides supplementary ASCII-folded CER/WER views.
   - **Strict prohibition**: `for_wer` MUST NEVER be called in agent prompt construction or inference.

2. **`to_llm_text(text)` (Agent / Qwen LLM Ingestion)**:
   - Unicode NFC canonical composition.
   - **Idempotent**: `to_llm_text(to_llm_text(x)) == to_llm_text(x)`.
   - Preserves capitalization (essential for Vietnamese proper names, kinship titles, acronyms like `TP.HCM`).
   - Preserves diacritics.
   - Preserves digits as digits and currency/unit symbols (`500.000đ`, `15h30`, `20/11`, `+84900000000`).
   - Preserves English loanwords (`book`, `shop`, `ok`).
   - Fixes spacing around punctuation without stripping meaning-bearing symbols.

### Normalization Rule Table (`normalization_report()`)

| Rule Name | Description | Example Before | Example After |
|---|---|---|---|
| `unicode_nfc` | Canonical composition | `e\u0301` | `é` |
| `lowercase` | Lowercase string (evaluation only) | `Hà Nội` | `hà nội` |
| `number_expansion` | Verbalize integers / amounts to Vietnamese words (`for_wer`) | `500.000đ` | `năm trăm nghìn đ` |
| `strip_punctuation` | Keep Unicode `L*` and `N*` categories (`for_wer`) | `Xin chào, bạn!` | `xin chào bạn` |
| `collapse_whitespace` | Collapse whitespace runs to single space, trim ends | `  a   b  ` | `a b` |
| `abbreviations` | Expand whole-token spoken chat abbreviations (`for_wer`) | `ko dc vs anh A` | `không được với anh a` |
| `units` | Expand whole-token unit abbreviations (`for_wer`) | `5 kg` | `năm ki lô gam` |
| `to_llm_conservative` | Conservative punctuation spacing without stripping (`to_llm_text`) | `Xin chào , có 500.000đ !` | `Xin chào, có 500.000đ!` |

### Critical Token Diagnostic Metrics (`critical_token_diagnostics`)

Beyond aggregate WER and CER, `lifemate.asr.metrics.critical_token_diagnostics`
classifies tokens into critical semantic categories from both reference and hypothesis:

- **`numbers`**: Digits and Vietnamese number words (`không`, `một`, ..., `nghìn`, `triệu`, `tỷ`).
- **`times`**: Time expressions (`\d{1,2}[h:]\d{2}`, `giờ`, `phút`).
- **`money`**: Currency patterns (`\d+[.,]?\d*\s*(đ|vnđ|đồng|nghìn|triệu|tỷ)`).
- **`entities`**: Proper names, capitalized tokens, kinship titles (`anh/chị/em/cô/chú/bác` + name).
- **`addresses`**: Street/number patterns (`số \d+`, `đường/phố/quận/huyện/phường/xã` + name).
- **`english_loanwords`**: Non-Vietnamese ASCII alphabetic tokens $\ge 2$ characters (`book`, `shop`, `ok`).

For each class, the diagnostics report:
- `n_ref`, `n_hyp`: Extracted token frequencies.
- `exact_set_match_rate`: Sentence-level exact set match rate across active utterances.
- `substitution_rate`: Fraction of reference tokens missing or substituted in hypothesis.
- `magnitude_errors`: Numeric-specific detector flagging when reference and hypothesis
  differ in scale (e.g. `500 nghìn` ($10^3$) vs `5 triệu` ($10^6$)).

### How to run ASR evaluation

The metric layer above is reachable from three commands (no code required):

```bash
# 1. transcribe a labeled split -> predictions JSONL + <output>.metrics.json
python3 scripts/transcribe_split.py --split train --data-root ./data \
    --output outputs/asr/train_predictions.jsonl

# 2. score any prediction JSONL inside the EDA report
python3 scripts/run_eda.py --data-root ./data --split train \
    --asr-predictions outputs/asr/train_predictions.jsonl --output outputs/eda

# 3. append an `asr` block to the evaluation artifacts
python3 scripts/evaluate.py --validation-only \
    --asr-predictions outputs/asr/train_predictions.jsonl
```

Prediction JSONL rows: `{"id", "reference", "hypothesis", "latency_s"}`
(plus `raw_hypothesis`, `duration_s`, `error` from `transcribe_split.py`);
`ref`/`hyp`/`text` are tolerated aliases. Malformed rows are counted and
skipped, and `run_eda.py` reports `n_missing`/`n_extra` against the split.

Reports show, per backend: raw-vs-normalized WER/CER (mean/median/p95),
sentence exact match, latency p50/p95, the six-class critical-token table and
the magnitude-error count. Predictions read from a file are labelled
*file-supplied predictions (not a benchmark result)* — references must come from
a labeled split, and **ASR WER/CER is independent of final-state correctness**,
which stays `available: false` until the official scorer is available.

ASR evaluation framework present; no predictions available $\Rightarrow$ empirical WER/CER not reported (synthetic-only).
