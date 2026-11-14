# Feature flags (switchboard)

Tài liệu này mô tả **hạ tầng cờ tính năng** trong `src/lifemate/features/`.
Gói này chỉ là *bảng công tắc*: nó KHÔNG cài đặt bất kỳ kỹ thuật nào. Mỗi
workstream khác đọc cờ của mình từ đây.

**Mọi cờ đều mặc định `False`.** Một tham số số học (ví dụ `nbest_size: 5`,
`noisy_asr_ratio: 0.5`) khi cờ của nó tắt là **trơ** — không bật gì, không báo
lỗi. Chỉ cờ `*_enabled: true` mới bật tính năng.

## Thứ tự phân giải

```
CLI  --enable-feature ID / --disable-feature ID
  >  ENV   LIFEMATE_FEATURE_<ID_UPPER_SNAKE>=1|0|true|false
  >  YAML  features:  (section trong config)
  >  mặc định trong FeatureConfig (tất cả False)
```

- Một đối tượng duy nhất: `lifemate.features.config.FeatureConfig`
  (`audio`, `asr`, `prompt`, `training`, `decoding`, `postprocessing`).
- Một danh sách duy nhất: `lifemate.features.registry.FEATURES`.
- Một chuỗi phân giải duy nhất: `lifemate.features.overrides.apply_overrides()`.
- `apply_overrides()` **trả về object mới** và gắn `cfg.provenance` để biết cờ
  nào bật do `cli` / `env` / `yaml` / `default`.
- Id sai (CLI, env, hoặc key YAML) luôn **báo lỗi** kèm gợi ý gần đúng
  (`difflib.get_close_matches`), không bao giờ bỏ qua im lặng.

## Bảng cờ

| id | default | scope | owner (module đích) | status |
|---|---|---|---|---|
| `audio.vad` | `false` | both | `scripts/infer.py` (`trim_silence` before ASR) | `WIRED` |
| `audio.rms_normalization` | `false` | both | `scripts/infer.py` (RMS gain before ASR) | `WIRED` |
| `asr.gipformer_v15` | `false` | inference | `src/lifemate/asr/model_specs.py`, `docker/entrypoint.sh` | `WIRED` |
| `asr.nbest` | `false` | both | `src/lifemate/asr/nbest.py`, `scripts/infer.py` | `WIRED` |
| `asr.multi_asr` | `false` | inference | `src/lifemate/asr/multi_backend.py`, `scripts/infer.py` | `WIRED` |
| `prompt.itn` | `false` | both | `scripts/infer.py`, `scripts/transcribe_split.py` (`apply_itn_pipeline`) | `WIRED` |
| `prompt.nbest_format` | `false` | both | `src/lifemate/data/asr_variants.py` (`render_request_section`), `scripts/prepare_dataset.py` | `WIRED` |
| `training.noisy_asr` | `false` | training | `src/lifemate/data/asr_variants.py` (`apply_noisy_asr`), `scripts/prepare_dataset.py` | `WIRED` |
| `training.slot_masking` | `false` | training | `src/lifemate/data/augmentation.py` (`apply_slot_masking`), `scripts/prepare_dataset.py` | `WIRED` |
| `training.tool_dropout` | `false` | training | `src/lifemate/data/augmentation.py` (`apply_tool_dropout`), `scripts/prepare_dataset.py` | `WIRED` |
| `decoding.constrained` | `false` | inference | `src/lifemate/models/generation.py` (`HFGenerator(constrained=True)`), `scripts/infer.py` | `WIRED` |
| `postprocessing.enhanced_schema_guardrail` | `false` | inference | `src/lifemate/agent/agent.py`, `scripts/infer.py`, `scripts/evaluate.py` | `WIRED` |

Mọi cờ đã được nối dây: bật cờ là đổi hành vi thật, tắt cờ là đường code
baseline byte-for-byte (`tests/test_baseline_equivalence.py`). Không còn cờ
`PENDING` nào. `tests/test_feature_integration.py` chứng minh từng seam bằng
stub/backend giả — không cần weights thật.

`decoding.constrained` — `HFGenerator(..., constrained=True)` định tuyến
`generate()` qua `lifemate.models.constrained.constrained_generate`. Cờ này
**không cần gì thêm** ngoài stack inference đang cài: nó là một
`transformers.LogitsProcessor` + prefix trie thuần Python, không dùng
`outlines`/`llguidance`/`xgrammar` (registry đã bỏ tuple `requires` đó).
`scripts/infer.py` truyền cờ và bộ tool của từng sample qua
`generator.set_allowed_tools(...)`.

## Cách bật

### YAML

```yaml
features:
  asr:
    nbest_enabled: true
    nbest_size: 3
  prompt:
    itn_enabled: true
```

Các overlay sẵn dùng (mỗi file chỉ đổi section `features:`) nằm trong
`configs/experiments/`:

| file | cờ bật |
|---|---|
| `baseline.yaml` | không có (mọi cờ `false`, ghi tường minh) |
| `gipformer_v15.yaml` | `asr.gipformer_v15` |
| `asr_nbest.yaml` | `asr.nbest` (`nbest_size: 3`) |
| `itn.yaml` | `prompt.itn` |
| `enhanced_guardrail.yaml` | `postprocessing.enhanced_schema_guardrail` |
| `noisy_asr_training.yaml` | `training.noisy_asr` (`noisy_asr_ratio: 0.25`) |
| `advanced_combined.yaml` | `asr.gipformer_v15`, `asr.nbest`, `prompt.itn`, `postprocessing.enhanced_schema_guardrail` |

Chúng là **overlay**, không phải config đầy đủ: `load_config()` áp section
`features:` đã biết rồi giữ mặc định cho phần còn lại, nên overlay hợp lệ trực
tiếp. `configs/base.yaml` ghi tường minh khối `features:` mặc định (mọi cờ
`false`) — baseline là một quyết định được viết ra, không phải hệ quả của việc
thiếu key.

**Nối dây đã hoàn tất.** Mọi entry point đọc cờ qua
`lifemate.features.overrides.build_feature_config` và in
`format_feature_report` trước khi nạp model:

| file | nối dây |
|---|---|
| `scripts/train.py` | `build_feature_config` + report; `training_features` vào metadata checkpoint |
| `scripts/infer.py` | `audio.vad` / `audio.rms_normalization` / `asr.nbest` / `asr.multi_asr` / `prompt.itn` / `decoding.constrained` / `postprocessing.enhanced_schema_guardrail`; sidecar `feature_report.json` cho run thí nghiệm |
| `scripts/evaluate.py` | cờ + report; `decoding.constrained` và guardrail chuyển vào generator/agent |
| `scripts/prepare_dataset.py` | `training.noisy_asr` / `training.slot_masking` / `training.tool_dropout` / `prompt.nbest_format` (qua `_augment`), kèm alias CLI cũ |
| `src/lifemate/agent/agent.py` | `postprocessing.enhanced_schema_guardrail` (chỉ siết, không bao giờ nới lỏng parser) |
| `src/lifemate/models/generation.py` | `decoding.constrained` → `constrained_generate` + `set_allowed_tools` |
| `notebooks/finetune_qwen35_colab.ipynb` | `#@param` cho cờ, validate trước khi chạy, từ chối cấu hình vô nghĩa |

### CLI

Mọi script nhận hai cờ lặp lại được:

```
--enable-feature asr.nbest  --enable-feature prompt.itn  --disable-feature training.tool_dropout
```

`--disable-feature` thắng cả YAML và env (nó là lớp CLI). Dạng `ID=true|false`
cũng được chấp nhận.

### ENV

```
LIFEMATE_FEATURE_ASR_NBEST=1
LIFEMATE_FEATURE_POSTPROCESSING_ENHANCED_SCHEMA_GUARDRAIL=true
```

Tên biến sai (bắt đầu bằng `LIFEMATE_FEATURE_` nhưng không map được) sẽ **lỗi**,
kèm gợi ý tên đúng.

## Kiểm tra hợp lệ

`REGISTRY.validate(cfg)` trả về danh sách vấn đề (không raise);
`REGISTRY.validate_or_raise(cfg)` raise `FeatureConfigurationError`
(subclass của `LifeMateError`, nên các `except LifeMateError` sẵn có trong
script vẫn bắt được).

Các luật đang có:

- id lạ → lỗi kèm gợi ý gần đúng;
- phụ thuộc chưa bật (`prompt.nbest_format` cần `asr.nbest`) → lỗi;
- xung đột khai báo mà cả hai cùng bật → lỗi (hiện chưa có cặp nào xung đột;
  `prompt.itn` và `prompt.nbest_format` **tương thích**);
- `asr.nbest` bật nhưng `nbest_size` ngoài `2..10` → lỗi;
- `asr.multi_asr` bật nhưng `secondary_model_id` rỗng → lỗi;
- `training.noisy_asr` bật nhưng `noisy_asr_ratio` ngoài `(0, 1]` → lỗi;
- tính năng bật nhưng `requires` không import được → lỗi.

Tính khả dụng là **lazy**: cờ tắt thì không `import`/`find_spec` bất cứ module
tuỳ chọn nào. `tests/test_features.py` chứng minh điều này bằng một import hook
(`sys.meta_path`) chặn `outlines`/`llguidance`/`xgrammar`: import
`lifemate.features` và `REGISTRY.validate(FeatureConfig())` không kích hoạt hook.

## Fingerprint

- `feature_fingerprint(cfg)` → 16 hex trên toàn bộ trạng thái có thể đổi hành vi.
- `training_feature_fingerprint(cfg)` → cùng dạng, chỉ gồm các spec có
  `scope` là `training` hoặc `both` (dùng cho metadata checkpoint / Colab).

Loại trừ có chủ ý (giống nhau ở cả hai hàm, hằng `FINGERPRINT_EXCLUDED`):
`audio.vad_min_silence_ms`, `audio.vad_padding_ms`, `audio.rms_target_dbfs`,
`training.augmentation_seed` — chỉnh tham số vận hành cùng một phân phối dữ
liệu, không phải một thí nghiệm mới. Tham số của tính năng **đang tắt** cũng bị
loại (chúng trơ theo định nghĩa). `asr.nbest_size` **được tính** khi `asr.nbest`
bật.

## Báo cáo

`format_feature_report(cfg, model_id=..., asr_id=...)` in khối
`LifeMate Effective Configuration` (Enabled/Disabled/Mode);
`feature_report_dict(cfg)` trả dict JSON-ready (enabled/disabled/mode/
fingerprint/training_fingerprint/provenance) để nhúng vào `report.json` hoặc
metadata checkpoint. `mode` là `BASELINE` khi và chỉ khi không cờ nào bật.

## Baseline equivalence (gate F3)

`tests/test_baseline_equivalence.py` chứng minh **cờ tắt == hành vi trước khi có
hệ thống cờ**. Sáu golden được tính MỘT LẦN từ đường code cũ và dán cứng vào
test (không tự sinh lại — golden tự sinh không chứng minh gì):

| # | thành phần | golden |
|---|---|---|
| 1 | `agent.prompt_builder.build_decision_prompt` | sha256 + độ dài + thứ tự section của prompt đầy đủ (có và không có prior steps) |
| 2 | `agent.protocol.render_action` | byte của `<action>...` cho một `tool_call` và một `finish` |
| 3 | `data.trajectory_builder.build_sft_dataset` | 26 record / 14 sample, sha256 trên toàn bộ record đã sắp xếp, sha256 của `stats`, record đầu/cuối |
| 4 | `asr.normalization.to_llm_text` | 12 chuỗi tiếng Việt (dấu, số, `15h30`, `500.000đ`, `+84900…`, HOA/thường), output chính xác + idempotent |
| 5 | `asr.interface.load_audio_16k_mono` / `resample_linear` | WAV PCM16 8000 Hz và 44100 Hz tự sinh: độ dài, mẫu đầu/cuối, sha256 của danh sách float đã lượng tử hoá |
| 6 | `evaluation.submission` | byte JSONL chính xác của một row |

Test thứ hai khẳng định với `FeatureConfig()` (mọi cờ mặc định) cả sáu output
**byte-identical** với golden. Test thứ ba khẳng định sáu module dưới test
**không hề import** `lifemate.features` (quét AST) — độ ghép nối phải bằng 0,
nếu không phép so sánh chỉ là tautology. Test thứ tư làm hỏng từng golden (20
đột biến) và đòi phép so sánh phải bắt được **tất cả**: golden không thể fail
là golden vô nghĩa.

Mọi thành phần đều deterministic ở đây (mỗi test render hai lần rồi mới so
golden). Giá trị duy nhất không deterministic trong payload evaluation là
`timestamp`, và nó không được pin. Không có giới hạn determinism nào khác.

## Ablation runner (gate F23)

`scripts/feature_ablation.py` chạy **cùng một** evaluation hai lần (hoặc N lần),
chỉ đổi bộ cờ, rồi ghi so sánh máy đọc được:

```
PYTHONPATH=src python3 scripts/feature_ablation.py \
  --arms configs/experiments/baseline.yaml configs/experiments/asr_nbest.yaml \
  --limit 20 --out outputs/reports/ablation.json
```

- **Fail fast**: mọi arm được validate qua registry TRƯỚC khi chạy; một arm sai
  (cờ xung đột, tham số ngoài miền, typo) → từ chối chạy, exit 3, không ghi
  report, không tiêu budget.
- **Không tự cài lại evaluation**: mỗi arm gọi `scripts/evaluate.py` và đọc
  `metrics.json` nó ghi ra. Thiếu file → arm `BLOCKED` kèm stderr tail, không
  bao giờ bịa số.
- **Header hằng số giữ cố định**: dataset path, split, model id, checkpoint,
  model mode, seed, `agent.max_steps`, `agent.time_budget_seconds`,
  `agent.temperature`, `agent.max_new_tokens`, `asr.model_id` — đọc từ config
  thật, không tự đặt tên.
- **Report JSON**: mỗi arm có `features` (id đang bật), `fingerprint`,
  `training_fingerprint`, `provenance`, `metrics` thực sự có, và `deltas` so với
  arm ĐẦU TIÊN. Metric không có ở cả hai arm → `0.0` (vắng mặt giống nhau);
  chỉ một bên có → `null` (không so sánh được), không bao giờ giả 0.
- **Tự so sánh là phép thử nhiễu**: chạy `baseline` hai lần phải ra delta 0 toàn
  bộ. Delta khác 0 nghĩa là harness/evaluation không deterministic.
- `--use-mock-env` → `mock_env: true`, `competition_score: false` và ghi rõ
  "NOT competition scores" cả ở stdout và JSON.
- `--dry-run` in kế hoạch + lệnh chính xác của từng arm, **không thực thi gì**
  (chỉ ghi file report JSON).

**Đã nối dây**: `scripts/evaluate.py` đọc section `features:` (qua
`build_feature_config`), nên mỗi arm chạy với đúng bộ cờ của nó. Harness báo
`features_wired: true` / `features_effective: true`. Cặp arm có nghĩa: mọi cặp
overlay trong `configs/experiments/`. Phép thử nhiễu vẫn là `baseline` vs
`baseline` (delta phải bằng 0 toàn bộ).

## Nối dây: hoàn tất

Không còn thay đổi treo. Mọi diff từng được liệt kê ở đây đã áp dụng và được
kiểm chứng bằng test:

- **(a) `features` là section thật của `AppConfig`** — `AppConfig.features`
  tồn tại, `validate()` gọi `REGISTRY.validate_or_raise`, và `to_dict()` dùng
  `FeatureConfig.to_dict()` nên `provenance` không rò rỉ vào `dump_config()`.
  Hệ quả: `load_config()` nhận `features:` trong YAML và
  `configs/experiments/*.yaml` nạp được.
- **(b) Scripts** — `add_feature_arguments` + `build_feature_config` +
  `format_feature_report` có trong `scripts/infer.py`, `scripts/evaluate.py`,
  `scripts/train.py`, `scripts/prepare_dataset.py`.
- **(c) Chặn gipformer-1.5** — đã thay bằng kiểm tra spec-aware
  (`lifemate.asr.model_specs.spec_key`): id đã biết được chấp nhận, id lạ vẫn
  là `ConfigError`. Cờ `asr.gipformer_v15` vì thế có tác dụng thật.
- **(d) `decoding.constrained`** — `scripts/infer.py` truyền cờ lúc dựng
  generator và gọi `set_allowed_tools(...)` mỗi sample.

Kiểm chứng: `tests/test_features.py` (67 passed),
`tests/test_baseline_equivalence.py` (12 passed, baseline byte-identical),
`tests/test_feature_integration.py` (32 passed). `--enable-feature` /
`--disable-feature` chạy được trên cả bốn script.
