# Audit agent + Evaluation trên dữ liệu THẬT (Phase D / H / I)

> Phạm vi: tích hợp `zaic_tools.ToolEnv` chính thức, vòng lặp agent, và tính đúng đắn của evaluation.
> Mọi con số dưới đây đến từ một lệnh đã chạy, được dán nguyên văn. Mục nào chưa chạy được ghi rõ **NOT RUN** kèm lý do.
>
> **Quy ước tuân thủ:** không có đoạn code nào của repo đọc/liệt kê/glob nội dung `train_environment/` hay `public_test_environment/`. Đường dẫn duy nhất được mở là qua `zaic_tools.ToolEnv`. Đã kiểm chứng bằng cách chặn `builtins.open` trong một lượt closed-loop: `state.json opens: 2 | non-zaic_tools openers: []`.

---

## 1. ToolEnv chính thức

### 1.1 Cơ chế import theo từng split

Bộ dữ liệu phát **một bản `zaic_tools` cho mỗi split**, và `apps/` của hai bản khác nhau:

| split | số app | số tool | app đặc trưng |
| --- | --- | --- | --- |
| `train` | 46 | 147 | `robot_vacuum`, `alarm`, `medication`, `lottery` |
| `public_test` | 36 | 122 | `ecommerce`, `grocery`, `stocks`, `public_transit` |

Lệnh: `PYTHONPATH=data/<split> python3 -m zaic_tools --split <split> --root data --apps`
→ `train: 46 apps, 147 tools, 0 not implemented` / `public_test: 36 apps, 122 tools, 0 not implemented`.

Vì vậy **không thể** dùng một bản cài pip cho cả hai split. `src/lifemate/env/official.py:44` (`_import_split_sdk`) thực hiện:

1. Yêu cầu `<data_root>/<split>/zaic_tools` tồn tại;
2. Đưa `<data_root>/<split>` lên **đầu** `sys.path` (xoá mọi lần xuất hiện cũ trước khi chèn — chỉ `insert` thường sẽ để thư mục của split trước thắng);
3. `importlib.invalidate_caches()` rồi import;
4. **Kiểm chứng nguồn gốc**: `Path(module.__file__).parent` phải bằng đúng thư mục của split được yêu cầu;
5. Nếu lệch (Python đã cache bản của split trước), xoá mọi `sys.modules["zaic_tools*"]`, thử lại một lần, nếu vẫn lệch thì **báo lỗi rõ ràng và dừng** — không âm thầm chạy sai bộ app.

Trước khi sửa (P0-1): `import zaic_tools` trơn → `ModuleNotFoundError`; `create_environment('train', <id>, 'data')` → `LifeMateError: zaic_tools not installed`. Trên bộ dữ liệu đầy đủ, inference thi đấu sẽ **exit 2** dù SDK có mặt.

Sau khi sửa, chạy xen kẽ hai split trong **cùng một process**:

```
train train n_tools 147 apps ['alarm', 'banking', 'calendar']
public_test public_test n_tools 122 apps ['banking', 'calendar', 'chat_ott']
train train n_tools 147 apps ['alarm', 'banking', 'calendar']
public_test public_test n_tools 122 apps ['banking', 'calendar', 'chat_ott']
```

Test hồi quy: `tests/test_inference.py::test_official_env_imports_the_requested_splits_copy`.

### 1.2 Quy ước `root`

`ToolEnv(split, id, root=R)` mở `R / split / f"{split}_environment" / id / "state.json"`.
⇒ `R` là thư mục **chứa** các thư mục split (`./data`), **không phải** `data/public_test`.

`scripts/infer.py` và `scripts/evaluate.py` đều truyền `cfg.data.root` (`./data` theo `configs/base.yaml:5`) — đúng. Truyền sai thư mục split giờ bị từ chối tường minh (`test_official_env_root_must_contain_the_split_dir`).

### 1.3 Luật "một ToolEnv cho mỗi mẫu" và defect đã vi phạm

README của BTC: *"Mỗi mẫu dùng một `ToolEnv` riêng và chỉ được tạo một lần"*.

**P0-6 — preflight tạo thêm một session.** `scripts/infer.py` (khối `-- environment preflight`) và `scripts/evaluate.py:226` gọi `create_environment(...)` chỉ để "kiểm tra sớm". Hệ quả đo được bằng cách instrumentation `create_environment`:

```
samples created: 40 | max env instances per sample: 2 | samples created more than once: ['public_0002']
```

Mẫu đầu tiên có **2 instance**, đúng điều bị cấm — mọi call của lần tạo thứ hai không nằm trong transcript nộp bài.

Sửa: thêm `check_official_sdk(split, data_root)` (`src/lifemate/env/official.py:35`) — chỉ import và kiểm chứng nguồn gốc SDK, **không tạo session**. Preflight của cả hai script dùng nó (`scripts/infer.py:538-540`, `scripts/evaluate.py:239-241`); nhánh mock giữ nguyên `create_environment` vì MockToolEnv không bị ràng buộc bởi luật này.

Sau khi sửa:

```
samples created: 40 | max env instances per sample: 1 | samples created more than once: []
rows identical to the live env's get_tool_calls/get_state_hash: 40/40
```

và với evaluator (103 mẫu train):

```
samples: 103 | max env per sample: 1 | repeated: []
executed calls: 283 | rows identical to env: 103 / 103
```

Test hồi quy: `tests/test_inference.py::test_official_preflight_creates_no_session`, `tests/test_eval.py::test_evaluate_closed_loop_builds_one_env_per_sample`.

### 1.4 Định dạng `state_hash`

`zaic_tools/utils.py:state_hash` = `sha256(json.dumps(canon(state), sort_keys=True, ensure_ascii=True, separators=(",", ":")))` → **64 ký tự hex thường**. `canon` loại `_meta`, các app chỉ-đọc (`weather`, `gold_price`, `fuel_price`, `lottery`, `exchange_rate`), và các trường free-text (`label`, `content`, `title`…) vì chúng được chấm trên *arguments*; chuỗi được lower-case, gộp khoảng trắng, `700 cm == 700cm`.

Đo được: 40/40 dòng của submission thật có hash khớp `[0-9a-f]{64}`, 40 hash phân biệt.

---

## 2. Schema parity với validator chính thức

Bộ đo: lấy mọi tool call vàng trong `train.jsonl`, sinh biến thể (bỏ từng tham số `required`, rỗng-hoá từng giá trị chuỗi, đổi kiểu, vi phạm `enum`/`minimum`/`exclusiveMinimum`/`format`/`items`, vi phạm `x-one-of`) rồi so `zaic_tools.validate.schema_error` với `ToolCatalog.validate_call`. Tổng 2156 ca.

| | số ca | |
| --- | --- | --- |
| TRƯỚC khi sửa | 1669 đồng ý / **487 bất đồng** | 22.6 % bất đồng |
| SAU khi sửa | 272 đồng ý / **0 bất đồng** | trên 272 ca vàng + biến thể |

Phân rã 487 bất đồng theo nguyên nhân:

| nguyên nhân | số ca | ví dụ |
| --- | --- | --- |
| giá trị chuỗi rỗng bị coi là "có mặt" | 373 | `alarm.delete {"alarm_id": ""}` — BTC từ chối, local chấp nhận; `appliances.list {"type": ""}` — BTC chấp nhận, local từ chối |
| `exclusiveMinimum` không được kiểm | 33 | `banking.transfer {"amount": 0}` |
| `items.type`/`items.enum` không được kiểm | 28 | `robot_vacuum.start {"rooms": [1]}` |
| `format` (date / date-time / time) không được kiểm | 26 | `tasks.add {"due": "31/12/2026"}` |
| nhóm `x-one-of` lồng nhau hiểu sai | 18 | `music.play {}` — BTC đòi `track_id` hoặc `playlist_id`, local chấp nhận |
| `minimum` không được kiểm | 6 | `appliances.start_cycle {"delay_minutes": -1}` |
| args rỗng cho tool có `required` | 3 | |

Kiểm tra từng extension trên catalog thật (`data/train/train_tools_catalog.json`):

| extension | tool thử | official | local | kết luận |
| --- | --- | --- | --- | --- |
| `x-one-of` | `music.play {}` | từ chối | chấp nhận | MISMATCH |
| `x-one-of` | `music.play {track_id, playlist_id}` | chấp nhận | chấp nhận | AGREE |
| chuỗi rỗng, required | `alarm.delete {"alarm_id": ""}` | từ chối | chấp nhận | MISMATCH |
| chuỗi rỗng, optional | `appliances.list {"type": ""}` | chấp nhận | từ chối | MISMATCH |
| `minimum` | `appliances.start_cycle {delay_minutes: -1}` | từ chối | chấp nhận | MISMATCH |
| `exclusiveMinimum` | `banking.transfer {amount: 0}` | từ chối | chấp nhận | MISMATCH |
| `format: date` | `tasks.add {due: "31/12/2026"}` | từ chối | chấp nhận | MISMATCH |
| `items` | `robot_vacuum.start {rooms: [1]}` | từ chối | chấp nhận | MISMATCH |
| `enum` | `smart_home.light_set {power: "maybe"}` | từ chối | từ chối | AGREE |
| `x-ref` | `smart_home.light_set {device_id: "light_kitchen"}` | chấp nhận | chấp nhận | AGREE |
| `x-match: fuzzy` | `alarm.create {label: "đi chợ"}` | chấp nhận | chấp nhận | AGREE |

Các dòng `MISMATCH` trên là trạng thái **trước** P1-4/D-23 (cột "local" khi đó là catalog). `ĐÃ CHẠY` đo lại sau D-23 trên cùng catalog thật, `schema_error(..., allowed=None)` vs `ToolCatalog.validate_call`, 1545 mẫu / 3541 gold call + biến thể (gold, bỏ/rỗng `required`, số 0/âm/bool/không hữu hạn, bound, `format`, `items`): **27.070 ca → 0 bất đồng** (0 mỗi chiều) — cả 11 dòng extension trên nay đều AGREE; không còn bất đồng nào cần fallback che.

**Sửa (P1-4).** `src/lifemate/data/catalog.py` thuộc worker khác nên không sửa tại đó. Thay vào đó, tại chỗ nối:
* `src/lifemate/agent/tool_parser.py:62` — `parse_model_output` gọi `env.validate` khi env có, thay cho catalog;
* `src/lifemate/agent/tool_executor.py:77` — `execute_validated` tương tự.

Registry **vẫn** quyết định tool nào được phép gọi (danh sách app đã cài); chỉ phần *kiểm tra tham số* chuyển sang env. Catalog validator còn lại là fallback khi không có env (test thuần, mock không catalog).

Bất biến đạt được: **agent không bao giờ từ chối một call mà env chính thức chấp nhận, và ngược lại** — 272/272 ca. Test: `tests/test_agent.py::test_env_validator_is_authoritative_over_catalog`.

---

## 3. Độ trung thực khi replay gold

Mỗi mẫu **một** `ToolEnv`, replay `label.tool_calls` theo thứ tự.

| tập | mẫu | transcript khớp gold | `tool_results` khớp | `final_state` khớp |
| --- | --- | --- | --- | --- |
| chọn theo nhóm hành vi | 13 | 13/13 | 13/13 | 6/6 (mẫu có `final_state`) |
| ngẫu nhiên (seed 0) | 80 | **80/80** | 80/80 | **61/61** |
| 103 mẫu (Task A) | 103 | 103/103 | — | — |

Nhóm hành vi đã phủ: single read (3), read-before-write (3), multi-step (3), gold `sys.ask_missing_info` (3), zero-tool gold (3). Nhóm "gold call có kết quả `ok:false`" **rỗng** — đo trên toàn bộ 1545 mẫu: `total ok:false gold results 0`. Ghi nhận, không bịa.

So `final_state` bằng `zaic_tools.utils.canon` (đúng phép so của scorer): `canon(final_state) equality: 48/48`.

---

## 4. Agent (Phase H)

### 4.1 Lọc tool

Trên 30 mẫu train thật, so `ToolRegistry.for_sample(s).tool_names()` với allow-list của env:

```
[filter] 30 real samples: registry tools not allowed by env=0, allowed-but-hidden=0
```

`installed_apps == []` → registry chỉ còn `sys.ask_missing_info`; prompt chứa meta tool và không có tool của app nào khác.

Hình dạng `installed_apps` thật là `{app: {"tools": [...]}}`; `Context.from_raw` (`src/lifemate/data/schemas.py:89-93`) lấy **key** của mapping, nên registry nhận đúng danh sách app.

**P1-5 — prompt rò tên tool chưa cài.** `src/lifemate/agent/prompt_builder.py:47` (`_RESPONSE_FORMAT_SPEC`) in ví dụ cố định `alarm.list` cho **mọi** mẫu, kể cả mẫu không cài `alarm`. Đo: với mẫu chỉ cài `messaging`, prompt vẫn chứa chuỗi `alarm.list`. Sửa: ví dụ dùng placeholder `<tên_công_cụ>` kèm câu "tên công cụ luôn lấy từ danh sách Công cụ khả dụng". Test: `test_prompt_mentions_no_tool_of_an_uninstalled_app`.

### 4.2 Giới hạn vòng lặp (cấu hình đang ship)

| tham số | giá trị | nguồn |
| --- | --- | --- |
| `max_steps` | 8 | `configs/inference.yaml:19` |
| `time_budget_seconds` | 120.0 | `configs/inference.yaml:20` |
| `max_consecutive_failures` | 2 | `StoppingPolicy` default |
| `max_repeated_identical_calls` | 2 | `StoppingPolicy` default |

Hành vi đo được: lặp đúng một action → `status=repeated_failure reason=invalid_repeat llm_calls=3 env_calls=3`; `max_steps=3` → `status=max_steps` với đúng 3 lệnh LLM. Observation của bước *k* có mặt trong prompt bước *k+1* (kiểm bằng cách so payload đã render với prompt).

### 4.3 Output lỗi

7 lớp output lỗi đã thử: không có `<action>`, JSON hỏng, `arguments` không phải object, tool không tồn tại, thiếu tham số bắt buộc, sai kiểu tham số, tool của app chưa cài.

Với **mỗi** lớp: `result.tool_calls == []`, observation là `{"ok": false, "skipped": true}`, và lỗi được ghi vào `result.errors` — không lớp nào chạm tới env, không lớp nào bị tính là thành công. Hai lỗi liên tiếp → dừng với `repeated_failure` (streak = 2). Test: `test_malformed_output_never_counts_as_success`, `test_consecutive_malformed_outputs_stop_the_loop`.

### 4.4 Đường submission

`{"id", "tool_calls", "state_hash"}` lấy **từ env** (`env.get_tool_calls()` / `env.get_state_hash()`), có `assert_env_sourced` đối chiếu canonical JSON + hash, và `validate_submission(..., require_state_hash=True, calls_source="env", require_order=True)`.

Lần chạy thật trên `public_test` (40 mẫu, `--asr none`, generator kịch bản):

```
wrote 40 row(s) to /tmp/sub_real.jsonl in 0.1s (0 sample(s) with failures, 0 ASR failure(s))   EXIT=0
```

```
40 lines checked, 0 schema errors                                                              EXIT=0     # CLI của BTC
```

Kiểm tra nội dung: 40 dòng, **đúng thứ tự dataset**, mọi `state_hash` là 64 hex, 40 hash phân biệt, mọi dòng đúng khoá `{id, tool_calls, state_hash}`. So sâu 3 dòng với env đang sống (`public_0004`, `public_0005`, `public_0006`): `tool_calls` **giống hệt**, hash bằng nhau.

Kịch bản (chọn từ catalog, không xem label, không xem `*_environment`): theo `i % 5` trên 40 mẫu đầu — 8 finish-only, 8 `sys.ask_missing_info(alarm.create,[time,label])`, 8 một read, 8 hai read, 8 read + gọi tool không tồn tại + read (để đi qua nhánh bị từ chối).

`--resume`: lượt 1 `--limit 10` → 10 dòng; lượt 2 `--limit 40 --resume` → `resume: 10 row(s) already present`, ghi 40 dòng, 10 dòng đầu **giữ nguyên từng byte**, 30 dòng mới nối đúng thứ tự dataset. Sửa id dòng đầu thành `public_9999` → `ERROR: --resume cannot reuse ...: row 0 has id 'public_9999', expected dataset id 'public_0002'` **exit 4**.

Hàng rào mock (giữ nguyên sau khi thêm opt-in): `--use-mock-env --output outputs/submission.jsonl` → `ERROR: --use-mock-env refuses to write the default official output path` **exit 4**, file không được tạo. Một lượt có `--generator` cũng không thể ghi vào `outputs/submission.jsonl` (`scripts/infer.py:483-487`).

---

## 5. Defect prompt đã sửa (Phase D, hai defect do review chỉ ra)

### 5.1 Lượt user trong history render trắng (D-1)

`src/lifemate/agent/prompt_builder.py:163` (`_turn_text`). Trước: chỉ đọc `content`, mà history thật là `{"role": "user", "audio": "..._u1.wav", "audio_text": "..."}`, nên mọi lượt user thành chuỗi rỗng.

Trước (mẫu thật `train_0008`):

```
Người dùng: 
Trợ lý: Mai có 2 báo thức: 06:15 và 07:00.
Người dùng: 
Trợ lý: Đã xoá báo thức 06:15.
```

Sau:

```
Người dùng: Mai tôi có những báo thức nào vậy?
Trợ lý: Mai có 2 báo thức: 06:15 và 07:00.
Người dùng: Xóa cái sớm nhất giúp tôi nhé.
Trợ lý: Đã xoá báo thức 06:15.
```

Đo trên 300 mẫu train: `user turns rendered=632 blank=0`.

Không bịa transcript: lượt chỉ có `audio` (không có text) render đúng nhãn `Người dùng:` và không thêm gì. Test: `test_history_user_turn_with_audio_text_is_rendered`, `test_history_turn_with_only_audio_renders_nothing_extra`.

### 5.2 Payload tool-result bị mất (D-2)

Nguyên nhân gốc nằm ở **parser**, không phải renderer: `HistoryTurn.from_raw` → `ToolResult.to_dict()` chỉ trả `{name, ok, result}` nên `{"ok": true, "alarms": [...]}` thành `{"name": null, "ok": true, "result": null}`.

Trước (`train_0008`):

```
  gọi: <action>{"tool_calls": [{"arguments": {}, "name": "alarm.list"}]}</action>
  kết quả: {"tool_results": [{"name": null, "ok": true, "result": null}]}
```

Sau:

```
  gọi: <action>{"type": "tool_call", "name": "alarm.list", "arguments": {}}</action>
  kết quả: {"alarms": [{"enabled": true, "id": "alm_01", "label": "họp sớm", "repeat": "daily", "time": "2026-11-14T06:15:00+07:00"}, {"enabled": true, "id": "alm_02", ...}], "ok": true}
```

Phần của renderer (`prompt_builder.py:210-254`, `256`): render **một cặp `gọi:`/`kết quả:` cho mỗi call** (trước đây render nguyên khối `{tool_calls: [...]}`), đọc cả khoá thô `tool_calls`/`tool_results` khi gặp dict, dùng bản ghi thô khi shape đã parse không còn payload, và bóc wrapper `{name, ok, result}` về đúng shape của `env.call()`.

Bất biến kiểm chứng:
* payload trong history render **nguyên văn** bản ghi: **3661/3661** lượt;
* một lượt đã parse và một lượt thô render **giống hệt**: **1818/1818**;
* text của history và text của bước sống (prior step) **giống hệt** cho cùng call+result: **206/206**.

Thêm: `_call_log_entry_to_step` (`prompt_builder.py:86`) không còn tự tạo `ok: true` cho entry `{name, arguments}` của log chính thức — log đó **không có** kết quả (định dạng nộp bài không có trường result), nên bịa `ok` là sai; `StateTracker` mới là nơi giữ return value sống.

Lưu ý phương pháp: replay một call *lịch sử* trên env hiện tại **không** tái tạo kết quả đã ghi (snapshot history là trạng thái tại lượt đó; `train_0008` `alarm.delete` giờ trả `done:false`). Bất biến đúng là *render == bản ghi*, không phải *render == `env.call()` hôm nay*.

---

## 6. Evaluation (Phase I)

### 6.1 Chín kiểm tra rò rỉ / đúng đắn — đã chạy

| # | kiểm tra | cách chạy | kết quả |
| --- | --- | --- | --- |
| 1 | gold không vào prompt | generator `Echo` ghi lại messages | `prompt sent: ['dat bao thuc 6h30'] | gold absent: True` |
| 2 | `tool_calls` không bị coi là gold | `evaluator._is_gradeable` trên dòng dạng prediction | `gradeable: False n_records: 0 metrics.available: False`; `_GOLD_KEYS = ('gold_tool_calls','expected_tool_calls')` |
| 3 | crash / không có `<action>` = 0 | generator ném `RuntimeError` | aggregate `{'available': True, 'value': 0.0, 'n': 1}`; per-record `('s1', False, 'RuntimeError: cuda oom')` |
| 4 | per-record khớp aggregate | 2 record, 1 đúng 1 sai | `aggregate: 0.5 per-record mean: 0.5 agree: True` |
| 5 | `val.jsonl` không bị nhầm với predictions | `_is_gradeable` trên 2 loại | val: `True`; prediction: `False` |
| 6 | checkpoint thật sự được nạp | thay `load_model_and_tokenizer` bằng stub ghi tham số | `loader calls: [('Qwen/Qwen3.5-4B', {... 'checkpoint': '/tmp/ckpt_x', 'tuning_mode': 'full' ...})]`, exact_match `1.0` |
| 7 | teacher-forced và closed-loop tách biệt | đọc `evaluate_split`: hai khoá `teacher_forced` / `closed_loop` | không dùng chung khoá metric (`source = teacher_forced if ... else closed_loop`) |
| 8 | call-sequence vs final-state | chạy closed-loop thật | `final_state_correctness: {'available': False, 'reason': 'official scoring tooling unavailable'}` |
| 9 | oracle trên split test | `--split public_test --asr oracle --allow-oracle` | `ERROR: --asr oracle is forbidden for split 'public_test' ...` **exit 2**, không ghi file |

Kiểm tra bổ sung: không đường eval nào đọc `*_environment/` — chặn `builtins.open` trong một lượt closed-loop: `state.json opens: 2 | non-zaic_tools openers: []`.

### 6.2 Crash → 0, có `n_unmeasured`

Trên 56 record dựng từ 15 mẫu train thật:

| generator | `tool_call_exact_match` | `n_unmeasured` | per-record |
| --- | --- | --- | --- |
| oracle (trả đúng `target`) | **1.0** (n=56) | 0 | tất cả `True` |
| file dự đoán bị làm hỏng | 0.2679 | 0 | — |
| crash | **0.0** | **56** | tất cả `False` |

Đây là **evaluator self-check on gold input, KHÔNG phải điểm của model**. Cơ chế: `metrics.decision_emitted` (`src/lifemate/evaluation/metrics.py:374`) chặn cả aggregate và per-record.

### 6.3 Oracle barrier

* `public_test` (không nhãn) + `--allow-oracle` → **exit 2**, không có file → đúng.
* `train` (có nhãn) + `--allow-oracle` → chạy được: `WARNING: oracle ASR is active on labeled split 'train'`, 5 dòng, CLI BTC `5 lines checked, 0 schema errors`. Debug dump cho thấy `asr.backend=oracle`, và `asr.raw_transcript == dataset audio_text` cho **5/5** mẫu (`train_0001`…`train_0005`).

### 6.4 Kết quả closed-loop trên `train` thật (Task A)

Bộ: 103 mẫu đầu của `train` (trong đó **60 mẫu có ≥3 gold call**, tổng 283 gold call), một `ToolEnv` mỗi mẫu, generator kịch bản.

**Lượt 1 — kịch bản GOLD. Nhãn: evaluator self-check on gold input, KHÔNG phải điểm của model.**

| chỉ số | giá trị |
| --- | --- |
| `n_run` | 103 |
| `status_counts` | `{'finished': 103}` |
| `tool_call_exact_match` | **1.0000** (n=103) |
| `sequence_f1` / precision / recall | 1.0000 / 1.0000 / 1.0000 |
| `argument_exact_match` | 1.0000 (n=283) |
| `tool_selection_accuracy` | 1.0000 |
| `stop_accuracy` | 1.0000 |
| `call_count_delta` | 0.0 (`total_delta=0`) |
| tool call đã thực thi | 283, **ok 283 (100 %)**, 0 bị từ chối schema |
| invalid / bị chặn trước env | 0 |
| bước LLM | 386 |
| latency trung bình mỗi mẫu | 0.0026 s (scripted, không có model) |
| hợp lệ theo CLI BTC | `103 lines checked, 0 schema errors` |
| `final_state_correctness` | `available: False` |

**Lượt 2 — kịch bản bị làm hỏng (giữ số call, đổi tên tool sang một read hợp lệ nhưng sai).**

| chỉ số | gold | corrupt |
| --- | --- | --- |
| `tool_call_exact_match` | 1.0000 | **0.0583** |
| `sequence_f1` | 1.0000 | **0.0715** |
| `tool_selection_accuracy` | 1.0000 | **0.0680** |
| `stop_accuracy` | 1.0000 | 0.6990 |
| `call_count_delta` | 0.0 | −0.4660 (total −48) |
| `status_counts` | `{'finished': 103}` | `{'repeated_failure': 60, 'finished': 43}` |
| `n_failures` | 0 | 60 |
| `final_state_correctness` | `available: False` | `available: False` |

Kết luận: evaluator phân biệt được đúng/sai rõ ràng; và khi tool bị dùng lặp lại đến mức vi phạm schema, vòng lặp dừng bằng `repeated_failure` chứ không âm thầm "thành công".

**Final-state scorer:** không có và **không thể tính** — BTC chỉ phát hành `zaic_tools` (runtime) tại thời điểm này; script chấm điểm cuối chưa có trong gói. Vì vậy `final_state_correctness` giữ `{"available": False, "reason": "official scoring tooling unavailable"}` ở **mọi** artifact (`evaluator.py:485`, `evaluator.py:659-661`, ghi chú `evaluator.py:678`). Không có nơi nào điền số thay thế. Phần *state* vẫn được kiểm chứng trong audit này bằng phép so `canon(final_state)` (mục 3) — đó là kiểm tra offline, không phải metric của evaluator.

---

## 7. Giới hạn còn lại

| giới hạn | số đo | hệ quả |
| --- | --- | --- |
| History của `public_test` chỉ có `audio`, không có text | 634 mẫu, 419 mẫu có history, **1299/1299 lượt user không có `audio_text`** → **419/634 (66.1 %)** mẫu (100 % mẫu có history) bước vào agent với lượt trước không đọc được | Prompt inference nghèo ngữ cảnh hơn prompt train. Không bịa transcript; repo không transcribe audio history. Đây là khoảng trống năng lực, không phải bug đã sửa |
| Không có suy luận model thật | `torch.cuda.is_available() == False`, torch `2.15.0+cpu`, không có byte trọng số Qwen3.5-4B | **NOT RUN**: ba baseline (base / full SFT / LoRA SFT). Lệnh đã chuẩn bị; chúng dừng ở loader với exit 3 |
| Không có scorer final-state chính thức | — | `final_state_correctness` luôn `available: False` (mục 6.4) |
| 0 gold call trong `train` cho kết quả `ok:false` | 0/1545 mẫu | Nhánh "gold call bị env từ chối" không có dữ liệu vàng để kiểm; đã kiểm bằng biến thể tổng hợp (mục 2) |
| Vòng lặp với model thật | — | **NOT RUN**; hành vi vòng lặp chỉ được kiểm bằng generator kịch bản |

### Ba lệnh baseline (chuẩn bị, chưa chạy)

```bash
# 1. base
python3 scripts/infer.py --split public_test --data-root data --asr gipformer \
  --history-transcripts outputs/asr/history_transcripts.jsonl \
  --model-mode base --output outputs/submission_base.jsonl
# 2. full SFT
python3 scripts/infer.py --split public_test --data-root data --asr gipformer \
  --history-transcripts outputs/asr/history_transcripts.jsonl \
  --model-mode full --checkpoint outputs/checkpoints/full --output outputs/submission_full.jsonl
# 3. LoRA SFT
python3 scripts/infer.py --split public_test --data-root data --asr gipformer \
  --history-transcripts outputs/asr/history_transcripts.jsonl \
  --model-mode lora --checkpoint outputs/checkpoints/lora --output outputs/submission_lora.jsonl
# kiểm tra chung
PYTHONPATH=data/public_test python3 -m zaic_tools --split public_test --root data --validate <file>
```

---

## 8. Thay đổi mã và test hồi quy

| file:line | thay đổi | test |
| --- | --- | --- |
| `src/lifemate/env/official.py:35` | `check_official_sdk` — preflight không tạo session | `test_official_preflight_creates_no_session` |
| `src/lifemate/env/official.py:44` | import theo split + kiểm chứng nguồn gốc | `test_official_env_imports_the_requested_splits_copy` |
| `src/lifemate/env/official.py:113` | giữ `raw["error"]` khi `validate` trả `{ok,error}` | `test_env_validate_false_with_no_error_list_is_still_a_rejection` |
| `src/lifemate/env/official.py:150,155` | `get_state`, `available_tools` theo allow-list của session | `test_official_env_call_rejections_are_logged_and_matchn_gold` |
| `src/lifemate/agent/tool_parser.py:62` | `env.validate` là chuẩn | `test_env_validator_is_authoritative_over_catalog` |
| `src/lifemate/agent/tool_executor.py:77` | như trên cho `execute_validated` | `test_env_validator_is_authoritative_over_catalog` |
| `src/lifemate/agent/prompt_builder.py:47` | bỏ `alarm.list` khỏi ví dụ định dạng | `test_prompt_mentions_no_tool_of_an_uninstalled_app` |
| `src/lifemate/agent/prompt_builder.py:86` | không bịa `ok:true` từ call log | `test_call_log_entry_without_result_never_fabricates_ok` |
| `src/lifemate/agent/prompt_builder.py:163,210,256` | history render trung thực (text + payload) | 5 test `test_history_*` |
| `scripts/infer.py:223,485,538` | opt-in `--allow-scripted-official`, chặn output chính thức, preflight không tạo session | `test_scripted_official_opt_in_and_official_output_barrier`, `test_scripted_run_on_real_split_is_accepted_by_the_official_validator` |
| `scripts/evaluate.py:135,239` | opt-in tương đương + preflight không tạo session | `test_evaluate_scripted_opt_in_requires_mock_or_explicit_flag`, `test_evaluate_closed_loop_builds_one_env_per_sample` |

### Trạng thái test

| lệnh | kết quả |
| --- | --- |
| `python3 tests/run_tests.py` | `passed=439 failed=0 skipped(needs fixtures)=6` |
| `.venv/bin/python -m pytest -q` | `439 passed, 6 skipped` |
| `/tmp/torchenv/bin/python -m pytest -q` | `444 passed, 1 skipped` |
