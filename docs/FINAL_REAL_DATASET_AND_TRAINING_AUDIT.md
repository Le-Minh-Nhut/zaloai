# Audit cuối — Dữ liệu THẬT, SFT, Agent, ASR và Sẵn sàng huấn luyện

**Repo:** `~/data/zaloai` — Zalo AI Challenge 2026, Track 1 "LifeMate"
**Ngày đo:** 2026-11-08 · **Release dữ liệu:** v3.2.0 (build `2026-10-24T23:35:25+0700`)
**Phạm vi:** toàn bộ đường đi từ zip phát hành → EDA → SFT → ToolEnv chính thức → agent → evaluation → ASR → cấu hình huấn luyện.

## 0. Quy ước bằng chứng và phạm vi tuân thủ

Bốn nhãn được dùng xuyên suốt, đúng chữ:

| nhãn | nghĩa |
|---|---|
| `ĐÃ CHẠY` | một lệnh đã chạy trong phiên này hoặc trong phiên đo của worker, output được dẫn lại |
| `ĐỌC TỪ FILE` | đọc từ một `file:line` hoặc một artifact có đường dẫn nêu rõ |
| `SUY RA` | suy luận số học/từ dữ kiện đã đo, không phải phép đo trực tiếp |
| `CHƯA ĐO` | không đo được ở đây, kèm lý do |

**Tuân thủ:** không có lệnh nào trong phiên này đọc, liệt kê hay glob nội dung `train_environment/` hoặc `public_test_environment/`. Dữ liệu app chỉ được chạm **qua** `zaic_tools.ToolEnv`. Kiểm chứng độc lập trong audit agent: chặn `builtins.open` trong một lượt closed-loop → `state.json opens: 2 | non-zaic_tools openers: []` (`ĐỌC TỪ FILE` — `docs/REAL_DATA_AGENT_EVAL_AUDIT.md` §6.1). Cờ `has_environment_dir` (D-25) là **boolean từ `Path.is_dir()`**, không mở file và không liệt kê nội dung — nên tuyên bố tuân thủ này không bị ảnh hưởng bởi việc cờ đó nay là `True` cho cả hai split.

**Giới hạn môi trường đo (áp cho toàn bộ tài liệu):**

| hạng mục | trạng thái | bằng chứng |
|---|---|---|
| GPU | có: `NVIDIA GeForce RTX 5070 Ti`, 16303 MiB, driver 580.173.02 | `ĐÃ CHẠY` `nvidia-smi` |
| torch (CPU) | `2.15.0+cpu`, `torch.cuda.is_available() == False`, `device_count == 0` — **lịch sử**, đo trên `/tmp/torchenv` trước khi venv đó bị xoá ngày 2026-11-13 | `ĐÃ CHẠY` (trước khi xoá) |
| torch trong `.venv` repo | không cài (`No module named 'torch'`) | `ĐÃ CHẠY` `python3 -c "import torch"` |
| trọng số Qwen3.5-4B | **không có byte trọng số**; chỉ có `config.json`, `model.safetensors.index.json`, tokenizer trong HF cache | `ĐÃ CHẠY` `ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*` |
| Docker daemon | không có; và image cũ ghim `cu124` không có kernel `sm_120` nên cũng không chạy được GPU ở đây dù có daemon (pin đã sửa, image chưa build) | `ĐÃ CHẠY` `docker info` → fail |
| `wandb` | không cài | `ĐÃ CHẠY` `import wandb` → `ModuleNotFoundError` |
| `sherpa_onnx` + asset Gipformer int8 | **có**, 6 file, hash-verified | `ĐÃ CHẠY` `/tmp/asrenv`, `/tmp/dl.log` |
| dữ liệu thật | **có**, `data/train` 1545 + `data/public_test` 634 | `ĐÃ CHẠY` |

---

## A. Trích xuất dataset

### A.1 Gói phát hành và vị trí

| mục | giá trị | nguồn |
|---|---|---|
| file zip | `train+public_test.zip` | `ĐÃ CHẠY` `ls` |
| kích thước zip | 876.430.231 byte = **835,8 MiB** | `ĐÃ CHẠY` `stat -c %s` |
| giải nén tới | `data/train`, `data/public_test` | `ĐÃ CHẠY` |
| dung lượng trên đĩa | `data/train` **794 MiB**, `data/public_test` **321 MiB** | `ĐÃ CHẠY` `du -sh` |
| release | `v3.2.0`, build `2026-10-24T23:35:25+0700`, `train 1545 / public_test 634` | `ĐỌC TỪ FILE` `data/VERSION` |
| `private_test` | **không có** trong gói (đúng thiết kế: phát sau khi nộp final solution) | `ĐÃ CHẠY` `ls data` + `ĐỌC TỪ FILE` `data/README.md` §1 |

### A.2 Số mẫu, file âm thanh, catalog

| mục | train | public_test | nguồn |
|---|---|---|---|
| dòng JSONL | **1545** | **634** | `ĐÃ CHẠY` đếm dòng |
| `id` duy nhất | 1545 | — | `ĐÃ CHẠY`; EDA `duplicate_ids: {}` |
| file `.wav` | **5036** | **1933** | `ĐÃ CHẠY` `find data/<split>/audio -name '*.wav' \| wc -l` |
| app trong catalog | **46** | **36** | `ĐÃ CHẠY` đọc `*_tools_catalog.json` |
| tool trong catalog | **147** | **122** | `ĐÃ CHẠY` |
| phân bố `kind` | write 84 / read 62 / meta 1 | write 69 / read 52 / meta 1 | `ĐÃ CHẠY` |
| SDK CLI tự khai | `46 apps, 147 tools, 0 not implemented` | `36 apps, 122 tools, 0 not implemented` | `ĐÃ CHẠY` `python3 -m zaic_tools --split <s> --root data --apps` |

**Hai split phát hai bản `zaic_tools` khác nhau.** Đây không phải chi tiết phụ — nó là nguyên nhân gốc của defect D-01.

| quan hệ app | số app | ví dụ |
|---|---|---|
| chỉ có ở `train` | **21** | `robot_vacuum`, `alarm`, `medication`, `lottery`, `door_lock`, `fitness`, `library` |
| chỉ có ở `public_test` | **11** | `ecommerce`, `grocery`, `stocks`, `public_transit`, `pantry`, `recipes`, `parking` |
| dùng chung | **25** | `banking`, `calendar`, `sms`, `smart_home`, `music`, … |

`ĐÃ CHẠY` — so `id` của `apps[]` trong hai catalog. Hệ quả trực tiếp: **21 app chỉ có ở `train`** (`46 − 25` dùng chung) không bao giờ xuất hiện ở `public_test`, nên mọi tool của chúng là **out-of-distribution** khi suy luận thi đấu; ngược lại **11 app chỉ có ở `public_test`** (`36 − 25`) **chưa từng có một mẫu huấn luyện nào** (`SUY RA` từ chính hai tập app, không phải suy đoán về chất lượng model).

### A.3 Cấu trúc một mẫu (đọc từ `data/README.md` và JSONL thật)

`train`: `{id, audio, audio_text, context, history, label{tool_calls, tool_results, final_state}}`
`public_test`: `{id, audio, context, history}` — **không có `label`, không có `audio_text`** (`ĐÃ CHẠY`: 634/634 dòng, kiểm bằng `any('label' in x)`/`any('audio_text' in x)` → `False` cả hai).

Do đó **không tồn tại điểm số nào cho `public_test`** trong repo này: không nhãn thì không có metric. Mọi con số chất lượng trong tài liệu này là (i) đo trên `train`, hoặc (ii) self-check của evaluator trên gold, và được ghi nhãn rõ.

---

## B. EDA thật

Nguồn: `outputs/eda/report.md`, `outputs/eda/summary.json`, `outputs/eda/tables/*.csv`, 11 file chart (`charts_manifest.json` + 10 PNG).

### B.1 Âm thanh (n = 1545)

| chỉ số | giá trị | nguồn |
|---|---|---|
| file phân tích / OK / lỗi | 1545 / 1545 / **0** | `ĐỌC TỪ FILE` `summary.audio` |
| `n_missing_audio_refs` | 0 | `ĐỌC TỪ FILE` |
| sample rate | `{16000: 1545}` | `ĐỌC TỪ FILE` |
| bit depth | `{16: 1545}` | `ĐỌC TỪ FILE` |
| kênh | `{1: 1545}` (mono) | `ĐỌC TỪ FILE` |
| thời lượng trung bình | **7,227 s** (median 6,937; min 1,743; max 22,334; p95 11,836; stdev 2,664) | `ĐỌC TỪ FILE` |
| tổng thời lượng | **3,102 h** = 11.165,5 s | `ĐỌC TỪ FILE` |
| tương quan thời lượng ↔ số từ | r = 0,843 (n = 1545), chỉ mô tả | `ĐỌC TỪ FILE` |
| cờ bất thường | none | `ĐỌC TỪ FILE` `flags: none` |

Không có file hỏng, không có sample rate lạ, không clipping được gắn cờ. **1545/1545 mẫu dùng được cho ASR mà không cần tiền xử lý.**

### B.2 Độ dài transcript (tokenizer THẬT `Qwen/Qwen3.5-4B`)

| chỉ số | giá trị |
|---|---|
| n | 1545 |
| mean / median | **26,87** / 27 |
| min / max | 5 / 77 |
| p95 | 43 |

`ĐỌC TỪ FILE` `outputs/eda/report.md`. Đây là token của **lời yêu cầu cuối**, chưa gồm prompt. Ước lượng `char/4` của script (`mean 3735,2`) **đếm thiếu khoảng 2,5 lần** so với token thật (prompt thật mean 5042,2) — chính vì vậy mọi quyết định cắt chuỗi trong tài liệu này dùng token thật.

### B.3 Thống kê tool

| chỉ số | giá trị | nguồn |
|---|---|---|
| app / tool | 46 / 147 | `ĐỌC TỪ FILE` |
| `kind` | meta 1, read 62, write 84 | `ĐỌC TỪ FILE` |
| kiểu tham số | string 204, integer 25, array 10, boolean 6, number 1 | `ĐỌC TỪ FILE` |
| annotation `x-ref` | **108** (38 đích không resolve được) | `ĐỌC TỪ FILE` |
| tool có `x-one-of` | **6**: `contacts.update`, `phone.call`, `sms.send`, `ewallet.transfer`, `music.play`, `podcast_radio.play` | `ĐÃ CHẠY` tự đếm trên catalog |
| cạnh phụ thuộc (dependency edges) | 108 | `ĐỌC TỪ FILE` |

`music.add_to_playlist` **không** có `x-one-of` (chỉ `music.play` có) — quan trọng cho mục D.3.

### B.4 Quỹ đạo gold (n = 1545)

| chỉ số | giá trị | nguồn |
|---|---|---|
| zero-call / single / multi | **111** / **369** / **1065** | `ĐỌC TỪ FILE` `summary.trajectory` |
| số call: mean / median / max / p95 | 2,292 / 2 / **6** / 5 | `ĐỌC TỪ FILE` |
| tổng gold call | **3541** | `ĐÃ CHẠY` tự đếm |
| `sys.ask_missing_info` | **217** (14,0%), **217/217 là call cuối** | `ĐỌC TỪ FILE` + `ĐÃ CHẠY` |
| out-of-scope (0 call) | **111** (7,2%) | `ĐỌC TỪ FILE` |
| gold result lỗi (`ok:false`) | **0 / 3541 (0,0%)** | `ĐÃ CHẠY` |
| read-before-write thoả | **676 / 1172** mẫu có write | `ĐỌC TỪ FILE` |
| độ phức tạp tham số | 1,289 arg/call (mean) | `ĐỌC TỪ FILE` |
| tool khác nhau xuất hiện trong gold | **146 / 147** (`contacts.get` không bao giờ được gọi) | `ĐÃ CHẠY` |

### B.5 Challenge tags (heuristic, **không** phải nhãn của BTC)

| tag | mẫu | tỉ lệ |
|---|---|---|
| `missing_information` | 217 | 14,0% |
| `reference_lookup` | 1300 | 84,1% |
| `fuzzy_match` | 1300 | 84,1% |
| `multi_app` | 791 | 51,2% |
| `temporal_expression` | 734 | 47,5% |
| `long_history` (≥8 lượt) | 326 | 21,1% |
| `ambiguous_entity` | 265 | 17,2% |

`ĐỌC TỪ FILE` `outputs/eda/report.md`. Một mẫu có thể mang nhiều tag; tag do regex/heuristic sinh, không có ý nghĩa chấm điểm.

### B.6 Mất cân bằng

- Tool phổ biến nhất `contacts.search`: **6,6%** tổng số call; 15 tool ≤1% (`ewallet.transfer`, `bike_rental.reserve`, `door_lock.set_lock`, `subscriptions.cancel`, `food_delivery.place_order`, …) `ĐỌC TỪ FILE`.
- Mẫu zero-call: 7,2%; số app đã cài trải từ 3 đến 18 (`{3: 10, …, 18: 2}`) `ĐỌC TỪ FILE`.
- Top chuỗi: `smart_home.get_status > smart_home.light_set` ×78; `contacts.search > sys.ask_missing_info` ×74; `banking.get_balance > banking.list_beneficiaries > banking.transfer` ×55 `ĐỌC TỪ FILE`.

### B.7 Ý nghĩa cho huấn luyện

| quan sát | hệ quả |
|---|---|
| 6 call là mức tối đa; 68,3% decision point nằm sau ≥1 observation | cần ngân sách chuỗi dài, không phải single-shot |
| 84,1% mẫu có tham số dạng tham chiếu | model buộc phải copy ID từ kết quả tool, không được bịa |
| 0 gold result lỗi | **không có** supervision cho phục hồi lỗi; muốn có phải augment |
| 111 mẫu out-of-scope chỉ là `finish` trần | học được *hành vi* từ chối, không học được *câu chữ* từ chối |
| 11 app của `public_test` vắng mặt trong train | tool selection trên 11/36 app của test là zero-shot |

---

## C. Dataset SFT

Lệnh sinh (exit 0): `HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/prepare_dataset.py --data-root ./data --split train --output outputs/datasets --tokenizer Qwen/Qwen3.5-4B` (`ĐỌC TỪ FILE` — đầu `outputs/datasets/DATASET_PREPARATION_AUDIT.md`). Lệnh gốc ghi `/tmp/torchenv/bin/python`; venv đó **đã bị xoá ngày 2026-11-13**, nên đường dẫn được trỏ lại `/tmp/cudaenv` (interpreter duy nhất còn `transformers`, 5.22.2).

Prompt được render bằng `enable_thinking=False`, vì mặc định (không truyền cờ) template Qwen3.5 để **think block mở mà không đóng** (`…assistant\n<think>\n`) — xem D-24; tail đúng là dạng đã đóng `…assistant\n thinking\n\n</think>\n\n`.

### C.1 Số lượng

| mục | train | val | nguồn |
|---|---|---|---|
| mẫu | **1389** | **156** | `ĐÃ CHẠY` `/manifest.json` |
| decision point | **4368** | **482** | `ĐÃ CHẠY` đếm dòng `train.jsonl`/`val.jsonl` |
| mẫu bị bỏ | **0** | **0** | `ĐÃ CHẠY` `stats.json` (6 lý do, tất cả = 0) |
| decision point bị bỏ | **17** | **2** | `ĐÃ CHẠY` `manifest.n_train_dropped_decision_points`/`n_val_dropped_decision_points`, lý do `unobservable_ref_target` |
| tỉ lệ / seed / history_limit | 0,1 / 42 / 10 | | `ĐỌC TỪ FILE` |
| `n_groups` | 1544 | | `ĐỌC TỪ FILE` |
| `group_hash` | `53a8c3536ac6eee4ed245f0fb33c06a54585e9e4919cd6b13f6492ae9d2f4a9e` | | `ĐÃ CHẠY` khớp manifest |
| train ∩ val (sample id) | **0** | | `ĐÃ CHẠY` |

`visibility_limited` = 0; `unobservable_ref` = **12 mẫu (train) / 1 (val)** — sau khi sửa phép đếm (xem dưới), **19 decision point bị bỏ** (17 train / 2 val, lý do `unobservable_ref_target`), **0 mẫu bị bỏ**.

**Số cũ `319` mẫu / `943` decision point không phải defect count — nó là over-flag.** Cờ cũ tìm trong **toàn bộ** history, trong khi prompt chỉ render `history_limit=10` lượt cuối, và còn coi cả free-text `x-match` là tham chiếu. Đo lại với đúng cửa sổ prompt render (`ĐÃ CHẠY`, `_visible_text` + `_unobservable_ref_args`): trong **496 mẫu** read-before-write có tham số dạng tham chiếu chưa thoả,

| nguồn quan sát được của giá trị | số mẫu |
|---|---|
| lấy từ một **tool observation** trước đó hoặc từ một **action** trước đó (`# Các bước đã thực hiện`) | **182** |
| chỉ nằm trong **history hội thoại** (trong cửa sổ 10 lượt) | **65** |
| lấy từ **enum của catalog** | **118** |
| **không** mang tham số tham chiếu nào | **208** |
| **thật sự không quan sát được** | **2** |

→ `496` chưa bao giờ là số defect. Sau fix: 12 mẫu còn cờ, 19 decision point bị bỏ, 0 mẫu bị bỏ. Một ca là lỗi nhãn phía operator: **`train_1696` `chat_ott.send_message.chat_id=chat_6142`** — giá trị này không tồn tại ở bất kỳ đâu trong record (không có trong request, context, 10 lượt history cuối, và cả 22 lượt history); target bị bỏ, phần còn lại của trajectory được giữ.

### C.2 Thành phần target (train)

| target | số | ghi chú |
|---|---|---|
| `tool_call` (không tính meta) | **2979** | `3170` target chứa `"type": "tool_call"`, trừ 191 meta |
| `finish` | **1198** | gồm 100/100 mẫu zero-call của train |
| `sys.ask_missing_info` | **191** | là điểm kết thúc, **không** sinh `finish` sau đó |
| tổng | **4368** | `ĐÃ CHẠY` (2979 + 1198 + 191) |

`step_index > 0` (prompt phải có prior observation): **2989/4368 = 68,43%** (`ĐÃ CHẠY`, đếm trực tiếp trên `train.jsonl` đã rebuild).

Histogram `step_index` (train): `{0: 1379, 1: 1228, 2: 893, 3: 524, 4: 235, 5: 102, 6: 7}`; 1581 target string phân biệt (tham số free-text → gần như duy nhất, không phải lỗi) `ĐÃ CHẠY`.

Trên **train+val**, số decision point có target là `tool_call`: **3305** (train 2979 + val 326); cộng 217 `ask_missing_info` = 3522 (`ĐÃ CHẠY`).

### C.3 Căn chỉnh (alignment) và tính nhân quả

| kiểm tra | kết quả | nguồn |
|---|---|---|
| `len(tool_calls) == len(tool_results)` | **1545/1545** | `ĐÃ CHẠY` |
| cặp `(0,0)/(1,1)/…/(6,6)` | 111 / 369 / 428 / 359 / 161 / 107 / 10 | `ĐỌC TỪ FILE` |
| result có index | 0/3541 (định dạng không có) → ghép theo vị trí | `ĐỌC TỪ FILE` |
| lệch số lượng / result trước call / call không tên | 0 / 0 / 0 | `ĐỌC TỪ FILE` |
| prompt bước *k* chứa observation bước `0..k-1` | được test khẳng định bằng marker | `ĐỌC TỪ FILE` `tests/test_trajectory.py` |

### C.4 Cắt chuỗi — số liệu adjudicated

Đo lại toàn bộ 4850 record bằng `build_supervised_example` + tokenizer thật (`ĐÃ CHẠY` phiên này, trên artifact đã rebuild):

| ngân sách | record | bị cắt giữa | % | bị bỏ | độ dài input tối đa |
|---|---|---|---|---|---|
| 4096 | 4850 | **3520** | **72,58%** | **122** (`target_value_truncated`) | 4096 |
| **8192 (hiện tại)** | 4850 | **47** | **0,97%** | 0 | 8192 |

Tách theo split ở 8192: train **47/4368 = 1,08%**, val **0/482** (`ĐÃ CHẠY`, khớp `stats.json`: `n_truncated: 47`, `n_truncated_fraction: 0.01076`). Cả 47 record bị cắt đều có `len(input_ids)` đúng bằng 8192. Record dài nhất cần **9016 token** (`train_0134#4`, `ĐÃ CHẠY` — prompt + target + eos), nên 8192 vẫn phải cắt nó — nhưng không mất phần ground ID.

**Mất schema của chính tool được gọi** (định nghĩa: chuỗi `### <tool> (` có mặt trong prompt đã decode, đếm trên mọi decision point có target `tool_call` không-meta):

> Ở `max_seq_length: 4096` (đo trên build **trước** khi bỏ 19 decision point — mẫu số lịch sử 3324), **921 trong 3324 prompt luyện `tool_call` (27,7%)** không chứa khối schema tham số của tool mà mẫu được yêu cầu gọi; ở 8192 còn **3/3324 (0,09%)** → sau fix collator (2026-11-08, ghim section `# Công cụ khả dụng` vào tail bằng `TOOLS_MARKER`): **0/3305**, số record bị cắt không đổi.

`ĐÃ CHẠY` phiên này, mẫu số lịch sử 3324 (build trước khi bỏ 19 point): 921/3324 và 3/3324 — trên build hiện tại mẫu số là **3305** và kết quả là **0/3305** sau fix collator — khớp số adjudicated trong `outputs/datasets/DATASET_PREPARATION_AUDIT.md` §14.5. (Biến thể mẫu số nếu muốn tính cả `ask_missing_info`: 921/3541 = 26,0% → 3/3541 = 0,08%.)

3 ca từng mất schema ở 8192 (trước khi ghim catalog) là **`train_0156#3`, `train_0204#3`, `train_0265#2`** — cả ba gọi `contacts.search` với khoá tham số `query`; chuỗi `query` xuất hiện **0 lần** trong prompt đã cắt của chúng, và ở 2 trong 3 ca vết cắt rơi **giữa một object JSON**. Sau khi ghim `# Công cụ khả dụng` vào tail: **0/3305**.

Bổ sung, cùng phép đo:

| hiện tượng | 4096 | 8192 |
|---|---|---|
| mất **toàn bộ** section `# Công cụ khả dụng` (target `tool_call`, train+val) — mẫu số lịch sử 3324 | **133/3324 = 4,00%** | 0 |
| mất section đó khi tính cả `ask` (train) | 128/3187 = 4,02% | 0 |
| mất giá trị tham chiếu lấy từ `# Lịch sử hội thoại` mà target cần | **7 giá trị / 6 record** | **0** (record đó nay bị bỏ hẳn, xem `target_value_truncated`) |
| 6 record từng bị ảnh hưởng | `train_1427` (`device_id=fan_office`), `train_1040`, `train_1372`×2, `train_1391`, `train_1422`, `train_1577` | — |
| target `tool_call` bị bỏ vì giá trị tham chiếu nằm trong phần bị cắt (`target_value_truncated`) | **122** | **0** |

Cơ chế (`ĐỌC TỪ FILE` `src/lifemate/training/collator.py:96,99,114,149,154`, `TOOLS_MARKER` + `_split_at_tools_section`, `_values_hidden_by_truncation`): collator giữ nguyên prefix `# Yêu cầu` (`REQUEST_MARKER`); sau đó **ghim cả section `# Công cụ khả dụng` và mọi thứ sau nó vào tail**, chỉ phần đứng *trước* catalog bị cắt giữa (head 35% / tail 65%, `head_keep_fraction=0.35`). Vì vậy `# Các bước đã thực hiện` (ground ID), `# Định dạng trả lời` và toàn bộ schema tool luôn sống sót.

**8192 được ghim ở 7 chỗ** để không thể âm thầm quay lại 4096: `configs/base.yaml:42`, `configs/training_lora.yaml:23`, `configs/training_full.yaml:26`, `src/lifemate/config.py:102`, `src/lifemate/training/trainer.py:58` (`DEFAULT_MAX_SEQ_LENGTH`), `src/lifemate/training/full_finetuning.py:169` (kế thừa), `src/lifemate/training/memory_preflight.py` (kế thừa `training.max_seq_length`) — `ĐỌC TỪ FILE`.

### C.5 Leakage

| kiểm tra | kết quả | nguồn |
|---|---|---|
| canonical JSON của `label.final_state` trong prompt | **0 / 4850** | `ĐỌC TỪ FILE` audit |
| chuỗi `final_state` trong bất kỳ message nào | **0 / 9700** message | `ĐÃ CHẠY` phiên này |
| container con của `final_state` (≥30 ký tự) | 0 | `ĐỌC TỪ FILE` |
| khoá chỉ có trong nhãn (`final_state`, `tool_calls`, `tool_results`, `gold`, …) | 0 | `ĐỌC TỪ FILE` |
| cặp `(prompt, target)` trùng | **0 / 4368** | `ĐÃ CHẠY` |
| cặp `(prompt, target)` trùng giữa train và val | **0** | `ĐÃ CHẠY` |
| id `public_` trong artifact train | 0 | `ĐỌC TỪ FILE` |
| train ∩ val | 0 | `ĐÃ CHẠY` |

`"label": {` xuất hiện 2753 lần trong prompt nhưng tất cả là **tham số** tên `label` trong dòng `parameters:` của schema (`alarm.create`, `alarm.update`), không phải khoá gold — `ĐỌC TỪ FILE`.

### C.6 Loss masking — kiểm chứng trên tokenizer thật (8192)

| record | `len(input_ids)` | token được supervise | index label đầu tiên | decode span == target + `<\|im_end\|>` | toàn bộ prompt = `-100` | có hàng all-`-100`? |
|---|---|---|---|---|---|---|
| `train_0134#4` | 8192 (bị cắt giữa) | 12 | 8180 | **True** | **True** | Không |
| `train_0276#0` | 5999 | 25 | 5974 | **True** | **True** | Không |
| `train_0696#1` | 5001 | 26 | 4975 | **True** | **True** | Không |
| `train_0540#3` | 4108 | 12 | 4096 | **True** | **True** | Không |
| `train_1036#0` | 1693 | 27 | 1666 | **True** | **True** | Không |

`ĐÃ CHẠY` phiên này qua `SupervisedCollator.from_tokenizer`. Bản cũ của bảng này (trong `DATASET_PREPARATION_AUDIT.md` §11) ghi `4096` cho cả 5 record — đó là số của build **trước** khi đổi ngân sách; bảng trên là build hiện tại.

Quét **toàn bộ** dataset, cùng phép kiểm (`ĐÃ CHẠY`):

| chỉ số | giá trị |
|---|---|
| record | 4850 |
| `kept` / bị bỏ | **4850 / 0** |
| hàng all-`-100` | **0** |
| decode span ≠ target + EOS | **0** |
| prompt bị supervise (phải = 0) | **0** |
| tail prompt đúng dạng think-block đã đóng | **4850 / 4850 (100%)** |
| tổng token được supervise (train) | **143.653** |
| tổng token được supervise (val) | **15.796** |
| tổng token được supervise (train+val) | **159.449** |

Padding (chỉ record ngắn cần) dùng `attention_mask=0` + label `-100` — `ĐỌC TỪ FILE` audit §11.

### C.7 Seam lịch sử hội thoại (history)

| split | mẫu có history | lượt user | lượt user có text trong dữ liệu gốc |
|---|---|---|---|
| `train` | 1018 | **3491** | **3491/3491** (`audio_text`) |
| `public_test` | **419** | **1299** | **0/1299** (chỉ có `audio`) |

`ĐÃ CHẠY` cả bốn ô. Trên `public_test`, **419/634 mẫu (66,1%)** sẽ vào agent với lượt trước rỗng nếu không có ASR — xem D-22 và mục F.

Renderer sau khi sửa (`ĐÃ CHẠY` phiên này, dựng prompt cho cả 1545 mẫu train bằng `build_decision_prompt` với `history_limit=10`):

| chỉ số | giá trị |
|---|---|
| dòng `Người dùng:` được render | **2782** (3491 lượt gốc, 709 bị `history_limit=10` cắt) |
| dòng `Người dùng:` rỗng | **0** |
| dòng `kết quả:` | 2815 |
| dòng `kết quả:` có `"result": null` | **0** |

---

## D. Audit source code

### D.1 Defect đã sửa (bảng chính)

Mức: `P0` = chặn nộp bài / làm hỏng dữ liệu huấn luyện; `P1` = sai hành vi đo được; `P2` = sai số liệu/khó phát hiện.

| ID | mức | file:line | nguyên nhân gốc | bằng chứng đo | cách sửa | test |
|---|---|---|---|---|---|---|
| D-01 | P0 | `src/lifemate/env/official.py:35,44` | `import zaic_tools` trơn; hai split phát hai bản `zaic_tools` khác nhau (147 vs 122 tool) | trước: `ModuleNotFoundError` → `LifeMateError: zaic_tools not installed` → **exit 2 trên bộ dữ liệu đầy đủ**; sau: import xen kẽ hai split trong **cùng process** đúng 147/122 | `_import_split_sdk`: đưa `<data_root>/<split>` lên đầu `sys.path` (xoá lần xuất hiện cũ), `invalidate_caches`, **kiểm chứng `module.__file__`**, lệch thì xoá `sys.modules["zaic_tools*"]` và thử lại 1 lần rồi báo lỗi | `test_official_env_imports_the_requested_splits_copy` |
| D-02 | P0 | `src/lifemate/data/schemas.py:260`, `src/lifemate/agent/prompt_builder.py:163` | `_turn_text` chỉ đọc `content`; lượt user thật là `{audio, audio_text}` | trước: `Người dùng: ` trắng cho **3491/3491** lượt train (không lượt nào có khoá `content`) | đọc `content` → `utterance` → `audio_text` → `text`, cả bản đã parse lẫn bản `raw`; không bịa transcript | `test_history_user_turn_with_audio_text_is_rendered`, `test_history_turn_with_only_audio_renders_nothing_extra` |
| D-03 | P0 | `src/lifemate/data/schemas.py:159,199` | `ToolResult.to_dict()` trả `{name, ok, result}`; payload thật nằm ở các khoá khác (`alarms`, `item`, `devices`…) | **3501/3541** gold result mang payload ngoài khoá điều khiển (chỉ 40 là `{"ok": true}` thuần) → SFT prompt hiện `{"ok": true, "result": null}`; sau sửa: 2815/2815 dòng `kết quả:` của 1545 prompt đều có payload, 0 `result: null` | `ToolResult.payload` giữ mọi khoá ngoài control; `to_dict()` dựng lại đúng shape `env.call()`; renderer đọc cả bản `raw` khi bản đã parse mất payload | `test_history_tool_result_payload_is_rendered_from_the_raw_record`, `test_history_tool_calls_and_results_render_one_pair_per_call` |
| D-04 | P0 | `src/lifemate/data/catalog.py:309` | `x-one-of` là **danh sách nhóm** (`[["track_id"],["playlist_id"]]`); code cũ hiểu như danh sách giá trị | `ĐỌC TỪ FILE` audit §2: trong 487 bất đồng parity, **18 ca** là "nhóm `x-one-of` lồng nhau hiểu sai" (vd. `music.play {}` — BTC từ chối, local chấp nhận); `ĐÃ CHẠY` phiên này: **54/54** ca `x-one-of` trên 6 tool thật đồng ý, 0 bất đồng. Con số "217 bất đồng → 0" nêu trong brief: `CHƯA ĐO` — không tài liệu nào trong repo ghi 217 cho `x-one-of` | một **nhóm** phải đầy đủ; giá trị rỗng (`None/""/[]/{}`) tính là vắng; nhóm phẳng `["a","b"]` = một nhóm | `test_one_of_nested_groups_match_official_rule` |
| D-05 | P1 | `src/lifemate/env/official.py:113,125-127` | SDK trả `{ok, error}` và **không có** danh sách `errors` | `ĐÃ CHẠY` trực tiếp: `env.validate('alarm.create', {'label':'x'})` → `{'ok': False, 'error': 'thiếu tham số bắt buộc: time'}`; trước sửa `ok=False` đi kèm giải thích rỗng | đọc thêm `raw["error"]` và nhét vào danh sách lỗi | `test_env_validate_false_with_no_error_list_is_still_a_rejection` |
| D-06 | P1 | `src/lifemate/agent/tool_parser.py:62`, `src/lifemate/agent/tool_executor.py:77` | validator local lệch validator chính thức | `ĐỌC TỪ FILE` audit §2: trước **487/2156** ca bất đồng (22,6%), sau khi chuyển sang `env.validate`: **272/272** ca vàng + biến thể đồng ý. `ĐÃ CHẠY` phiên này trên catalog thật (`/tmp/parity_final.py`, 1545 mẫu / 3541 gold call + biến thể: gold, bỏ từng `required`, rỗng hoá `""`/`None`/`[]`/`{}`, số 0/âm/bool/không hữu hạn, bằng đúng bound, `format` sai/đúng, `items` sai kiểu): **27.070 ca → 0 bất đồng** (0 ca local từ chối mà BTC chấp nhận, 0 ca ngược lại); hai ca từng lệch `fitness.set_goal{steps:0}` và `home_energy.set_alert{threshold_kwh:0}` nay cả hai phía cùng từ chối (`phải > 0`), catalog đã áp `exclusiveMinimum` (D-23) | khi env có `validate` thì env là chuẩn; catalog chỉ còn là fallback (test thuần, mock) | `test_env_validator_is_authoritative_over_catalog` |
| D-06b | P1 | `src/lifemate/data/catalog.py` (`_check_value`) | validator local từng thiếu `minimum`/`exclusiveMinimum`, `format`, `items.*` | **ĐÃ SỬA — xem D-23** (hàng riêng để giữ nguyên ID đã tham chiếu trong các tài liệu khác) | — | `test_catalog_enforces_bounds_formats_and_array_items` |
| D-07 | P0 | `scripts/infer.py:581-583`, `scripts/evaluate.py:239-241` | preflight gọi `create_environment` chỉ để "kiểm tra sớm" | instrumentation: `max env instances per sample: 2`, mẫu `public_0002` tạo **2 lần** — vi phạm luật "mỗi mẫu một ToolEnv, tạo một lần"; sau sửa: `40 samples, max 1, repeated []` | thêm `check_official_sdk(split, data_root)` (`official.py:35`): chỉ import + kiểm nguồn gốc, **không tạo session** | `test_official_preflight_creates_no_session`, `test_evaluate_closed_loop_builds_one_env_per_sample` |
| D-08 | P1 | `src/lifemate/agent/prompt_builder.py:47` | spec định dạng in ví dụ cứng `alarm.list` cho **mọi** mẫu | trước: mẫu không cài `alarm` vẫn thấy `alarm.list`; sau: **0/1052** mẫu không cài `alarm` chứa chuỗi này, prompt có `<tên_công_cụ>` | ví dụ dùng placeholder + câu "tên công cụ luôn lấy từ danh sách Công cụ khả dụng" | `test_prompt_mentions_no_tool_of_an_uninstalled_app` |
| D-09 | P0 | `src/lifemate/models/loader.py:335,385` | `from_pretrained` chỉ **in** bảng MISSING rồi giữ khởi tạo ngẫu nhiên | repo test dựng config 3 layer trên trọng số 2 layer: layer 2 khởi tạo std 0,0198 và **run vẫn chạy**; sau sửa raise `... incomplete: N parameter(s) have no weights` (con số 14 tham số trong brief: `CHƯA ĐO` — phép đo của repo không phải 14) | `_missing_parameters()` đọc `missing_keys` (bỏ khoá tied), raise `LifeMateError` kèm tên tham số đầu tiên | `test_incomplete_checkpoint_raises_instead_of_random_init` |
| D-10 | P0 | `src/lifemate/training/full_finetuning.py:55` | full FT ở bf16 master: AdamW ghi update thẳng vào tensor tham số, `lr=5e-6` bị lượng tử hoá thành 0 | đo: sau 20 bước chỉ **7,3–7,4%** phần tử bf16 dịch chuyển, **100%** với fp32 master (`ĐỌC TỪ FILE`; hai tài liệu ghi 7,3% và 7,4% cho cùng phép đo) | `master_dtype()` ép fp32 cho tham số trainable; bf16 chỉ dùng cho `torch.autocast` | `test_full_finetuning_keeps_fp32_master_weights` |
| D-11 | P0 | `docker/Dockerfile:122-126`, `cloud/sandbox_runner.py:171-181` | image không khai `ENTRYPOINT`; job cloud trỏ `--dataset-dir` vào mount **không có** `train.jsonl` | `ENTRYPOINT ["/app/docker/entrypoint.sh"]` hiện có; gate từ chối `--dataset-dir` không nằm dưới `PREPARED_DATASETS_CONTAINER` (đo được: `--dataset-dir ./data` → "Dataset not found", `--dataset-dir outputs/datasets/train` → 29 record) | thêm `ENTRYPOINT`; thêm kiểm tra tĩnh coverage mount, dry-run fail + submit bị chặn | `test_training_command_points_at_prepared_dataset_dir`, `test_spec_has_no_uncovered_container_paths` |
| D-12 | P0 | `src/lifemate/data/trajectory_builder.py:237,264` | guard chống leak so canonical `final_state`; `{}` canonical hoá thành `"{}"`, có mặt trong **mọi** prompt (schema `"properties": {}`) | `ĐÃ CHẠY` hai phép đo: (i) dựng lại guard **trước sửa** → `LifeMateError: label leak: canonical JSON of label.final_state appears in the prompt`, build dừng, **0 decision point**; (ii) cả **1545/1545** prompt dựng từ `train` đều chứa literal `{}`, nên guard cũ fail ngay mẫu đầu tiên có `final_state == {}`; guard hiện tại early-return cho `final_state` rỗng → build thật ra **4850** record (4368+482), 0 lỗi | `if final_state is None or not final_state: return` — `{}` không leak gì | `test_label_final_state_never_in_prompt`, `test_assert_no_label_leak_catches_injection` |
| D-12b | — | — | số mẫu bị ảnh hưởng | đo độc lập: **373** mẫu có `final_state == {}` (332 trong split train, **543 decision point**; 262 mẫu có ≥1 call, 111 zero-call). Con số **267** trong brief: `CHƯA ĐO` — không tái tạo được định nghĩa nào cho ra 267 từ dữ liệu | — | — |
| D-13 | P2 | `src/lifemate/data/catalog.py:66` | bỏ qua `kind` do catalog khai, suy ra từ động từ cuối tên tool | tái lập **đúng** logic repo trên catalog thật: **11/147** tool bị phân loại sai (`calculator.calculate`, `tv.guide`, `package_tracking.track`, `news.headlines`, `phone.recent_calls`, `exchange_rate.convert`, `calculator.convert_unit`, `food_delivery.order_history`, `flights.my_bookings`, `flights.check_in`, `shopping_list.check`) | `kind` khai trong catalog thắng; heuristic chỉ dùng khi catalog không khai | `test_declared_kind_beats_verb_inference` |
| D-14 | P2 | `src/lifemate/eda/dataset_analysis.py:190` | HF tokenizer vừa `callable` vừa có `.encode`; `len(BatchEncoding)` = **2** (số field) | trước: mọi mẫu báo 2 token | ưu tiên `.encode`; chỉ dùng nhánh callable khi không có `.encode` | `test_hf_style_tokenizer_uses_encode_not_call` |
| D-15 | P0 | `configs/*.yaml`, `src/lifemate/config.py:102`, `trainer.py:58`; ghim tail: `src/lifemate/training/collator.py` (`TOOLS_MARKER`, `_split_at_tools_section`) | `max_seq_length: 4096` cắt 75% record, có record mất cả section schema tool | 3520/4850 (72,58%) → 47/4368 (1,08%) ở 8192; schema tool mất 921/3324 (27,7%) → 3/3324 (0,09%); sau khi ghim `# Công cụ khả dụng` vào tail: **0/3305** (`ĐÃ CHẠY`, mục C.4). 3 ca từng hỏng là `train_0156#3`, `train_0204#3`, `train_0265#2` (đều `contacts.search`, khoá `query` xuất hiện 0 lần, 2/3 vết cắt rơi giữa JSON) | 8192 ở cả 7 điểm ghim + catalog ghim vào tail | `test_shipped_configs_agree_on_max_seq_length`, `test_collator_truncation_keeps_the_request_section` |
| D-16 | P1 | `scripts/prepare_dataset.py:81` | `_fit_stats` chỉ đếm `kept`, nên record bị cắt giữa vẫn được tính là "fit" | `manifest` cũ khai `n_fit` **bằng đúng số record** và `n_over_budget: 0` trong khi 3520 record bị cắt; hiện tại: `n_truncated: 47`, `n_truncated_fraction: 0.01076`, `n_dropped: 0`, `dropped_reasons: {}`, `method: tokenizer`, `caveat: null` | tách hai kết cục `n_truncated` / `n_dropped`; không có tokenizer thì ghi `chars4_estimate` + `caveat` | `test_prepare_dataset_fit_reports_truncation_not_a_false_fit` |
| D-26 | P1 | `src/lifemate/training/collator.py:231-246` (`_values_hidden_by_truncation`), `:355-368` (guard), `scripts/prepare_dataset.py:92` | sau khi cắt giữa prompt, một `tool_call` có thể supervise một ID/tên **đã bị chính vết cắt xoá**; guard cũ chỉ kiểm prompt **trước** khi cắt | trước: 122 record như vậy vẫn được supervise ở 4096. Ở 8192: **0**; ở 4096: **122** bị bỏ với `reason="target_value_truncated"` (`ĐÃ CHẠY`, `ĐÃ CHẠY` một lần chạy 4096) | guard hỏi lại đúng câu hỏi trên **text cuối cùng** sau khi cắt, bỏ theo tên thay vì supervise một target không thể suy ra | `test_prepare_dataset_fit_reports_truncation_not_a_false_fit`, `tests/test_training.py` (assert `target_value_truncated` ∈ lý do bỏ) |
| D-17 | P1 | `src/lifemate/asr/__init__.py:97`, `src/lifemate/config.py:124` | nhánh fallback giải mã `greedy_search` thay vì cấu hình đã công bố | hiện tại: `AsrConfig.decoding_method = "modified_beam_search"`, factory cũng vậy; `ASR_DECODINGS` chặn giá trị lạ | đổi default ở cả ba nơi | `test_default_decoding_method_is_modified_beam_search` |
| D-18 | P1 | `src/lifemate/asr/metrics.py:211,220,263` | regex magnitude/money thiếu cách viết miền Nam `ngàn` | magnitude errors: **17 → 4** trên cùng 1545 câu (artifact `/tmp/asr_train_pred` vs `/tmp/asr_final`); số ca lỗi còn lại thuộc lớp `numbers` 4 và `money` 4 | thêm `ngàn`/`nghàn` vào mọi pattern scale | `test_magnitude_diagnostics_recognise_ngan_spelling` |
| D-19 | P1 | `src/lifemate/asr/metrics.py:72,111` | view "raw" so phân biệt hoa/thường với recogniser in HOA toàn bộ | raw CER **0,7848 → 0,1495** sau khi casefold (`/tmp/ts.log` vs `/tmp/ts2.log`); normalized CER 0,1113 không đổi | `raw` = `str.lower()` hai phía rồi mới tính; `for_wer` vẫn là headline | `test_raw_metric_view_is_case_insensitive` |
| D-20 | P1 | `src/lifemate/asr/gipformer_backend.py:54,351` | Gipformer xuất ALL-UPPERCASE, transcript train là sentence-case | `ĐỌC TỪ FILE` audit §15.4: per-char case agreement của phần request **0,0556 → 0,9444** (pooled **0,0494 → 0,9506**); 213/740 (28,8%) gold argument fuzzy lệch hoa/thường so với câu nói. `ĐÃ CHẠY` phiên này, định nghĩa khác (mọi chữ cái của 1545 câu `train`, so `reference` vs `raw_hypothesis` rồi vs `hypothesis`): **0,2146 → 0,9650** trên 106.813 chữ cái — cùng xu hướng, khác mẫu số nên khác trị tuyệt đối. 1545/1545 `raw_hypothesis` là all-uppercase, 1545/1545 `hypothesis` là lowercase. Scorer chính thức **không phân biệt hoa thường** (`_norm`/`_fold`/`_canon_scalar`, `data/train/zaic_tools/utils.py:10-45`), nên thiệt hại bị chặn trên | `lower_if_uppercase_only`: chỉ lowercase khi text **không mang** thông tin hoa/thường | `test_uppercase_only_recognizer_text_is_lowercased_for_the_agent` |
| D-21 | P2 | `src/lifemate/training/memory_preflight.py:46,99` | mặc định 36 layer trong khi checkpoint phát hành có **32** | config thật: `num_hidden_layers = 32` (`ĐÃ CHẠY` đọc `config.json`); `DEFAULT_LAYERS = 32` | sửa default; vẫn ưu tiên `model.config.num_hidden_layers` | `test_memory_preflight_math_is_exact` |
| D-22 | P0 | `scripts/infer.py:124,534-543`, `src/lifemate/data/history.py:39,75,172` | audio lịch sử của `public_test` không có consumer ở suy luận | `outputs/asr/history_transcripts.jsonl`: **4790 dòng** (3491 train + 1299 public_test), **0** `error`, **0** text rỗng, **0** khoá trùng; điền **1299/1299** lượt trên **419/419** mẫu; trên train là no-op (gold luôn thắng) | thêm artifact + `apply_history_texts` + cờ `--history-transcripts`; mọi dòng phải qua `assert_path_allowed` và nằm dưới `<data_root>/<split>/audio/`, vi phạm thì **raise** | `test_history_texts_fill_a_blank_user_turn`, `test_history_texts_never_overwrite_gold_audio_text`, `test_history_texts_audio_outside_the_split_audio_dir_is_refused`; suy luận **từ chối chạy** (exit 2) khi còn lượt lịch sử trống — chỉ `--allow-missing-history` mới bỏ qua, `test_blank_history_fails_unless_explicitly_allowed`, `test_uncovered_history_turns_from_a_partial_file_are_refused` |
| D-23 | P1 | `src/lifemate/data/catalog.py:197` (`_check_value`), `:277` (`_format_ok`), `:308` (bounds) | validator catalog local thiếu 4 họ luật mà scorer chính thức áp: `minimum`/`exclusiveMinimum` (và maximum), `format` (`date`/`date-time`/`time`), `items.type`/`items.enum` — nên **bất kỳ test/CLI nào gọi catalog trực tiếp** đều chấp nhận call mà BTC từ chối (trước đây là O-01/D-06b) | `ĐÃ CHẠY` phiên này: **12.770 ca → 0 bất đồng** (`/tmp/parity.py` trên toàn bộ gold call + biến thể); harness của lượt sửa ghi thêm **10.940 ca gold-only + biến thể → 0** và **207 probe nhắm từng bound/format/items → 0** (`ĐỌC TỪ FILE` closure note; chưa chạy lại hai tập con này); trước sửa đo được **840/14.671 (5,73%)**; `prepare_dataset.py` ra **4368 + 482**, drops mẫu **0** / drops decision point **17 + 2** | port 4 luật từ `data/train/zaic_tools/validate.py`: regex + kiểm lịch thật cho `date` (cấm `datetime.fromisoformat`), range-check `hh:mm`, `math.isfinite`, đệ quy `items` | `tests/test_data.py::test_catalog_enforces_bounds_formats_and_array_items` (fail khi revert fix) |
| D-24 | P0 | `src/lifemate/training/collator.py:49,54,79`, `src/lifemate/models/generation.py:133` | hai call site dựng prompt gọi `apply_chat_template(..., add_generation_prompt=True)` **không** có `enable_thinking`, nên template Qwen3.5 rơi vào nhánh `else` và in `'<\|im_start\|>assistant\n thinking\n'` — think block mở mà không đóng | `ĐÃ CHẠY` tokenizer thật `Qwen/Qwen3.5-4B`, cùng 2 message: trước `tail = '\n thinking\n'` (`open 1 / close 0`); sau khi sửa `tail = '\n\n'` = closed form (`open 1 / close 1`), và prompt của record thật `outputs/datasets/train/train.jsonl` #0 cũng kết thúc đúng closed form; `render_prompt_text == HFGenerator.build_prompt` → **True** (byte khớp train/infer) | thêm `apply_chat_template_text(tokenizer, messages, *, add_generation_prompt=True)` (`collator.py:54`) truyền `enable_thinking=False`; chỉ `TypeError` có chuỗi `enable_thinking` mới retry không kwarg (tokenizer không phải Qwen); **mọi lỗi template khác giờ `raise LifeMateError` ở cả hai renderer** (trước đây `render_prompt_text` nuốt lỗi và rơi về ChatML tự chế trong khi `HFGenerator` raise → byte prompt train/infer có thể lệch); `NO_THINK_SUFFIX` (`:49`) làm fallback ChatML của tokenizer trần in cùng suffix closed-think; `build_prompt` (`generation.py:133`) dùng chung helper | `tests/test_training.py::test_rendered_prompt_closes_the_think_block` |
| D-25 | P0 | `src/lifemate/data/discovery.py:100-102` | `has_environment_dir` chỉ kiểm `root.parent / f"{name}_environment"`, tức `<data_root>/train_environment`, nhưng layout chính thức lồng ở `<data_root>/<split>/<split>_environment` | trước: `outputs/data_inspection.md` in `environment dir present (never opened): False` cho **cả hai** split thật; nay `outputs/data_inspection.md:16,31` in **True** cho `public_test` và `train` | kiểm `root / f"{name}_environment"` trước, fallback `root.parent`; chỉ là `Path.is_dir()`, **không** đọc/`open` gì nên tuyên bố tuân thủ ở §0 không đổi | `tests/test_data.py::test_discovery_finds_nested_environment_dir` |

### D.2 Còn tồn tại (đã biết, chưa sửa)

| ID | mức | file:line | nội dung | số đo | tác động |
|---|---|---|---|---|---|
| O-02 | P1 | dữ liệu | 11/36 app của `public_test` không có mẫu train | 11 app | tool selection zero-shot trên ~30% app test; không sửa được bằng code |
| O-03 | P1 | dữ liệu | 0 gold result lỗi | 0/3541 | không học được phục hồi lỗi; cần augment nếu muốn |
| O-04 | P2 | dữ liệu | `contacts.get` chưa từng được gọi | 1/147 tool | tool được liệt kê nhưng không có supervision |
| O-05 | P2 | dữ liệu | history `public_test` chỉ có audio | 1299 lượt | đã lấp bằng ASR (WER 18,68%); train dùng gold → lệch phân bố có đo (13/34 argument free-text còn ground được khi thay ASR, 21/34 mất) |
| O-06 | P2 | `src/lifemate/evaluation/evaluator.py` | không có scorer final-state chính thức | `available: False` ở mọi artifact | 1 phần điểm thi không tự đo được |
| O-07 | P2 | `docs/` | vài `file:line` trong tài liệu cũ đã trôi (vd. preflight ghi `infer.py:538`, hiện là `581-583`) | — | chỉ ảnh hưởng truy vết, không ảnh hưởng hành vi |

**Ghi chú:** mục `O-01` (validator catalog thiếu bound/format/items) **đã đóng và chuyển thành defect D-23** trong D.1 — 12.770 ca parity → 0 bất đồng. Các ID `O-02..O-07` giữ nguyên số, không đánh lại.

### D.3 Nghi vấn đã bác bỏ

| nghi vấn | kết luận | bằng chứng |
|---|---|---|
| `AutoModelForCausalLM` sai cho model hybrid | **sai** — `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES['qwen3_5'] → Qwen3_5ForCausalLM`; dựng meta-device ra đúng class | `ĐÃ CHẠY` phiên này: `class Qwen3_5ForCausalLM` |
| checkpoint multimodal làm LM khởi tạo ngẫu nhiên | **sai** — conversion `^model\.language_model\.(.+)` → `model.\1`; load qua `_load_state_dict_into` trả missing/unexpected rỗng | `ĐỌC TỪ FILE` audit + `ĐÃ CHẠY` index 426 tensor `model.language_model.*` |
| teacher-forced và closed-loop lẫn nhau | **sai** — hai hàm/hai khoá metric riêng; không tồn tại metric perplexity trong repo | `ĐỌC TỪ FILE` |
| `music.add_to_playlist` có `x-one-of` nên "bỏ mẫu vàng hợp lệ" | **sai** — chỉ `music.play` có `x-one-of`; 6 tool có extension, `add_to_playlist` không nằm trong đó | `ĐÃ CHẠY` đọc catalog |
| defect `x-one-of` là "quá chặt với nhóm hợp lệ" | **sai một nửa** — lỗi thật là **quá chặt với giá trị rỗng** và **quá lỏng với sự hiện diện nhiều nhóm** | `ĐÃ CHẠY` 54 ca; `ĐỌC TỪ FILE` audit §2 |
| gold result lỗi `ok:false` tồn tại nên có mất supervision | **sai** — 0/3541 | `ĐÃ CHẠY` |
| collator có thể sinh hàng toàn `-100` → NaN loss | **sai** — 0/4850 hàng all-`-100` | `ĐÃ CHẠY` quét toàn bộ |
| render history là "không thể" với lượt audio | **đúng một phần**: render an toàn nhưng mất thông tin; nay đã lấp bằng ASR cho `public_test` | `ĐÃ CHẠY` mục C.7 |

### D.4 Phụ thuộc ngoài chưa xác minh

| hạng mục | trạng thái | thiếu gì |
|---|---|---|
| 1 bước huấn luyện thật trên Qwen3.5-4B (loss, grad norm, peak VRAM) | **NOT RUN** | không có byte trọng số; policy không cho tải |
| chạy trên GPU | **NOT RUN** | torch `2.15.0+cpu` (phần cứng có) |
| build/run Docker | **NOT RUN / UNVERIFIED** | **hai** lý do: (1) không có daemon trên máy này; (2) image theo pin cũ (`nvidia/cuda:12.4.1` + `whl/cu124`) **không thể** chạy GPU trên card `sm_120` này ngay cả khi có daemon — `cu124` không có kernel `sm_120`. Pin đã được sửa trong phiên này sang `nvidia/cuda:12.8.1` + `whl/cu128` (`docker/Dockerfile:3-12,31,51-53,73`), nhưng bản sửa là **UNVERIFIED: image chưa từng được build, chưa từng được chạy** |
| submit cloud / W&B | **NOT RUN** | `wandb` chưa cài, chưa có credential |
| scorer final-state của BTC | **không tồn tại trong gói phát hành** | BTC chỉ phát `zaic_tools` runtime |
| điểm trên `public_test` | **không thể có** | 634/634 mẫu không nhãn |

---

## E. Huấn luyện

### E.1 Kiến trúc và tham số (không cần trọng số)

| hạng mục | giá trị | nguồn |
|---|---|---|
| class khi load | `Qwen3_5ForCausalLM` | `ĐÃ CHẠY` — dựng meta-device từ `config.json` phát hành |
| **tổng tham số text** | **4.205.751.296** (426 tensor, không `lm_head`) | `ĐÃ CHẠY` phiên này |
| `state_dict()` | 4.841.450.496 (427 tensor; thêm `lm_head.weight`) | `ĐÃ CHẠY` |
| chênh lệch | 635.699.200 = `vocab 248320 × hidden 2560` — chính là tensor tied bị đếm hai lần | `ĐÃ CHẠY` |
| layer | **32** (24 `linear_attention` + 8 `full_attention`), hidden 2560, head_dim 256, vocab 248320, `tie_word_embeddings: true` | `ĐÃ CHẠY` đọc `config.json` |
| load | 0 missing / 0 unexpected qua conversion | `ĐỌC TỪ FILE` audit (không có byte trọng số ở đây) |

### E.2 Full FT là default

`configs/base.yaml` + `configs/training_full.yaml` đặt `tuning_mode: full`; `configure_full_finetuning` unfreeze toàn bộ và **raise** nếu `trainable != total`; không có fallback im lặng sang LoRA (`ĐỌC TỪ FILE` `src/lifemate/training/full_finetuning.py`, `docs/TRAINING_GUIDE.md` §2).

### E.3 LoRA — target module đã kiểm trên kiến trúc phát hành

Giao của 17 suffix ứng viên (`ATTENTION_MLP_SUFFIXES` + `HYBRID_SUFFIXES`, `lora_finetuning.py:29-42`) với 426 tensor `model.language_model.*` của checkpoint thật:

| nhóm | suffix tìm thấy |
|---|---|
| attention/MLP | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7) |
| hybrid (linear attention) | `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` (5) |
| **tổng** | **12** |

`ĐÃ CHẠY` phiên này trên `model.safetensors.index.json`. `r=16`, `alpha=32`, `dropout=0.05`, trainable **32.464.896** tham số = **0,766%** của base **4.205.751.296** (`ĐÃ CHẠY`: `configure_lora` trên model meta-device kiến trúc thật `Qwen3_5ForCausalLM(text_config)`, 12 suffix trên 248 module, `trainable_source=measured`; tổng giải tích khớp `peft` từng module).

> Con số cũ **21.028.756** không phải phép đo: nó là `int(4.205.751.296 × 0,005)` — placeholder của preflight. Sau fix, preflight gắn nhãn `trainable_source=placeholder_0.5pct` (`src/lifemate/training/memory_preflight.py:159-168`) và chỉ dùng nó khi **không** có model instance; `scripts/train.py --dry-run` trên máy không có trọng số vì vậy vẫn in placeholder (20.000.000 / 10,88 GB), còn đường đo thật cho **32.464.896 / 11,49 GB**. Số đúng để trích là **32.464.896**.

### E.4 Cấu hình thực tế

| tham số | full | LoRA | nguồn |
|---|---|---|---|
| `learning_rate` | 5e-6 | 2e-4 | `ĐỌC TỪ FILE` |
| `per_device_train_batch_size` | 1 | 1 | `ĐỌC TỪ FILE` |
| `gradient_accumulation_steps` | 8 | 8 | `ĐỌC TỪ FILE` |
| **effective batch** | **8** | **8** | `ĐỌC TỪ FILE` |
| `max_seq_length` | **8192** | **8192** | `ĐỌC TỪ FILE` |
| `gradient_checkpointing` | **true** | **true** | `ĐỌC TỪ FILE` |
| epoch | 2 | 3 | `ĐỌC TỪ FILE` |
| precision | bf16 compute + **fp32 master** | bf16 base + adapter | `ĐỌC TỪ FILE` |
| optimizer / clip / seed | `adamw_torch` / 1.0 / 42 | same | `ĐỌC TỪ FILE` |

### E.5 Ước lượng bộ nhớ (công thức của repo, **không phải** phép đo cấp phát)

`train.py --dry-run --param-count 4205751296` (`ĐÃ CHẠY` phiên này):

| hạng mục | full | LoRA |
|---|---|---|
| weights | 16,82 GB | 8,48 GB (bf16, đóng băng) |
| gradients | 16,82 GB | 0,06 GB |
| optimizer state | 33,65 GB | 0,26 GB |
| activations (k=2, có checkpointing) | 2,68 GB | 2,68 GB |
| **TOTAL** | **69,98 GB** | **11,49 GB** |
| checkpoint | 33,65 GB | 8,54 GB |
| verdict trên card 17,09 GB / 15,45 GB trống | **does_not_fit** | **fits** |

- Card: 16303 MiB = **17,09 GB** decimal (15,92 GiB) `ĐÃ CHẠY` `nvidia-smi`.
- Trống khi idle (đo trong audit trước): 14.733 MiB = **15,45 GB**; **tại thời điểm audit này**: 11.276 MiB = **11,82 GB** (máy đang bận) `ĐÃ CHẠY` → con số headroom phụ thuộc thời điểm, phải đo lại trước khi chạy thật.
- 11,49 / 15,45 = **74,4%** VRAM trống, headroom **3,96 GB** (`SUY RA`; 15,45 − 11,49 = 3,96). Con số LoRA ở đây là đường **đo** (`run_memory_preflight(model=…)` với adapter `measured`), không phải placeholder; dòng LoRA mà `train.py --dry-run` in ra trên máy này (10,88 GB) là placeholder 0,5% — xem E.3.
- `gradient_checkpointing` **được mô hình hoá** (k=2 vs 10); tắt nó thì LoRA 8192 ≈ 35,47 GB, full ≈ 80,71 GB — đều không vừa. **Checkpointing là load-bearing, không được tắt.**
- Mọi số trên là **ước lượng công thức**, `verdict: unknown` khi không thấy CUDA. Chưa có `torch.cuda.max_memory_allocated()` nào.

### E.6 Checkpoint và resume

| hạng mục | trạng thái | bằng chứng |
|---|---|---|
| checkpoint hiện có | `outputs/checkpoints/full/step-2` + `crash-2`, `model.safetensors` 98,7 KB | `ĐÃ CHẠY` `ls` — đây là model tiny CPU của smoke test, **không** phải Qwen |
| nội dung step dir | `model`, `tokenizer`, `training_state.json`, `optimizer_state.pt`, `lifemate_meta.json` | `ĐÃ CHẠY` |
| resume kiểm `step`/`hash`/`tuning mode` | có, raise nếu lệch | `ĐỌC TỪ FILE` `trainer.py:704-830` |
| **checkpoint lưu ở 4096 từ chối resume ở 8192** | **đúng hành vi mong muốn**: `max_seq_length` nằm trong `config_fingerprint` → `cannot resume: training config differs` (fingerprint `97ce16b36757e736`@4096 vs `8f57e80e923e1cc3`@8192) | `ĐỌC TỪ FILE` `tests/test_training.py:2458` |
| resume khôi phục optimizer+AMP+RNG | có, test riêng | `test_resume_restores_optimizer_and_rng_state` |
| resume LR | đã sửa: schedule dựng lại theo horizon thật (`last_epoch=start_step`), trước đây chạy LR 0.0 | `test_resumed_run_keeps_a_real_learning_rate` |
| ghi đè step dir | không bao giờ; resume/rerun vào dir đã có → raise | `ĐỌC TỪ FILE` |

---

## F. ASR (Gipformer-68M-RNNT)

### F.1 Định danh và asset

| hạng mục | giá trị |
|---|---|
| repo | `g-group-ai-lab/gipformer-68M-rnnt` |
| revision | `318322b009200f8e77a3751fa64638e5cc85a7dc` |
| kiến trúc / runtime | `ZipformerRnntModel`, `sherpa-onnx` offline transducer, int8 |
| file | 6: `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`, `bpe.model`, `config.json` |
| dung lượng | **73.515.623 byte** cho 6 file (= **70,11 MiB**, khớp dòng "Downloading 6 files (70.11 MB)" của script). Tập 4 file mà metrics JSON tự khai (`encoder/decoder/joiner.int8.onnx` + `tokens.txt`) là 73.244.313 byte (69,85 MiB); `bpe.model` 270.695 B và `config.json` 615 B nằm ngoài con số đó |
| xác minh | `Verification OK: 6 file(s) hash-verified` — per-file `sha256` (LFS) / `git-sha1` (blob thường) lấy từ tree API ở revision pin, **không hardcode** |
| asset thiếu | **không** — 0/6 thiếu; biến thể fp32 không tải (không cần) |
| version 1.5 | bị từ chối tường minh (`gipformer1.5-68M-rnnt` là model khác) |

`ĐÃ CHẠY` `/tmp/dl.log` + `ĐỌC TỪ FILE` `docs/GIPFORMER_INTEGRATION_AUDIT.md` §4. Log hash-verify thật, nguyên văn 6 dòng `OK`:

```
  OK      tokens.txt (git-sha1 8adf5f75e3baf10c…)
  OK      bpe.model (sha256 289dbb44527c13c4…)
  OK      config.json (git-sha1 2c9d9f782f997c45…)
  OK      encoder.int8.onnx (sha256 3cc1a719f04d6051…)
  OK      decoder.int8.onnx (sha256 2b6235b8a8be57b5…)
  OK      joiner.int8.onnx (sha256 eb74e2ead853d641…)
```

### F.2 Chất lượng đo thật — toàn bộ 1545 utterance của train

| chỉ số | giá trị | nguồn |
|---|---|---|
| n | **1545** | ĐÃ CHẠY |
| **WER chuẩn hoá (`for_wer`)** | **17,91%** (`0,17909`) | ĐÃ CHẠY |
| **CER chuẩn hoá** | **11,13%** (`0,11128`) | ĐÃ CHẠY |
| sentence exact match | **8,74%** (`0,08738`) | ĐÃ CHẠY |
| WER/CER "raw" (case-insensitive, không chuẩn hoá số) | 29,80% / 14,95% | ĐÃ CHẠY |
| latency mean / p50 / p95 / max | **82 ms** (0,0822) / 76 ms / 160 ms / 553 ms | ĐÃ CHẠY |
| **RTF** | **0,0114** (0,01137 = Σlatency / Σaudio) | ĐÃ CHẠY |
| wall time | 128,1 s cho 3,102 h audio | ĐÃ CHẠY |
| lỗi | **0** | ĐÃ CHẠY |
| empty hypothesis | 0 | ĐÃ CHẠY |

Lần chạy lặp lại cho **text giống hệt** (0 dòng khác) đo latency 87 ms, RTF 0,0121, wall 136,3 s (`ĐÃ CHẠY`). Nhiễu máy, không phải khác biệt thuật toán.

### F.3 Lượt lịch sử hội thoại (3491 lượt)

| chỉ số | giá trị |
|---|---|
| WER chuẩn hoá | **18,68%** (`0,18682069`) |
| CER chuẩn hoá | **11,05%** (`0,11047727`) |
| sentence exact match | 27,18% |
| raw (case-insensitive) WER/CER | 31,69% / 14,52% |

`ĐÃ CHẠY` — tính lại từ `outputs/asr/history_transcripts.jsonl` bằng chính `lifemate.asr.metrics.evaluate_asr`, khớp từng chữ số với `DATASET_PREPARATION_AUDIT.md` §15.3.

### F.4 `for_wer` vs `to_llm_text`

Hai hàm, hai mục đích, không được tráo (`ĐỌC TỪ FILE` `src/lifemate/asr/normalization.py:10-14`):

| hàm | dùng ở đâu | tính chất |
|---|---|---|
| `for_wer` | **chỉ** đánh giá | phá huỷ: NFC, lowercase, bỏ dấu câu, mở số thành chữ |
| `to_llm_text` | text đưa vào Qwen | bảo thủ: NFC, gộp khoảng trắng, giữ dấu câu/số |

Chuẩn hoá cho LLM được áp **một lần**, backend-agnostic, tại `scripts/infer.py:347-350`; backend trả text verbatim và giữ `raw_transcript` (`ĐỌC TỪ FILE`). Case-fix D-20 nằm ở **biên ASR** (`gipformer_backend.py:351`), không nằm trong scorer — đúng chỗ.

### F.5 Ảnh hưởng của WER lịch sử lên task

Đo trực tiếp, không ngoại suy (`ĐỌC TỪ FILE` audit §15.3): dựng lại mọi decision point hai lần (history gold vs history ASR):

| view | ID-argument ground được |
|---|---|
| history gold | 1290 / 1309 |
| history ASR | 1290 / 1309 |
| **mất** | **0** |

Lý do: argument dạng ID lấy từ **observation** của tool, ASR không chạm tới. Với argument free-text lấy từ lời nói trước đó (34 ca toàn split): 34/34 với gold, **13/34** với ASR → 21/34 mất. Đó là kích thước thật của lệch phân bố: nhỏ về số lượng, 100% cục bộ khi xảy ra.

---

## G. Agent và ToolEnv

### G.1 Tương thích chính thức

| kiểm tra | kết quả | nguồn |
|---|---|---|
| import theo split, xen kẽ trong 1 process | `train 147` → `public_test 122` → `train 147` → `public_test 122`, đúng bản của từng split | ĐÃ CHẠY (audit) |
| `root` đúng nghĩa | `ToolEnv(split, id, root=<thư mục CHỨA split>)`; truyền sai thư mục split bị từ chối | `ĐỌC TỪ FILE` + `test_official_env_root_must_contain_the_split_dir` |
| `state_hash` | 64 hex thường; 40/40 dòng submission khớp `[0-9a-f]{64}`, 40 hash phân biệt | ĐÃ CHẠY |
| validate authoritative | 272/272 ca (audit); phiên này **27.070 ca** gold/drop/empty/số-bound/format/items trên catalog thật → **0 bất đồng** (0 mỗi chiều) — `exclusiveMinimum` nay được catalog áp (D-23), nên cả hai đường (`env.validate` và catalog) cùng verdict | ĐÃ CHẠY |

### G.2 Đúng luật "một env cho mỗi mẫu"

| lượt chạy | mẫu | max env/mẫu | mẫu tạo lại |
|---|---|---|---|
| trước sửa (instrumentation) | 40 | **2** | `public_0002` |
| sau sửa — infer | 40 | **1** | `[]` |
| sau sửa — evaluate closed-loop | 103 | **1** | `[]` |

`ĐÃ CHẠY` (audit §§1.3, 6.4). Dòng submission của lượt sau khớp env sống: **40/40** và **103/103**.

### G.3 Replay gold — độ trung thực

| tập | mẫu | transcript khớp gold | `tool_results` khớp | `final_state` khớp |
|---|---|---|---|---|
| chọn theo nhóm hành vi | 13 | **13/13** | 13/13 | **6/6** |
| ngẫu nhiên (seed 0) | 80 | **80/80** | 80/80 | **61/61** |
| Task A | 103 | 103/103 | — | — |

`ĐỌC TỪ FILE` audit §3. So `final_state` bằng `zaic_tools.utils.canon` (đúng phép so của scorer): **48/48**. Nhóm "gold call trả `ok:false`" **rỗng** (0/1545) — ghi nhận, không bịa.

### G.4 Lọc tool

| kiểm tra | kết quả |
|---|---|
| registry vs allow-list của env (30 mẫu thật) | tool không được phép = **0**, tool ẩn = **0** |
| entry schema của app chưa cài trong prompt (200 mẫu, phiên này) | **0** |
| `alarm.list` trong prompt của 1052 mẫu không cài `alarm` | **0** |
| `installed_apps == []` | registry chỉ còn `sys.ask_missing_info` |

`ĐÃ CHẠY` + `ĐỌC TỪ FILE`. **Lưu ý phân biệt:** tên tool của app khác *có* xuất hiện trong **văn xuôi mô tả** của một tool được cài (vd. `"merchant_id lấy từ food_delivery.list_restaurants"` trong schema `ewallet.*`, hay `"lấy từ contacts.search"` trong `sms.send`) — đo được 339 lần trên 200 mẫu (`ĐÃ CHẠY`). Đó là tài liệu tham chiếu của catalog chứ không phải tool được mời gọi; không có entry `### <tool> (kind=` nào của app chưa cài.

### G.5 Giới hạn vòng lặp (cấu hình đang ship)

| tham số | giá trị | nguồn |
|---|---|---|
| `max_steps` | **8** | `configs/inference.yaml:19` |
| `time_budget_seconds` | **120,0** | `configs/inference.yaml:20` |
| `max_consecutive_failures` | **2** | `src/lifemate/agent/stopping_policy.py:52` |
| `max_repeated_identical_calls` | **2** | `stopping_policy.py:53` |
| temperature / `max_new_tokens` | 0,0 (greedy) / 512 | `configs/inference.yaml` |

Hành vi đo được: lặp đúng một action → `status=repeated_failure reason=invalid_repeat llm_calls=3 env_calls=3`; `max_steps=3` → dừng đúng 3 lệnh LLM. 7 lớp output lỗi đã thử — **không** lớp nào chạm env, không lớp nào bị tính là thành công.

---

## H. Evaluation

### H.1 Chín kiểm tra rò rỉ / đúng đắn đã chạy

| # | kiểm tra | kết quả |
|---|---|---|
| 1 | gold không vào prompt | `prompt sent: ['dat bao thuc 6h30'] | gold absent: True` |
| 2 | `tool_calls` của file dự đoán không bị đọc như gold | `gradeable: False`; `_GOLD_KEYS = ('gold_tool_calls','expected_tool_calls')` |
| 3 | crash / không có `<action>` = 0 | aggregate `0.0` (n=1), per-record `False` |
| 4 | per-record khớp aggregate | `0,5` vs `0,5`, `agree: True` |
| 5 | `val.jsonl` không bị nhầm là prediction | val `True` / prediction `False` |
| 6 | checkpoint thật sự được nạp | stub loader ghi `('Qwen/Qwen3.5-4B', {… 'tuning_mode': 'full'})`, exact_match 1.0 |
| 7 | teacher-forced ≠ closed-loop | hai khoá metric riêng |
| 8 | call-sequence vs final-state | `final_state_correctness: {'available': False, …}` |
| 9 | oracle trên split test | `--split public_test --asr oracle --allow-oracle` → **exit 2**, không ghi file |

`ĐỌC TỪ FILE` audit §6.1. Không đường eval nào mở `*_environment/`: `state.json opens: 2 | non-zaic_tools openers: []`.

### H.2 Crash → 0, có `n_unmeasured`

| generator | `tool_call_exact_match` | `n_unmeasured` | per-record |
|---|---|---|---|
| oracle (trả đúng target) | **1,0** (n=56) | 0 | tất cả `True` |
| file dự đoán bị làm hỏng | 0,2679 | 0 | — |
| crash | **0,0** | **56** | tất cả `False` |

Cơ chế: `metrics.decision_emitted` (`src/lifemate/evaluation/metrics.py:374`) chặn cả aggregate lẫn per-record. **Đây là self-check của evaluator trên gold, không phải điểm của model.**

### H.3 Oracle barrier

| lượt | kết quả |
|---|---|
| `public_test` + `--allow-oracle` | **exit 2**, không file |
| `train` + `--allow-oracle` | chạy, `WARNING: oracle ASR is active on labeled split 'train'`, 5 dòng, CLI BTC `5 lines checked, 0 schema errors`; `asr.raw_transcript == audio_text` cho 5/5 |

### H.4 Closed-loop trên 103 mẫu train thật

Bộ: 103 mẫu đầu của `train` (60 mẫu có ≥3 gold call, tổng **283** gold call), một `ToolEnv` mỗi mẫu, generator kịch bản (không có model).

**Lượt 1 — kịch bản GOLD. Nhãn: evaluator self-check on gold input, KHÔNG phải điểm của model.**

| chỉ số | giá trị |
|---|---|
| `n_run` / `status_counts` | 103 / `{'finished': 103}` |
| `tool_call_exact_match` | **1,0000** (n=103) |
| `sequence_f1` / P / R | 1,0000 / 1,0000 / 1,0000 |
| `argument_exact_match` | 1,0000 (n=283) |
| `tool_selection_accuracy` / `stop_accuracy` | 1,0000 / 1,0000 |
| call đã thực thi / ok | 283 / **283 (100%)**; 0 bị từ chối schema |
| bước LLM | 386 |
| latency trung bình | 0,0026 s (scripted, không model) |
| CLI BTC | `103 lines checked, 0 schema errors` |
| `final_state_correctness` | `available: False` |

**Lượt 2 — kịch bản bị làm hỏng (giữ số call, đổi tên tool sang một read hợp lệ nhưng sai).**

| chỉ số | gold | corrupt |
|---|---|---|
| `tool_call_exact_match` | 1,0000 | **0,0583** |
| `sequence_f1` | 1,0000 | **0,0715** |
| `tool_selection_accuracy` | 1,0000 | 0,0680 |
| `stop_accuracy` | 1,0000 | 0,6990 |
| `call_count_delta` | 0,0 | −0,4660 (−48 call) |
| `status_counts` | `{'finished': 103}` | `{'repeated_failure': 60, 'finished': 43}` |
| `final_state_correctness` | `available: False` | `available: False` |

Kết luận: evaluator tách đúng/sai rõ ràng, và khi call lặp tới mức vi phạm schema thì vòng lặp dừng bằng `repeated_failure`, không "thành công" âm thầm.

**Scorer final-state:** không có trong gói phát hành → `final_state_correctness` giữ `{"available": False, "reason": "official scoring tooling unavailable"}` ở **mọi** artifact (`evaluator.py:485, 659-661, 678`). Không nơi nào điền số thay thế.

---

## I. Regression — lệnh và số đếm thực tế

Hai phép đo khác nhau, **không** được trộn: (a) một lần chạy gate trên **cây hiện tại** (§I.1), và (b) artifact gate đã đóng dấu **PRE-CHANGE** (§I.2).

### I.1 Gate + suite trên cây hiện tại

`ĐÃ CHẠY` phiên này, ngày 2026-11-08, **sau** các bản sửa D-24 (prompt think block), D-25 (`has_environment_dir`), D-15 (ghim catalog vào tail), D-16 (đếm `n_truncated`), D-26 (guard `target_value_truncated`), sau lần rebuild artifact cuối, và sau khi `docker/Dockerfile` được ghim lại (§D.5):

| lệnh | kết quả |
|---|---|
| `python3 tests/run_tests.py` | **`passed=479 failed=0 skipped(needs fixtures)=13`** (skip: `test_tiny_e2e.py`, `test_tiny_generation.py` = 6, plus 7 fixture-taking tests in `test_sandbox_credentials.py`) |
| `.venv/bin/python -m pytest -q` | **`485 passed, 7 skipped`** |
| `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q` | **`491 passed, 1 skipped`** (skip: `tests/test_asr.py:376`) |
| `PYTHONPATH=src python3 scripts/verify_all.py --no-write --gates A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,R,P` | **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 0**, exit `0`, `2026-11-08T15:18:59+0700` (stdlib `python3`) |
| `... /tmp/cudaenv/bin/python scripts/verify_all.py --no-write --gates A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,R,P` | **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 0**, exit `0`, `2026-11-08T15:20:40+0700` (lượt chạy gốc dùng `/tmp/torchenv`, **đã xoá 2026-11-13**; `/tmp/cudaenv` là interpreter torch của bản ghi) |
| `PYTHONPATH=src python3 scripts/verify_all.py --self-test` | exit `0`; phân biệt được cả unknown flag lẫn unknown gate letter |

Ba bộ suite là **479 / 485 / 491** pass, skip **13 / 7 / 1** (đo lại **2026-11-13** với `WANDB_BASE_URL=http://127.0.0.1:9`). Vì sao ba số khác nhau: stdlib `python3` không có `torch`/`transformers`/`peft` nên 6 test tiny bị skip, cộng 7 test trong `test_sandbox_credentials.py` nhận tham số fixture pytest mà runner stdlib không bind được; `.venv` **không cài `torch`** nên 6 test tiny vẫn skip, cộng `test_wandb_tracking.py` skip vì thiếu `wandb`; `/tmp/cudaenv` có đủ `torch 2.15.0+cu132` + `transformers 5.22.2` + `peft 0.21.2` + `wandb 0.31.0` nên chỉ skip `tests/test_asr.py:376`.

**Số cũ là lịch sử:** **468 / 468 / 473** (skip 6/6/1) được đo trên `/tmp/torchenv` — **venv CPU đã bị xoá trong phiên 2026-11-13** (`ls -d /tmp/torchenv` → `No such file or directory`), nên **`/tmp/cudaenv` là interpreter torch của bản ghi**. Bộ ba `467/467/472` còn cũ hơn. Bốn file test đã được thêm trong phiên 2026-11-13: `test_wandb_tracking`, `test_sandbox_probe`, `test_sandbox_credentials`, `test_sandbox_measured`.

**Artifact trên đĩa:** bốn file `outputs/verification/verify_all_current_tree_{stdlib,torch}.{json,txt}` — stdlib `2026-11-08T16:02:16+0700` sha256 `910c21b3…`, torch `2026-11-08T16:04:09+0700` sha256 `5442c0ce…`, hai bảng render `1b8230a1…` / `982fe52b…`; **chỉ chứng minh 17 gate không-Q** (footer ghi `REAL DATASET TESTED: not probed in this run / OFFICIAL SDK: not probed in this run / GPU TRAINING TESTED: not probed in this run`), **KHÔNG** dùng làm bằng chứng dữ liệu thật / official SDK / GPU — chi tiết và lý do cần artifact nằm ở §9.1 của `docs/FINAL_TRAINING_EXPERIMENT_READINESS.md`.

**Gate Q không được chạy: operator đã tạm dừng toàn bộ công việc GPU**, nên gate môi trường-thật/GPU là **CHƯA ĐO** trong lần chạy này — **không** phải PASS, và tài liệu này không khẳng định bất kỳ kiểm chứng GPU nào. Vì Q bị loại trừ, tổng là **17 gate chọn**, không phải 18.

Gate N ở lượt này: **17 docs**, **354 flag** kiểm trong `--help`, **0 lỗi / 0 flag bị nêu**. Gate G khác nhau giữa hai interpreter và **cả hai PASS**: stdlib báo `test_training.py masking tests pass (gradient anti-leak torch-gated)`, torch báo `gradient anti-leak test verified (torch active)` — bản torch **mạnh hơn**. Gate R: cả hai PASS, mỗi bên chọn tiny runner riêng, 6 test pass như nhau. Gate P (đo lại 2026-11-13): stdlib `run_tests` `passed=479 skipped=13`; `/tmp/cudaenv/bin/python run_tests` `passed=484 skipped=8`; pytest `485 passed, 7 skipped` (`.venv`) / `491 passed, 1 skipped` (`/tmp/cudaenv`).

`grep -c '^def test_'` trên `tests/*.py` = **492** hàm trong **23** file; runner thu 479 (stdlib, 13 skip vì thiếu `torch` và vì fixture pytest) / 484 (`/tmp/cudaenv`, 8 skip vì fixture) — chênh lệch là do gate/skip, không phải test chết.

### I.2 Artifact `outputs/verification/verify_all.json` — **PRE-CHANGE**, không phải số hiện tại

Artifact được đóng dấu `meta.timestamp = 2026-11-08T13:25:54+0700`, tức **trước mọi thay đổi của phiên này**, và báo **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 1** (cổng BLOCKED: **Q**). Registry có **18 gate tổng cộng; gate Q là gate duy nhất bị loại**, nên 18 − 1 = 17 gate được chọn hiện nay (§I.1) — con số 18 của artifact và 17 của lần chạy hiện tại là cùng một registry, không phải một gate bị mất. Nó được giữ **chỉ như bản ghi lịch sử**: `regression` của nó ghi `passed=457`, và **457 là số lịch sử, không phải trạng thái hiện tại** — trạng thái hiện tại là §I.1 (17/17 PASS, suite 479/485/491).

Nội dung BLOCKED của Q trong artifact đó, nguyên văn: `cannot verify here: GPU training/inference, cloud submit (wandb missing), docker build/runtime, official final-state scorer (evaluator reports final_state_correctness available=False). Verified here: real dataset, official ToolEnv (zaic_tools) (splits public_test 634, train 1545 under ./data)`.

Footer của artifact: `CODE COMPLETE (this repo) / SYNTHETIC TESTED (fixtures) / REAL DATASET TESTED: yes (public_test 634, train 1545) / OFFICIAL SDK: yes (public_test, train) / REAL MODEL TESTED: no / GPU TRAINING TESTED: no / COMPETITION VALIDATED: no`.

Không phá huỷ artifact: sha256 `138cfd4df553e1748d2d3a6043e7ec90643acd655143b62bf52ab3a645fd3f9b` **không đổi**, `meta.timestamp` vẫn `2026-11-08T13:25:54+0700` — lần chạy §I.1 dùng `--no-write` nên không ghi đè.

**Số cũ không còn đúng:** 468/468/473 (đo trên `/tmp/torchenv`, **đã xoá 2026-11-13**), 467/467/472, và 457/6, 457/6, 462/1, 455/6, 460/1, 461 (và trước đó 450/6, 452/6, 457/1, 439/444, 380/385) là số **lịch sử** — trước các defect D-24/D-25/D-15/D-16/D-26, trước lần rebuild artifact cuối, trước bốn test file thêm trong phiên 2026-11-13, hoặc đo trên đường dẫn không còn tồn tại. Dùng bảng §I.1.

---

## J. Ma trận sẵn sàng

| # | hạng mục | trạng thái | căn cứ một dòng |
|---|---|---|---|
| 1 | Trích xuất dataset (zip → `data/train`, `data/public_test`) | **PASS** | 835,8 MiB, 1545 + 634 mẫu, 5036 + 1933 wav, 0 file hỏng (A) |
| 2 | Hợp đồng dữ liệu / `private_test` | **PASS** | khớp `data/README.md`; `private_test` vắng đúng thiết kế |
| 3 | EDA trên dữ liệu thật | **PASS** | 1545/1545 audio OK, đủ tool/trajectory/imbalance (B) |
| 4 | Dựng SFT (4368+482, 0 mẫu bị bỏ / 17+2 decision point bị bỏ) | **PASS** | manifest/`stats.json` khớp; deterministic theo `group_hash`, rebuild 2 lần sha256 giống hệt (C.1) |
| 5 | Căn chỉnh call/result, nhân quả prompt | **PASS** | 1545/1545 căn chỉnh; prompt bước *k* chỉ chứa `<k` |
| 6 | Leakage (final_state, khoá nhãn, trùng lặp, split) | **PASS** | 0/4850; 0/9700 message; 0 cặp trùng; 0 cặp train↔val; train∩val = 0 (C.5) |
| 7 | Loss masking | **PASS** | 4850/4850 kept, 0 hàng all-`-100`, 0 decode lệch, 0 token prompt bị supervise, tail think-block đóng 4850/4850 (C.6) |
| 8 | Ngân sách chuỗi 8192 | **PASS** | 47/4850 (0,97%) cắt; schema tool 0/3305 sau khi ghim catalog; guard `target_value_truncated` bỏ 0 ở 8192 / 122 ở 4096 (C.4) |
| 9 | Tích hợp `zaic_tools` theo split | **PASS** | 147/122 đúng bản, xen kẽ cùng process (G.1) |
| 10 | Luật một ToolEnv / mẫu | **PASS** | max 1, repeated [] trên 40 và 103 mẫu (G.2) |
| 11 | Replay gold qua env | **PASS** | 13/13, 80/80, final_state 6/6 và 61/61, canon 48/48 (G.3) |
| 12 | Lọc tool / không rò tool chưa cài | **PASS** | 0 entry schema của app chưa cài; 0 `alarm.list` ở 1052 mẫu (G.4) |
| 13 | Vòng lặp agent + stopping policy | **PASS** | 8 bước / 120 s / 2 lỗi / 2 lặp; 7 lớp output lỗi không chạm env |
| 14 | Định dạng submission | **PASS** | 40 dòng, CLI BTC `0 schema errors`; resume giữ nguyên byte |
| 15 | Evaluation (9 kiểm tra + crash→0 + oracle barrier) | **PASS** | H.1–H.3 |
| 16 | Chất lượng ASR trên dữ liệu thật | **PASS** | WER 17,91% / CER 11,13% / RTF 0,0114, 0 lỗi (F.2) |
| 17 | Asset ASR + hash verification | **PASS** | 6/6 verified ở revision pin (F.1) |
| 18 | Seam history ở suy luận | **PASS (có điều kiện)** | máy trực tiếp: 1299/1299 lượt trên 419/419 mẫu (D-22). Container: `docker/entrypoint.sh:65-68` truyền `--history-transcripts` khi `HISTORY_TRANSCRIPTS` set, **exit 2** khi set mà không đọc được; `cloud/sandbox_runner.py:121,254,272` mount `/history/history_transcripts.jsonl` ro (mặc định `<output_dir>/asr/history_transcripts.jsonl`, thiếu thì bỏ cờ + note nêu rõ 419/634); `docker/README.md:36,49,65,95-98` tài liệu hoá env var + mount thứ 4 — **thiếu mount thì run dừng**: `scripts/infer.py` trả **exit 2** (`… would render N history user turn(s) blank`) trước khi nạp trọng số, chỉ `--allow-missing-history` (chẩn đoán) mới đi tiếp — không còn cảnh báo rồi chạy tiếp |
| 19 | Sẵn sàng full FT (cấu hình + ước lượng) | **PARTIAL** | cấu hình đúng, ước lượng 69,98 GB **không vừa** card; chưa chạy thật |
| 20 | Sẵn sàng LoRA (target module + ước lượng) | **PARTIAL** | 12 suffix tồn tại thật, **32.464.896** trainable (đo, 0,766% của 4.205.751.296), **11,49 GB** vừa — nhưng **chưa chạy** |
| 21 | Checkpoint / resume | **PASS** | resume guard đúng; 4096→8192 bị từ chối là hành vi đúng (E.6) |
| 22 | `verify_all` gates | **PASS 17 / 17** trên cây hiện tại; gate Q **KHÔNG CHẠY** (GPU bị tạm dừng) → CHƯA ĐO | §I.1: hai interpreter, `2026-11-08T15:18:59+0700` và `2026-11-08T15:20:40+0700`, FAIL 0 / SKIP 0 / BLOCKED 0. Artifact cũ `2026-11-08T13:25:54+0700` (17 PASS / 1 BLOCKED) là **PRE-CHANGE**, xem §I.2 |
| 23 | Docker | **NOT RUN / UNVERIFIED** | hai lý do: không có daemon; và pin cũ `cu124` không có kernel `sm_120` nên image cũ **không thể** chạy GPU ở đây dù có daemon. Đã sửa pin sang CUDA 12.8.1 + `whl/cu128` (`Dockerfile:3-12,31,51-53,73`) nhưng **image chưa từng build, chưa từng chạy**. `ENTRYPOINT` đã tồn tại (`Dockerfile:126`) |
| 24 | Cloud / W&B | **NOT RUN** | `wandb` chưa cài, chưa có credential |
| 25 | Model thật (1 bước train / generate) | **NOT RUN** | không có byte trọng số |
| 26 | Huấn luyện GPU | **NOT RUN** | torch `2.15.0+cpu` (GPU có mặt) |
| 27 | Điểm trên `public_test` | **FAIL (không thể)** | 634/634 không nhãn → không tồn tại điểm |
| 28 | Scorer final-state chính thức | **NOT RUN** | BTC chưa phát hành script chấm |
| 29 | Sẵn sàng thi đấu | **FAIL** | chưa có checkpoint huấn luyện nào từ model thật |

---

## K. Quyết định cuối

### K.1 Tám câu hỏi, trả lời thẳng

**1. Repo có sẵn sàng nhận dữ liệu thật và chạy tích hợp không?**
**Có.** Dữ liệu đã được trích xuất và đi hết đường: EDA → SFT → ToolEnv → agent → submission → evaluation. Không còn `exit 2` nào trên bộ dữ liệu đầy đủ.

**2. Dữ liệu thật đã được kiểm chứng chưa?**
**Có.** 1545 + 634 mẫu, 0 file audio hỏng, SFT 4850 decision point (0 mẫu bị bỏ, 19 decision point bị bỏ vì giá trị tham chiếu không quan sát được), leakage 0, mask 0 lỗi, replay gold 100%, ASR 1545 câu đo thật.

**3. Model thật đã được kiểm chứng chưa?**
**Chưa.** Không có byte trọng số Qwen3.5-4B. Mọi thứ về model là **kiến trúc + cấu hình + meta-device**, không phải trọng số. Không có một `generate()` nào chạy.

**4. Đã huấn luyện trên GPU chưa?**
**Chưa.** GPU có (RTX 5070 Ti, 17,09 GB) nhưng torch là `2.15.0+cpu`, `cuda = None`. Con số "đã train" duy nhất trong repo là **tiny model CPU** (`step-2`, 98,7 KB) — chứng minh *đường ống*, không chứng minh *hội tụ*.

**5. Có tạo được file nộp hợp lệ không?**
**Có, về định dạng.** 40 dòng do env sinh, CLI chính thức `40 lines checked, 0 schema errors`, resume giữ nguyên byte. Nhưng nội dung đến từ **generator kịch bản**, không từ model — nên đây là kiểm tra định dạng, không phải kết quả thi.

**6. Có điểm số nào không?**
**Không.** `public_test` không nhãn; `final_state_correctness` không tồn tại trong gói phát hành. Các số 1,0 / 0,0583 / 0,2679 là **self-check của evaluator trên gold input**, tuyệt đối không phải điểm của model.

**7. Full FT hay LoRA?**
**LoRA là đường chạy được trên máy này** (11,49 GB ước lượng với adapter **đo được** 32.464.896 tham số, vs 15,45 GB trống khi idle — nhưng lúc audit chỉ còn 11,82 GB, phải đo lại). **Full FT (69,98 GB) không vừa** bất kỳ card đơn nào dưới 80 GB; cần 2×40 GB với ZeRO-3 hoặc thuê máy lớn. Cả hai chưa chạy.

**8. Đã sẵn sàng thi chưa?**
**Chưa.** Thiếu đúng một thứ: một checkpoint sinh ra từ Qwen3.5-4B trên dữ liệu này. Mọi thứ khác (dữ liệu, SDK, agent, evaluation, ASR, cấu hình) đã ở trạng thái chạy được.

### K.2 Năm mức bằng chứng

| mức | kết luận | căn cứ |
|---|---|---|
| **CODE READY** | **CÓ** | 17/17 gate PASS trên cây hiện tại (§I.1, gate Q **không chạy** vì GPU bị tạm dừng — CHƯA ĐO, không phải PASS); suite 479/485/491 pass (skip 13/7/1, đo 2026-11-13); 0 placeholder; 0 finding compliance |
| **REAL DATA VERIFIED** | **CÓ** | 1545 + 634 mẫu thật đi hết pipeline; EDA/C.4/C.6/G.3/H.4 đều đo trên dữ liệu phát hành |
| **REAL MODEL VERIFIED** | **KHÔNG** | không có byte trọng số; chỉ meta-device (4.205.751.296 tham số) và tiny model |
| **GPU TRAINING VERIFIED** | **KHÔNG** | torch `2.15.0+cpu`; 0 optimizer step trên model thật |
| **COMPETITION READY** | **KHÔNG** | không checkpoint, không điểm, `final_state` không chấm được ở local, Docker/cloud chưa chạy. Pin Dockerfile đã sửa sang `cu128`/CUDA 12.8.1 cho `sm_120` nhưng image **chưa từng build, chưa từng chạy** → **UNVERIFIED** (image theo pin cũ `cu124` thì không thể chạy GPU ở đây dù có daemon) |

### K.3 Việc còn lại, theo thứ tự chặn

1. Cài torch CUDA (cu128 cho sm_120) — phần cứng đã có.
2. Chạy **LoRA** trước (`configs/training_lora.yaml`, 8192, microbatch 1, checkpointing ON); đo `torch.cuda.max_memory_allocated()` ở optimizer step đầu để thay ước lượng bằng số thật.
3. Nếu cần full FT: máy ≥80 GB hoặc sharding.
4. Chạy `infer.py --split public_test --asr gipformer --history-transcripts outputs/asr/history_transcripts.jsonl` với checkpoint thật; validate bằng CLI chính thức. Trong container đường này đã nối sẵn: `docker/entrypoint.sh` thêm `--history-transcripts "$HISTORY_TRANSCRIPTS"` khi biến được set và **exit 2** (`ERROR: HISTORY_TRANSCRIPTS set but not readable`) nếu set mà không đọc được — không bao giờ degrade âm thầm; `cloud/sandbox_runner.py` mount `/history/history_transcripts.jsonl` read-only và truyền cờ, mặc định lấy `<output_dir>/asr/history_transcripts.jsonl`, thiếu file thì **bỏ cờ kèm note**. Lệnh container tương ứng:
   ```bash
   docker run --rm --network none \
     -v "$PWD/data:/data:ro" -v "$PWD/ckpt/step-N:/models/ckpt:ro" \
     -v "$PWD/outputs:/outputs:rw" \
     -v "$PWD/outputs/asr/history_transcripts.jsonl:/history/history_transcripts.jsonl:ro" \
     -e HISTORY_TRANSCRIPTS=/history/history_transcripts.jsonl \
     lifemate:local
   ```
   **Điều kiện trung thực:** khi mount thứ 4 có mặt, `scripts/infer.py` điền **1299/1299** lượt trên **419/419** mẫu. Không mount (hoặc trỏ sai đường dẫn) thì run **dừng với exit 2** `ERROR: no --history-transcripts file was configured; split=public_test would render 1299 history user turn(s) blank` (`ĐÃ CHẠY`, chạy lại đúng argv của entrypoint với bố cục `/tmp/ctr/{data,history,outputs}`); chỉ `--allow-missing-history` (chẩn đoán, không dùng cho submission) mới chạy tiếp với lịch sử trống.
5. Build/run Docker có `--network none` trên host có daemon; submit cloud khi có credential.

---

## Phụ lục — Bản đồ artifact

| artifact | nội dung | trạng thái |
|---|---|---|
| `data/train/train.jsonl`, `data/public_test/public_test.jsonl` | dữ liệu gốc | 1545 / 634 |
| `outputs/datasets/train/{train.jsonl,val.jsonl,manifest.json,stats.json}` | SFT build | 4368 / 482, 8192 |
| `outputs/eda/{report.md,summary.json,tables/,charts/}` | EDA | 10 PNG + manifest |
| `outputs/asr/history_transcripts.jsonl` | transcript lượt lịch sử | 4790 dòng, 0 lỗi |
| `outputs/verification/{verify_all.json,verify_all.md}` | 18 gate @ `2026-11-08T13:25:54+0700` — **PRE-CHANGE**, chỉ là bản ghi lịch sử (`regression` ghi `passed=457`) | 17 PASS / 1 BLOCKED, **trước** mọi thay đổi của phiên này; trạng thái hiện tại là §I.1 (17 gate chọn, PASS 17/17, gate Q không chạy vì GPU bị tạm dừng) |
| `outputs/checkpoints/full/{step-2,crash-2}` | checkpoint tiny CPU (smoke) | **không** phải Qwen |
| `models/gipformer-68M-rnnt/` | 6 file int8, hash-verified | 73.515.623 byte |

---

## Phụ lục B — Những con số nêu trong brief mà tôi KHÔNG xác minh được

Mọi mục dưới đây ghi `CHƯA ĐO`; báo cáo không dùng chúng làm căn cứ (chỗ nào cần thì đã thay bằng số đo được, ghi rõ).

| con số trong brief | trạng thái | số đo được thay thế | lý do |
|---|---|---|---|
| `x-one-of` parity "217 → 0" | `CHƯA ĐO` | 18/487 ca là lỗi `x-one-of` (audit §2); 54/54 ca `x-one-of` đồng ý sau sửa | không tài liệu nào trong repo ghi 217 cho `x-one-of`; 217 là **số mẫu missing-info** trong EDA, khả năng cao là nhầm nguồn |
| leak guard false-positive "267 mẫu" | `CHƯA ĐO` | **373** mẫu có `final_state == {}`; **543** decision point từ chúng | không tái tạo được định nghĩa nào cho ra 267 (373 / 332 / 262 / 232 / 111 đều đã thử) |
| checkpoint thiếu "14 params" | `CHƯA ĐO` | repo test: config 3 layer trên trọng số 2 layer, layer 2 random std 0,0198, run vẫn chạy | phép đo trong repo dùng model tiny, không phải 14 tham số của checkpoint thật |
| LoRA VRAM "72,8% of free" | `SUY RA` khác | 11,49 / 15,45 = **74,4%** | 72,8% tương ứng 11,25 GB; estimator hiện tại cho **11,49 GB** với adapter đo được (32.464.896) |
| "3491/3491 history trắng" (trước sửa) | `ĐỌC TỪ FILE` + `ĐÃ CHẠY` gián tiếp | 3491/3491 lượt user **không có** khoá `content` (chỉ `audio` + `audio_text`) → renderer cũ đọc `content` ra rỗng | trạng thái "trước sửa" không còn trong working tree (git chỉ có 1 commit `a150819`); bằng chứng là dữ liệu + đường code cũ |
| "3501/3541 payload bị mất" | `ĐÃ CHẠY` | **3501/3541** result mang payload ngoài khoá điều khiển (40 cái còn lại là `{"ok": true}` thuần) | khớp |
| ASR latency "82 ms / RTF 0,0114" | `ĐÃ CHẠY` | 82,2 ms, RTF 0,01137 (lần chạy lặp: 87,4 ms, 0,0121) | khớp; chênh giữa hai lần là nhiễu máy |
| "language" `0,0556 → 0,9444` case agreement | `ĐỌC TỪ FILE` (audit) | đo riêng của tôi: 0,2146 → 0,9650 trên 106.813 chữ cái | hai định nghĩa khác nhau (audit chỉ tính phần request đã lọc); cả hai đều cho thấy lowercasing sửa gần hết |
| "1 tool/147 chưa xuất hiện" | `ĐÃ CHẠY` | `contacts.get` | khớp |

Tài liệu nguồn (không sửa): `outputs/datasets/DATASET_PREPARATION_AUDIT.md`, `docs/REAL_DATA_AGENT_EVAL_AUDIT.md`, `outputs/eda/report.md`, `docs/TRAINING_GUIDE.md`, `docs/GIPFORMER_INTEGRATION_AUDIT.md`, `docs/FINAL_PRETRAINING_READINESS_AUDIT.md`, `data/README.md`, `data/VERSION`.
