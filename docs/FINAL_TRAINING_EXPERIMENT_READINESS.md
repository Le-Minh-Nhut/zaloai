# Sẵn sàng cho thí nghiệm huấn luyện THẬT đầu tiên — Qwen3.5-4B / LifeMate

**Repo:** `~/data/zaloai` — Zalo AI Challenge 2026, Track 1 "LifeMate"
**Ngày đo:** 2026-11-08 · **Người viết:** tổng hợp từ các phép đo của worker trong phiên; mọi con số đã được đọc lại từ artifact/code trước khi dẫn.
**Phạm vi:** *không* lặp lại phần dữ liệu cơ bản — xem `docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md` cho EDA, cấu trúc mẫu, leakage, masking. Tài liệu này chỉ trả lời một câu: **có chạy được thí nghiệm SFT Qwen3.5-4B thật ngay bây giờ không, và nếu không thì thiếu đúng cái gì.**

## 0. Quy ước bằng chứng

| nhãn | nghĩa |
|---|---|
| `ĐÃ CHẠY` | lệnh đã chạy trong phiên này (hoặc phiên worker, output được dẫn lại), kết quả nêu nguyên văn |
| `ĐỌC TỪ FILE` | đọc từ `file:line` hoặc artifact có đường dẫn nêu rõ |
| `SUY RA` | suy luận số học từ dữ kiện đã đo, không phải phép đo trực tiếp |
| `CHƯA ĐO` | không đo được/không tái tạo được, kèm lý do |
| `NOT RUN` | hành động chưa từng được thực thi ở đây |
| `BLOCKED` | không thể thực thi vì thiếu điều kiện vật lý (trọng số, daemon, GPU runtime) |

**Tuân thủ:** không lệnh nào trong phiên này đọc, liệt kê hay glob bất kỳ đường dẫn chứa thành phần thư mục khớp `*_environment`. Dữ liệu app chỉ được chạm qua `zaic_tools.ToolEnv`.

**Giới hạn đo của phiên này:** không có byte trọng số Qwen3.5-4B (chỉ `config.json` + `model.safetensors.index.json` + tokenizer trong HF cache); `.venv` **không cài `torch`** và bản CPU `/tmp/torchenv` đã bị xoá trong phiên 2026-11-13, nên `/tmp/cudaenv` là interpreter torch duy nhất còn tồn tại; **không có binary và không có daemon `docker`** (`ĐÃ CHẠY` `which docker` → exit 127, `docker info` → 127). Vì vậy mọi kết luận về *model thật* dưới đây là kiến trúc + cấu hình + meta-device, **không phải** kết quả huấn luyện.

### 0.1 Trạng thái artifact nền (đã đọc lại, không tin lời kể)

| hạng mục | giá trị | nguồn |
|---|---|---|
| `outputs/datasets/train/train.jsonl` | **4368** dòng | `ĐÃ CHẠY` `wc -l` |
| `outputs/datasets/train/val.jsonl` | **482** dòng | `ĐÃ CHẠY` |
| **sha256 của build này** (neo mọi con số vào *một* build, không phải file đang trôi) | `train.jsonl` **`0162f94a`**, `val.jsonl` **`633ed31f`**, `manifest.json` **`67b26f5b`**, `stats.json` **`6a8b0e9d`** (4 byte đầu) | `ĐÃ CHẠY` `sha256sum outputs/datasets/train/*` |
| `manifest.n_train_decision_points` / `n_val_decision_points` | 4368 / 482 | `ĐỌC TỪ FILE` `manifest.json` |
| `n_truncated` train / val | **47** (0,01076) / **0** | `ĐỌC TỪ FILE` `manifest.fit` |
| `n_dropped` (mẫu) train / val | **0** / **0** | `ĐỌC TỪ FILE` |
| `dropped_decision_points` | `{"unobservable_ref_target": 17}` train, `{…: 2}` val | `ĐỌC TỪ FILE` |
| `group_hash` | `53a8c3536ac6eee4ed245f0fb33c06a54585e9e4919cd6b13f6492ae9d2f4a9e` | `ĐỌC TỪ FILE` (khớp brief) |
| `n_groups` / `ratio` / `seed` / `history_limit` | 1544 / 0,1 / 42 / 10 | `ĐỌC TỪ FILE` |
| `max_seq_length` | 8192 | `ĐỌC TỪ FILE` `fit.max_seq_length` |
| `unobservable_ref` (mẫu bị gắn cờ) | **12** train / **1** val | `ĐỌC TỪ FILE` `stats.json` |
| target `tool_call` (train+val) | **3522**, trong đó **217** `sys.ask_missing_info` → **3305** non-meta | `ĐÃ CHẠY` đếm lại toàn bộ 4850 dòng |
| target `finish` | 1328 | `ĐÃ CHẠY` |

`stats.json` khai `fit.method: "tokenizer"`, `caveat: null` — số token ở `fit` là thật. Khối `prompt_tokens_estimate` (mean 3733,7) vẫn là ước lượng `chars/4`, **không** dùng làm căn cứ; số token thật nằm ở §8.7.

---

## 1. Reference-ID observability — câu hỏi "496 mẫu" đã được giải

### 1.1 Cơ chế hiện tại (đọc code, không suy đoán)

`src/lifemate/data/trajectory_builder.py`:

| hàm | dòng | việc nó làm |
|---|---|---|
| `_ref_keys(spec)` | 371-390 | trả về các tên tham số mà catalog đánh `x-ref`/`x_ref`/`ref`/`x-from`/`x_from`. **`x-match` bị loại có chủ đích**: 40/40 lần xuất hiện trong catalog chính thức là free text kiểu `fuzzy` (`content`, `title`, `body`, `note`, `subject`, `address`, `pickup`/`dropoff`) do model tự soạn, không phải giá trị copy từ observation |
| `_visible_text(sample, prior_obs, history_limit, prior_calls)` | 393-428 | dựng **đúng** blob mà prompt hiển thị: `audio_text` + `context` + history **đã cắt theo `history_limit`** + tool_call/tool_result của từng lượt + **hành động trước đó** (`prior_calls`) + observation trước đó |
| `_unobservable_ref_args(calls, specs, sample, observations, history_limit)` | 431-467 | chỉ xét call **kind=write**; với mỗi tham số thuộc `x-ref` hoặc có hậu tố `_id`/`_ref`/`_name`, nếu giá trị **không** xuất hiện trong blob trên thì gắn cờ, theo từng `call index`: `{index: ["stepK:tool.arg=value"]}` |
| `build_sft_dataset` | 661-703 | điểm nào có cờ thì **bỏ đúng điểm đó** (`n_dropped_decision_points`), tăng `unobservable_ref` (số mẫu), **không sửa** giá trị vàng, **không** bỏ cả mẫu |

Hai chi tiết là bản chất của bản sửa:

1. `_visible_text` dùng **cùng cửa sổ** với renderer (`history_limit`) và tính **hành động của chính model** ở các bước trước. Docstring nêu ca thật: `chat_ott.send_location.chat_id=chat_9017` ở step 2 của `train_1657` hiện trong action của step 1 — bản cũ báo "không thấy" trong khi prompt in ra rõ ràng.
2. Bỏ **theo decision point**, không theo mẫu: một tham số mồ côi không còn làm hỏng cả mẫu.

### 1.2 "496" chưa bao giờ là số defect

496 là số mẫu **trượt một heuristic read-before-write ngây thơ** ("mọi write có tham số giống ID phải thấy giá trị ở đâu đó trước đó"). Đo lại trên toàn bộ 1545 mẫu (`/tmp/rbaudit/promptlevel.json`, cửa sổ 10 lượt):

| phân loại (đếm theo *sự có mặt*, có thể chồng lấn) | số mẫu | nghĩa |
|---|---|---|
| lấy giá trị từ **tool observation / action trước** | **182** | quan sát được thật |
| lấy từ **history hội thoại** (trong cửa sổ render) | **65** | quan sát được thật |
| lấy từ **enum của catalog** | **118** | quan sát được thật (giá trị nằm trong schema) |
| **không có tham số tham chiếu nào** | **208** | heuristic bắt sai ngay từ giả thiết |
| **thật sự không quan sát được** | **2** | defect thật |

`ĐÃ CHẠY` — tính trực tiếp từ `promptlevel.json` + `unsat496.json` (giao đúng 496 mẫu).

Vì các nhóm trên chồng lấn (12 mẫu vừa có nguồn catalog vừa có nguồn history; nhiều mẫu có cả nguồn prior-tool lẫn nguồn khác), một **phân hoạch rời** theo thứ tự ưu tiên cho con số khác:

| phân hoạch rời (ưu tiên prior → history → enum → không có arg → không quan sát được) | số mẫu |
|---|---|
| prior tool result/action | **182** |
| history | **51** |
| catalog enum | **54** |
| không có tham số tham chiếu | **208** |
| không quan sát được | **1** (mẫu thứ 2 nằm trong nhóm khác vì còn arg quan sát được) |
| tổng | **496** |

`ĐÃ CHẠY`. Kết luận: **496 không phải số lỗi; nó là số mẫu mà heuristic cũ không giải thích được.** Số lỗi thật là 2 mẫu (trong 496) và 19 tham số trên toàn bộ dataset.

### 1.3 Tham chiếu "proper" trên 1172 mẫu có write

Định nghĩa *references proper* = `x-ref`/`x-from` + hậu tố `*_id`/`*_ref`/`*_name` + enum. Trên toàn bộ 1172 mẫu có ít nhất một write:

| nguồn đầu tiên | số arg |
|---|---|
| prior tool result + prior action | **939** (`prior_observation` 882, `prior_action` 57) |
| context / history | **682** (`catalog` 378, `history` 304) |
| **không quan sát được** | **19** |
| **ambiguous** | **0** |
| **tổng** | **1640** |

`ĐÃ CHẠY`.

`x-match` (free text) là **nhóm đối chứng**, không phải tham chiếu: 643 arg, trong đó 256 "không thấy ở đâu" — đúng như thiết kế, vì model tự soạn câu. Nếu tính chúng là tham chiếu thì con số "lỗi" phình lên 275 arg/242 mẫu — đó chính là nguồn của 319.

### 1.4 Đối chiếu cờ cũ `319 mẫu / 943 decision point`

| điều brief nói | đo được | kết luận |
|---|---|---|
| 319 mẫu bị gắn cờ (train) | **319** | `ĐÃ CHẠY` khớp |
| 943 "decision point" | **378** decision point mang cờ ở train; **410** toàn bộ (train+val) | `ĐÃ CHẠY` — **943 là tổng số gold call của 319 mẫu đó**, không phải số decision point. `ĐỌC TỪ FILE` `DATASET_PREPARATION_AUDIT.md:71` gọi nhầm |
| cờ "báo thừa ~91%" | **98,1%** nếu chỉ tính tham chiếu proper (chỉ **6/319** mẫu có defect thật); **18,2%** nếu tính cả `x-match` | `ĐÃ CHẠY`. Con số 91% **CHƯA ĐO** — không tái tạo được định nghĩa nào cho ra 91% |
| cờ bỏ sót 13 mẫu / 19 arg | **13 mẫu / 19 arg** (12 mẫu + 17 arg train; 1 mẫu + 2 arg val) | `ĐÃ CHẠY` khớp |
| nguyên nhân over-report | **409/410** arg bị gắn cờ là `REF_FUZZY` (`x-match`); 1 arg còn lại là strict và **đúng** | `ĐÃ CHẠY` |
| nguyên nhân under-report | cờ cũ tìm trong **toàn bộ** history, renderer chỉ hiện 10 lượt → 18/19 arg "không quan sát được" nằm trong các lượt bị cắt | `ĐÃ CHẠY` |

Một ca đơn lẻ: `train_1696` step 2, `chat_ott.send_message.chat_id=chat_6142`. Kiểm tra bản ghi gốc: `chat_6142` **không** xuất hiện trong `audio_text`, `context`, `history`, hay bất kỳ `tool_result` nào (`chat_ott.list_chats` trả `chat_329`, `chat_858`, `chat_559`, `chat_313`, `chat_7450`); nó chỉ tồn tại trong chính nhãn `tool_calls`. `ĐÃ CHẠY` — đây là **bất nhất của người tổ chức**: gold yêu cầu một ID không tồn tại trong thế giới của mẫu. Không sửa được ở phía repo, và bị bỏ bởi rule mới (đúng cách: không bao giờ bơm giá trị vàng vào prompt).

### 1.5 Trạng thái sau sửa

| chỉ số | giá trị | nguồn |
|---|---|---|
| mẫu bị gắn cờ | **12** train / **1** val | `ĐỌC TỪ FILE` `stats.json` |
| decision point bị bỏ | **17** train / **2** val = **19** | `ĐỌC TỪ FILE` |
| mẫu bị bỏ | **0** | `ĐỌC TỪ FILE` |
| tham số bị thay đổi | **0** (không inject, không sửa vàng) | `ĐỌC TỪ FILE` `trajectory_builder.py:431` docstring |
| test ghim | `tests/test_trajectory.py:372` `test_ref_arg_only_in_dropped_history_is_unobservable` — khẳng định cả hai chiều: ngoài cửa sổ → bỏ điểm, trong cửa sổ → giữ | `ĐỌC TỪ FILE` |

Kiểm tra độc lập sau rebuild: quét toàn bộ 4850 record, mọi tham số `*_id`/`*_ref`/`*_name` của target phải xuất hiện trong `prompt_body` của chính nó — **0 ca vi phạm** (`ĐÃ CHẠY` `/tmp/underdetect.py`: `id_args 1507, invisible 0`).

---

## 2. Ba ca schema bị cắt — đã nhận diện và xử lý

### 2.1 Ba ca

Ở `max_seq_length=8192` với collator **cũ** (cắt giữa theo head 35% / tail 65%), đúng **3** decision point có target `tool_call` mất khối schema của chính tool được gọi:

| record | tool | số lần chuỗi `query` xuất hiện trong prompt đã cắt | phán quyết |
|---|---|---|---|
| `train_0156#3` | `contacts.search` | **0** | UNANSWERABLE |
| `train_0204#3` | `contacts.search` | **0** | UNANSWERABLE |
| `train_0265#2` | `contacts.search` | **0** | UNANSWERABLE |

`ĐỌC TỪ FILE` `outputs/datasets/DATASET_PREPARATION_AUDIT.md` §14.5 + `tests/test_training.py:218` docstring ("3 of 3324 tool_call prompts lost the block … all three `contacts.search`, whose `query` argument name appears nowhere else" — con số **lịch sử**, đo trên build 4869 record trước khi ghim catalog; trên build hiện tại 4850 record là **0/3305**).

Cả ba rơi vào cùng một tool vì `contacts.search` là tool phổ biến nhất (6,6% tổng số call) và có **một** tham số `query` mà tên chuỗi này không xuất hiện ở bất kỳ khối schema nào khác — mất khối là mất luôn tên tham số. Ở hai trong ba ca, vết cắt rơi **giữa JSON** của dòng `parameters:`.

Nhãn phán quyết: **không** phải lỗi dữ liệu, không phải lỗi nhãn — là lỗi **renderer/collator**: prompt được huấn luyện không chứa thông tin cần để sinh đáp án.

### 2.2 Sửa: ghim section catalog vào tail

`src/lifemate/training/collator.py`: `TOOLS_MARKER = "# Công cụ khả dụng"` (:161) và `_split_at_tools_section(text)` (:179-193) tách `(trước catalog, catalog-và-sau)` bằng phép tìm chuỗi (`None` nếu không có mốc). `build_supervised_example` (:329-352) giữ nguyên prefix `# Yêu cầu` (`REQUEST_MARKER`), **ghim toàn bộ catalog + phần sau vào tail**, chỉ cắt giữa phần *trước* catalog. Hợp lệ vì đo được catalog **luôn vừa** ngân sách tail của mọi record (`ĐỌC TỪ FILE` docstring). Cắt theo văn bản tốn tối đa vài merge ở biên — rẻ hơn nhiều so với mất schema.

### 2.3 Kết quả đo (sau sửa, trên artifact đang ship)

`ĐÃ CHẠY` phiên này — chạy `build_supervised_example` với tokenizer thật trên **cả 4850 record**. Cột "target `tool_call` bị cắt" đếm **non-meta** (`sys.ask_missing_info` tách riêng, `finish` không tính):

| ngân sách | record bị cắt | target `tool_call` bị cắt | mất khối schema của chính tool | mất **toàn bộ** section catalog | bị bỏ vì `target_value_truncated` | độ dài input tối đa |
|---|---|---|---|---|---|---|
| **8192 (hiện tại)** | **47** | **34** | **0 / 3305** | **0** | **0** | 8192 |
| 4096 | 3520 | **2432** | **846 / 3305** | 87 | **122** | 4096 |

Ba record từng hỏng, đo lại ở 8192 (`ĐÃ CHẠY` `/tmp/remeasure_schema.py`, `skip_special_tokens=False` cho cột decode):

| record | tool | `len(input_ids)` | prompt token | target token | token bị cắt | khối `### contacts.search (kind=` |
|---|---|---|---|---|---|---|
| `train_0156#3` | `contacts.search` | **8192** | 8162 | 30 | 387 | **có** |
| `train_0204#3` | `contacts.search` | **8192** | 8162 | 30 | 210 | **có** |
| `train_0265#2` | `contacts.search` | **8192** | 8162 | 30 | 146 | **có** |

Ba record đều chạm đúng ngân sách (8162 + 30 = 8192) và đều **giữ** khối schema — đây là điểm cần chứng minh, và token là đơn vị đo của nó.

**Về ký tự — ba con số khác nhau, không cái nào sai, chỉ khác phép đo** (đo cùng ba record, `ĐÃ CHẠY`):

| phép đo | `train_0156#3` | `train_0204#3` | `train_0265#2` |
|---|---|---|---|
| `len(messages[1]["content"])` — prompt lưu trong `train.jsonl`, **chưa** cắt, **chưa** qua chat template | 25752 | 25501 | 25133 |
| `len(tok.decode(prompt_ids, skip_special_tokens=False))` — prompt **đã cắt** | 26133 | 26507 | 26320 |
| `len(render_prompt_text(...))` — **chưa cắt**, **đã** qua chat template | 27226 | 26975 | 26607 |

Bản trước của tài liệu in **hàng thứ nhất** (25752 / 25501 / 25133) nhưng gọi nó là "len", khiến người đọc decode prompt đã cắt (hàng thứ hai) thấy lệch. Cả ba hàng đều tái tạo được; nay ghi rõ phép đo. Nếu ai đó ra 26133 / 26507 / 26320 thì đó là hàng thứ hai — cùng record, khác invocation (`skip_special_tokens`, và có/không qua chat template).

Kiểm tra độc lập toàn bộ corpus: **3305/3305** prompt `tool_call` non-meta chứa đúng **một** header `### <tool> (kind=` của chính nó (`ĐÃ CHẠY`, `/tmp/leak_rank.py`: `with !=1 own header 0`).

**Phân rã 47 record bị cắt ở 8192 — không phải tất cả đều là `tool_call`** (`ĐÃ CHẠY`): **34** `tool_call` non-meta + **12** `finish` + **1** `sys.ask_missing_info`. Vì vậy phát biểu chính xác là: trong **34** prompt `tool_call` non-meta bị cắt, **0** mất khối schema của chính tool; trên toàn bộ **3305** prompt `tool_call` non-meta, **3305/3305** có khối đó. Tổng số record bị cắt vẫn là 47 và `max len(input_ids)` = **8192**.

### 2.4 Guard thứ hai: `target_value_truncated`

| thành phần | dòng | hành vi |
|---|---|---|
| `_target_values(target_text)` | 197-228 | lấy các giá trị scalar không rỗng của tham số có hậu tố `_id`/`_ref`/`_name` trong `<action>…</action>` |
| `_values_hidden_by_truncation(tokenizer, prompt_ids, target_text)` | 231-252 | decode **prompt cuối cùng** (chỉ khi record thực sự bị cắt), trả về giá trị không còn xuất hiện; tokenizer không decode được → trả `[]` (không bao giờ bỏ dựa trên phỏng đoán) |
| gọi trong `build_supervised_example` | 355-368 | có giá trị bị ẩn → `kept=False`, `reason="target_value_truncated"` |

Đo: **8192 → 0 lần kích hoạt; 4096 → 122 lần** (`ĐÃ CHẠY`, khớp `tests/test_training.py:263` docstring). Đây là lỗ **tiềm ẩn** đã đóng: ở 8192 chưa từng có ca nào, nhưng nếu ai hạ ngân sách hoặc thêm history thì guard bắt ngay thay vì âm thầm supervise một ID model chưa từng thấy.

Test ghim: `tests/test_training.py:263` `test_collator_never_supervises_a_target_whose_reference_value_truncation_removed` — quét **mọi** ngân sách từ 0..1200 để không phụ thuộc một con số ma thuật (hình học vết cắt không đơn điệu vì catalog bị ghim làm dịch chỗ cắt).

---

## 3. Rủi ro tổng quát hoá trên tool chưa thấy

### 3.1 Quy mô (đo lại trên catalog thật)

| chỉ số | giá trị | nguồn |
|---|---|---|
| tool / app ở `train` | **147 / 46** | `ĐÃ CHẠY` |
| tool / app ở `public_test` | **122 / 36** | `ĐÃ CHẠY` |
| tool **vắng mặt hoàn toàn** ở train | **34** | `ĐÃ CHẠY` |
| app vắng mặt hoàn toàn ở train | **11** | `ĐÃ CHẠY` |

11 app: `ecommerce`, `event_tickets`, `grocery`, `habit_tracker`, `pantry`, `parking`, `public_transit`, `recipes`, `stocks`, `train_bus_tickets`, `translator`.

### 3.2 Ba tầng cấu trúc

Định nghĩa tầng theo *chữ ký* (kind + multiset kiểu + arity + số required):

| tầng | điều kiện | số tool | tên |
|---|---|---|---|
| 1 | có **bản sao cấu trúc chính xác** trong train | **30** | — |
| 2 | cùng kind + multiset kiểu, chỉ **khác arity** | **3** | `parking.reserve`, `stocks.set_price_alert`, `train_bus_tickets.search` |
| 3 | chữ ký kiểu **hoàn toàn mới** | **1** | `stocks.place_order` |

`ĐÃ CHẠY` (`/tmp/unseen_check.py`). Diễn giải: 30/34 tool lạ có "anh em sinh đôi" về hình dạng trong train, nên *cú pháp* tham số không phải ẩn số; cái chưa học được là **ngữ nghĩa** (khi nào gọi tool nào) và giá trị enum của app mới. Chỉ `stocks.place_order` là ca thật sự không có tiền lệ hình dạng.

### 3.3 Phơi nhiễm và render

| chỉ số | giá trị | nguồn |
|---|---|---|
| mẫu test cài ≥1 app chưa thấy | **410 / 634** (64,7%) | `ĐÃ CHẠY` |
| khối schema tool-chưa-thấy được render (step 0, cả 634 mẫu) | **2509** | `ĐÃ CHẠY` |
| trong đó có dòng `parameters:` đầy đủ | **2509 / 2509** | `ĐÃ CHẠY` |

Không có hiện tượng "tool lạ được render sơ sài": mọi khối đều mang schema đầy đủ, đúng như app đã cài.

### 3.4 Tính xác định và tính nhất quán của prompt

| kiểm tra | kết quả | nguồn |
|---|---|---|
| catalog trong prompt sắp xếp **alphabet** (app và tool) | **0 / 3305** prompt lệch thứ tự | `ĐÃ CHẠY` `/tmp/order.py` |
| catalog là **theo mẫu** (app đã cài + `sys.ask_missing_info`) | đúng: `ToolRegistry(catalog).for_sample(s)`; 0 entry của app chưa cài | `ĐỌC TỪ FILE` `tool_registry.py:47-66` + test `test_rendered_prompt_has_no_uninstalled_tools` |
| prompt lưu trong dataset == prompt dựng lại từ data builder (step 0) | **1534 / 1534**, 0 lệch | `ĐÃ CHẠY` |
| `render_prompt_text` (train path) == `HFGenerator.build_prompt` (infer path) | **0 lệch / 1534** — so byte trên **toàn bộ** 1534 message list step-0 lấy từ artifact, cùng tokenizer thật | `ĐÃ CHẠY` `/tmp/parity_full.py` |

Hai phép đo khác nhau, không được trộn:

1. **Prompt lưu == prompt builder dựng lại** — `1534/1534` prompt-body khớp `build_decision_prompt` (`ĐÃ CHẠY` `/tmp/path_identity.py`).
2. **Chat-template hai phía** — `0/1534` byte lệch giữa `render_prompt_text` (train) và `HFGenerator.build_prompt` (infer), đo trên **toàn bộ** 1534 step-0 (`ĐÃ CHẠY` `/tmp/parity_full.py`).

Con số đúng cho (2) là **0/1534**, không phải bất kỳ mẫu số con nào: một khẳng định, một mẫu số.

### 3.5 Có rò rỉ đáp án trong prompt không?

| phép đo | kết quả |
|---|---|
| tên tool đích xuất hiện **ngoài** khối của chính nó | **0 / 3305** (sau khi tách khối, đếm riêng) — `ĐÃ CHẠY` |
| mỗi prompt có đúng **một** khối của tool đích | **3305 / 3305** — `ĐÃ CHẠY` |
| độ "nổi bật" vị trí: rank chuẩn hoá của khối đích trong catalog | **0,4434** trung bình (null đều = 0,5) — `ĐÃ CHẠY` |

**CHƯA ĐO:** `0,4643` trong brief không tái tạo được — thử 4 định nghĩa rank (`i/(n-1)`, `i/n`, `(i+0,5)/n`, rank trong cùng kind) được 0,4434 / 0,4296 / 0,4449 / 0,4588; báo cáo dùng **0,4434**. Với `public_test`, rank của tool-chưa-thấy là **0,5672** (2509 khối / 634 prompt) — app mới nằm *sau* trung vị, không được ưu tiên; rank toàn bộ tool trên test = 0,5000 (20072 khối).

### 3.6 Ablation: văn bản prompt có tự xác định đáp án?

Ghim 20 decision point thật, **đổi tên tool đích** trong prompt (chỉ header `### <name> (kind=`, giữ mọi thứ khác): **20/20** vẫn xác định đầy đủ tham số từ schema còn lại **và** tool vẫn duy nhất theo (kind + schema + mô tả); 0 ca mất tham số, 0 ca nhập nhằng. `ĐÃ CHẠY` (`/tmp/ablate2.py`). Độc lập: trên cả 3305 target, (kind + tên tham số + kiểu) cho **2594/3305 = 78,5%** duy nhất; thêm mô tả → **3305/3305 = 100%** (`ĐÃ CHẠY` `/tmp/amb3.py`).

Cách đọc đúng: **văn bản prompt tự nó xác định đáp án**. Đây là hệ quả của việc catalog phải nằm trong prompt (model phải đọc schema để sinh call hợp lệ), **không** phải rò rỉ nhãn. Rủi ro thật: model có thể học "chọn theo mô tả" và đạt điểm cao mà không học được *quyết định* khi nhiều tool cùng khớp. Model có đi theo đường đó không — **CHƯA ĐO**, không đo được khi chưa có trọng số.

---

## 4. Docker và seam lịch sử — 419/634 mẫu

### 4.1 Vấn đề gốc

`public_test` chỉ có **audio** cho lượt user trong history (1299 lượt trên 419 mẫu, **0/1299** có text trong dữ liệu gốc). `train` có `audio_text` cho **3491/3491** lượt. Nếu container không mount file transcript lịch sử, agent nhìn thấy 1299 lượt user **rỗng** — prompt suy giảm âm thầm, không có gì trong submission nói ra.

### 4.2 Đã sửa: hai lớp

**Lớp 1 — file transcript.** `outputs/asr/history_transcripts.jsonl`: **4790 dòng** (3491 train + 1299 public_test), **0** `error`, **0** text rỗng, **1437** khoá `(split,id)`, turn_index tăng dần và liên tục từ 1 cho **cả 1437** khoá (`ĐÃ CHẠY`). `apply_history_texts` trên `public_test`: `filled=1299, existing=0, missing=0, samples_with_history=419, samples_touched=419` (`ĐÃ CHẠY`). Trên `train` là no-op vì text vàng luôn thắng (`test_history_texts_never_overwrite_gold_audio_text`).

**Lớp 2 — fail cứng.** `scripts/infer.py:578-583`: khối bắt đầu ở `if blank and not getattr(args, "allow_missing_history", …)` (dòng 578): nếu số lượt user sẽ render rỗng > 0 và **không** có `--allow-missing-history`, in `ERROR: … would render N history user turn(s) blank` và `return EXIT_DATASET` (**2**) — **trước khi nạp trọng số**. Cờ `--allow-missing-history` (`infer.py:130-135`) được ghi rõ là `DIAGNOSTIC ONLY`.

### 4.3 Ba kịch bản container

| kịch bản | hành vi | nhãn |
|---|---|---|
| mount có mặt + `HISTORY_TRANSCRIPTS` set | `docker/entrypoint.sh:67-68` thêm `--history-transcripts`; **1299/1299** lượt được điền | `ĐÃ CHẠY` (replicate argv của entrypoint trong layout tạm `/tmp/ctr/{data,history,outputs}`) |
| set nhưng **không đọc được** | `entrypoint.sh:69-71` → `ERROR: HISTORY_TRANSCRIPTS set but not readable: …`, `exit 2` | `ĐỌC TỪ FILE` + test `test_entrypoint_wires_history_transcripts` |
| **không** set | `infer.py` tính `blank` = 1299 → `ERROR: no --history-transcripts file was configured; split=public_test would render 1299 history user turn(s) blank`, exit 2 | `ĐÃ CHẠY` (chạy lại argv) + `tests/test_inference.py:1162` |

Tài liệu hoá: `docker/README.md:65` (bảng env var, nêu rõ "419 of 634"), `:81` (mount thứ 4 read-only), `:95-104` (blank history là hard failure); `docs/INFERENCE_GUIDE.md:86-87`.

### 4.4 Không có đường dẫn tuyệt đối của host

| file | kết quả |
|---|---|
| `docker/Dockerfile` | 0 đường dẫn host-absolute trong lệnh runtime |
| `docker/entrypoint.sh` | 0 |
| `cloud/sandbox_runner.py` | 0 |
| `configs/inference.yaml` | 0 |

`ĐỌC TỪ FILE`. Loading history thuần `json` + `pathlib`, **không** network (`src/lifemate/data/history.py`).

### 4.5 Docker thật

**NOT RUN.** `which docker` → không có binary; `docker info` → 127. Vì vậy mọi tuyên bố "container chạy được" ở đây là *tái lập argv trong layout container-hoá*, không phải `docker run`. Không có cách nào đo thời gian build image ở đây.

Pin CUDA đã được sửa sang `sm_120` (§6.3), nhưng image vẫn **chưa từng được build và chưa từng được chạy**: không có daemon trên máy này, và pin `cu124` trước đó thì **không thể** chạy `sm_120` dù có daemon. Bản sửa pin là **UNVERIFIED** cho tới khi có người thật sự build image.

---

## 5. ASR — lỗi số và tác động lên hành động

### 5.1 Chất lượng nền (1545 request, Gipformer-68M int8)

| chỉ số | giá trị | nguồn |
|---|---|---|
| n | 1545 | `ĐỌC TỪ FILE` `/tmp/asr_final.jsonl.metrics.json` |
| WER chuẩn hoá | **17,91%** (0,17909) | `ĐỌC TỪ FILE` |
| CER chuẩn hoá | **11,13%** (0,11128) | `ĐỌC TỪ FILE` |
| sentence-exact | **8,74%** (0,08738) | `ĐỌC TỪ FILE` |
| WER/CER raw (case-insensitive) | 29,80% / 14,95% | `ĐỌC TỪ FILE` |
| empty hypothesis | 0 | `ĐỌC TỪ FILE` |

Khớp baseline đã chấp nhận; đây là chất lượng **của ASR**, không phải của model.

### 5.2 Span số: phân bố lỗi

**885/1545** request có chứa span số (`ĐÃ CHẠY` `/tmp/numan/s.txt`). **110 span sai trong 101 câu** = **6,54%** request.

| loại | câu tham chiếu | span sai | câu sai | tỉ lệ câu sai |
|---|---|---|---|---|
| money | 137 | 10 | 9 | **6,6%** |
| cardinal (số đếm) | 494 | 62 | 57 | **11,5%** |
| date | 58 | 4 | 4 | 6,9% |
| time | 380 | 22 | 22 | **5,8%** |
| phone | 17 | 5 | 5 | **29,4%** |
| address | 4 | 1 | 1 | **25,0%** |
| other-digit | 17 | 6 | 6 | **35,3%** |

`ĐÃ CHẠY`. History (3491 lượt của train): 1158 lượt có span số, **158 span sai / 138 lượt** = 4,0%.

Nhận xét định lượng quan trọng: **số điện thoại là lớp tệ nhất tính theo tỉ lệ** (29,4%) nhưng chỉ có 17 câu; **cardinal đóng góp nhiều lỗi tuyệt đối nhất** (62/110 span). `time` sai 22 lần nhưng **0** lần material (§5.3) — vì ASR vẫn giữ đúng "giờ/phút" và sai phần đuôi mà lệnh không dùng, hoặc model đã có sẵn mốc thời gian trong `context.current_time`.

### 5.3 Tác động lên hành động (thay vào gold call, validate bằng catalog chính thức)

| tập | span material | mẫu material | tỉ lệ | span cosmetic |
|---|---|---|---|---|
| **request** (1545) | **15** | **13** | **0,84%** | 95 |
| **history** (3491 lượt) | **20** | **20** | **0,57%** | 138 |

`ĐÃ CHẠY` (`/tmp/numan/final.json`, `table2.txt`). **Lệch brief:** brief ghi history 19 span / 0,54%; đo được **20 / 0,57%** — chênh đúng ca `cardinal_quantity|material_schema_invalid`. Báo cáo dùng 20.

Phân rã 15 span material của request: cardinal `different_value` 6, cardinal `absent` 3, phone `different_value` 5, money `different_value` 1.

**Hỏng schema do mất số** (validator chính thức trên gold call đã thay giá trị ASR):

| ca | tool.arg | required? | kết quả |
|---|---|---|---|
| `train_1271` | `home_energy.set_alert.threshold_kwh` | **có** | **KHÔNG hợp lệ** — `thiếu tham số bắt buộc: threshold_kwh` |
| `train_0199`, `train_0536`, `train_0236/u5`, `train_0628/u1`, `train_1130/u2`, `train_1698/u5` | `climate_set.temperature`, `shopping_list.add.quantity`, `calendar.list_events.date_from#month` | không | hợp lệ |

`ĐÃ CHẠY` (`/tmp/numan/absent.txt`). Đo được **1** ca làm call không hợp lệ, không phải 2. Con số "2 of 35" trong brief: **CHƯA ĐO** — ca thứ hai không tái tạo được (`train_0515/u1` đổi ngày 29→95 vẫn parse được).

**Vì sao money phần lớn cosmetic:** ASR giữ từ chỉ độ lớn (`nghìn`/`triệu`/`tỷ`), chỉ sai đuôi chữ số; gold call thường số tròn nên khớp sau chuẩn hoá, hoặc tham số là free-text (`note`).

### 5.4 Có nên sửa ASR không? Đo, không phán đoán

| biện pháp | lợi | hại | kết luận |
|---|---|---|---|
| **Phone-length gate** (giá trị lớp phone phải dài 10-11 chữ số) | bắt **8/12** span phone material (histogram độ dài giá trị ASR: `{10:3, 9:4, 8:2, 13:2, 11:1}`) | trên **reference**: 17 span phone có độ dài `{11:14, 8:1, 6:2}` → gate theo độ dài sẽ **cờ oan 3** span vàng | **cân nhắc**; nếu dùng thì chỉ dùng làm *cảnh báo*, không tự sửa |
| **Ghép chữ số đọc rời** (concat rule) | material gain **0** (15 → 15 span) | rule bắn trên **75** mention tham chiếu, chỉ **27** được gold/text ủng hộ → **49% không có cơ sở** (37/75) | **từ chối**, có số |
| Sửa gì khác | — | — | **không làm** |

`ĐÃ CHẠY` (`/tmp/numan/gates.py`, `/tmp/numan/rulemeasure.py`, `/tmp/numan/concatfp.py`).

**CHƯA ĐO:** brief nêu phone gate có "0 false positive trên reference" và concat rule "49% FP". Vế 49% khớp chính xác; vế "0 FP" thì **không** khớp phép đo độ dài (3 reference có 8/6 chữ số). Báo cáo ghi 3 thay vì 0.

### 5.5 Kết luận ASR

| tuyên bố | căn cứ |
|---|---|
| ASR **không** làm hỏng dữ liệu SFT | 4385→**4368** record (mất 17 decision point vì observability, **không** vì ASR); target trong `train.jsonl` là **gold transcript nguyên văn** |
| ASR **không** làm hỏng đường thi | 0,84% request bị ảnh hưởng material << 17,91% WER; 1 ca hỏng schema / 1545 |
| ASR **có** ảnh hưởng thật, cục bộ | 5 ca phone + 1 money + 9 cardinal: model nhận số sai và **không có cách tự phát hiện** — rủi ro chấp nhận, không phải đã triệt tiêu |
| Không được nói | "ASR sạch", "0,84% không đáng kể" — 13 mẫu là 13 lần mất điểm có thật |

---

## 6. Qwen3.5-4B trên CUDA — trạng thái máy

### 6.1 Phần cứng

| hạng mục | giá trị | nguồn |
|---|---|---|
| GPU | `NVIDIA GeForce RTX 5070 Ti` | `ĐÃ CHẠY` `nvidia-smi` |
| VRAM | **16303 MiB** = 17,09 GB decimal / 15,92 GiB | `ĐÃ CHẠY` |
| driver | **580.173.02**, CUDA API **13.0** | `ĐÃ CHẠY` |
| compute capability | **(12, 0)** = `sm_120`, Blackwell | `ĐÃ CHẠY` |

### 6.1b VRAM trống là **đại lượng biến động**, không phải một con số

Bốn lần đo trên cùng card, cùng driver, trong cùng ngày. Hai lần cuối cách nhau **78 giây** và free đã rơi từ 7,72 GB xuống 3,03 GB, cùng một pid nhưng bộ nhớ của nó tăng 7582 → 12062 MiB. Cột `used` của hàng audit là `16303 − 11276` (số `used` không được ghi lại trong artifact audit); các hàng còn lại đọc trực tiếp từ `nvidia-smi`.

| thời điểm đo | used | free (decimal) | process ngoài | so với 11,49 GB của LoRA | nguồn |
|---|---|---|---|---|---|
| audit trước (trong `docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md` §E.5) | 5027 MiB | **11,82 GB** (11276 MiB / 11,01 GiB) | không ghi lại | 97,2% — `tight`, không đủ biên | `ĐÃ CHẠY`, artifact audit trước |
| lúc viết §6/§8 (bản trước của tài liệu này) | 852 MiB | **15,68 GB** (14955 MiB / 14,60 GiB) | **không có** (`--query-compute-apps` rỗng) | 73,3% — `fits` | `ĐÃ CHẠY` |
| **2026-11-08T15:07:40+07:00** | **8442 MiB** | **7,72 GB** (7365 MiB / 7,19 GiB) | **pid 200020** `.../envs/ilearn39-adapted/bin/python`, **7582 MiB** | **148,8% — KHÔNG vừa** | `ĐÃ CHẠY` |
| **2026-11-08T15:08:58+07:00** (78 giây sau) | **12921 MiB** | **3,03 GB** (2886 MiB / 2,82 GiB) | cùng **pid 200020**, đã tăng lên **12062 MiB** | **379,7% — KHÔNG vừa** | `ĐÃ CHẠY` |

Lệnh đo (hai lệnh, chạy ngay trước khi launch):

```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

**Vì sao ghi cả bảng chứ không ghi một số:** "trống bao nhiêu" không phải thuộc tính của máy, nó là thuộc tính của **thời điểm**. Đã quan sát dao động **15,68 → 11,82 → 7,72 → 3,03 GB** trong cùng một ngày trên cùng một card — biên độ **12,65 GB**, lớn hơn cả mức 11,49 GB mà run cần. Hai mốc cuối chỉ cách nhau 78 giây, nên ngay cả "đo ngay trước khi chạy" cũng có thể sai nếu có độ trễ giữa lệnh đo và lệnh launch. Bất kỳ con số đơn lẻ nào in ở đây cũng sẽ sai trong vòng một giờ; đó là lý do phần này là một bảng có timestamp chứ không phải một câu khẳng định.

**Hệ quả cho verdict:** 11,49 GB là **nhu cầu** của run (số đo được, không đổi). "Có vừa hay không" là **hàm của thời điểm**. Câu hành động đúng: đo ngay trước khi launch; nếu free < 12 GB thì chờ tenant thoát rồi đo lại — **không** hạ `per_device_train_batch_size` xuống dưới 1 (đã là 1) và **không** tắt `gradient_checkpointing` (tắt thì ước lượng ~35,47 GB, §8.4).

### 6.2 torch: ba môi trường, ba trạng thái

| môi trường | torch | CUDA khả dụng | nguồn |
|---|---|---|---|
| system `python3` | không có | — | `ĐÃ CHẠY` |
| `.venv` repo | không có | — | `ĐÃ CHẠY` |
| ~~`/tmp/torchenv`~~ (đã xoá 2026-11-13) | `2.15.0+cpu` — **lịch sử**, đường dẫn không còn tồn tại | `False`, `device_count 0` (đo lúc còn tồn tại) | `ĐÃ CHẠY` (trước khi xoá) |
| **`/tmp/cudaenv`** | **`2.15.0+cu132`** | **True** | `ĐÃ CHẠY` |

`/tmp/cudaenv` được tạo **offline** từ uv cache (wheel `torch 2.15.0+cu132` đã có sẵn, không tải mạng). Đo (`ĐÃ CHẠY`): `torch 2.15.0+cu132`, `cuda 13.2`, `is_available() True`, `cap (12, 0)`, `bf16_supported True`, `arch list ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`, bf16 matmul + backward **finite**, `mem_get_info (15237120000, 16573923328)`. Kèm `huggingface_hub 2.4.0`, `peft 0.21.2`, `transformers 5.22.2`. Tức **toàn bộ stack GPU sẵn sàng ở `/tmp/cudaenv/bin/python`** — thiếu duy nhất trọng số.

### 6.3 Dockerfile pin — đã sửa sang `sm_120`, image **chưa từng build, chưa từng chạy**

Bản trước ghim `docker/Dockerfile` vào `cu124`. `cu124` **không có kernel `sm_120`**: image build ra **chạy được CPU** nhưng **không chạy được GPU** trên Blackwell — `torch.cuda.is_available()` sẽ False hoặc kernel launch fail. Đây là lỗi cấu hình im lặng: build thành công, chỉ chết lúc huấn luyện. Vì vậy image cũ **không thể** chạy trên máy này dù có daemon.

Đã sửa (tối thiểu, `ĐỌC TỪ FILE` `docker/Dockerfile:3-12,31,51-53`):

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS build     # was 12.4.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128          # was .../whl/cu124
# `torch>=2.4` giữ nguyên; escape hatch CPU vẫn còn:
# --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
```

Comment block phía trên pin nay giải thích `sm_120`/Blackwell và nói rõ `cu124` không có kernel đó, thay cho câu "matches modern drivers" trước đây. `ubuntu22.04` giữ nguyên (glibc 2.35 vẫn thoả manylinux_2_28 của wheel torch/transformers).

**Trạng thái: UNVERIFIED.** Máy này **không có docker** (`which docker` → 127, `docker info` → 127), nên image **chưa từng được build và chưa từng được chạy** — bản sửa chỉ là sửa pin, chưa có phép đo nào xác nhận. Khi có daemon, phép kiểm phải là `docker run --rm --gpus all lifemate:local python -c "import torch;print(torch.cuda.get_arch_list())"` và arch list phải chứa `sm_120`. Kích thước image và thời gian build: **CHƯA ĐO**.

### 6.4 Tương thích model (meta device, không có byte trọng số)

| hạng mục | giá trị | nguồn |
|---|---|---|
| class khi load qua loader của repo | `Qwen3_5ForCausalLM` | `ĐÃ CHẠY` |
| tham số | **4.205.751.296** / **426** tensor | `ĐÃ CHẠY` |
| `state_dict()` | 4.841.450.496 / 427 tensor | `ĐÃ CHẠY` |
| chênh | 635.699.200 = 248320 × 2560 (tensor tied đếm hai lần); `lm_head.weight is embed_tokens.weight` → **True** | `ĐÃ CHẠY` |
| layer | **32** = 24 `Qwen3_5GatedDeltaNet` + 8 `Qwen3_5Attention` | `ĐÃ CHẠY` |
| config | hidden 2560, vocab 248320, `tie_word_embeddings: true`, `architectures: ["Qwen3_5ForConditionalGeneration"]` | `ĐỌC TỪ FILE` `config.json` |
| index trọng số | **738 tensor**, **2 shard** (`00001-of-00002`: 87, `00002-of-00002`: 651), `total_size` 9.319.737.856 | `ĐỌC TỪ FILE` `model.safetensors.index.json` |

**Chat template.** Với `enable_thinking=False`, template Qwen3.5 phát ra think block **đã đóng**:

```
<|im_start|>assistant\n\n</think>\n\n
```

`ĐÃ CHẠY` (byte-verified: `train tail == infer tail`, `byte-identical: True`; `prompt ends with closed think == True`). Header `### <tool>` **không** bị ảnh hưởng. Nếu cờ này thiếu, template rơi vào nhánh `else` và in `…assistant\n\n` (think **mở**) — action sẽ được supervise *bên trong* reasoning và ở suy luận model không có tín hiệu dừng. Đây là defect D-24 (§10), đã sửa và ghim bằng test.

---

## 7. Full Fine-Tuning — trạng thái

### 7.1 Cấu hình (đang là DEFAULT)

| tham số | giá trị | nguồn |
|---|---|---|
| `tuning_mode` | **`full`** | `configs/base.yaml:24`, `configs/training_full.yaml:7` |
| `learning_rate` | 5,0e-6 | `ĐỌC TỪ FILE` |
| microbatch / accum | 1 / 8 → **effective batch 8** | `ĐỌC TỪ FILE` |
| `max_seq_length` | **8192** | `ĐỌC TỪ FILE` |
| `gradient_checkpointing` | **true** (load-bearing) | `ĐỌC TỪ FILE` |
| epoch | 2 | `ĐỌC TỪ FILE` |
| optimizer / clip / seed | `adamw_torch` / 1,0 / 42 | `ĐỌC TỪ FILE` |
| precision | bf16 compute + **fp32 master** | `ĐỌC TỪ FILE` `full_finetuning.master_dtype` |
| `configure_full_finetuning` | unfreeze toàn bộ, **raise** nếu `trainable != total`; không fallback im lặng sang LoRA | `ĐỌC TỪ FILE` |

### 7.2 Ước lượng bộ nhớ (công thức của repo, không phải phép đo cấp phát)

`ĐÃ CHẠY` `run_memory_preflight(model=None, param_count=4205751296, cfg=configs/training_full.yaml, seq_len=8192)`:

| hạng mục | GB |
|---|---|
| weights (fp32 master) | **16,82** |
| gradients (fp32) | **16,82** |
| optimizer (AdamW, 8 B/tham số) | **33,65** |
| activations (k=2, có checkpointing, 32 layer × 8192 × 1 × 2560) | **2,68** |
| **TOTAL** | **69,98** |
| checkpoint | 33,65 |
| verdict | **`does_not_fit`** |

### 7.3 Phán quyết

| mốc | so sánh | kết luận |
|---|---|---|
| card 17,09 GB | 69,98 / 17,09 = **4,09×** | không vừa |
| bất kỳ card đơn < 80 GB | — | không vừa (cần ≥70 GB chỉ cho trạng thái ổn định) |
| tắt gradient checkpointing | activations ~10× → ~13,4 GB, total ~80,7 GB | càng không |

`SUY RA` từ các số trên. **NOT RUN** — chưa có optimizer step nào trên model thật, và sẽ không có trên máy này.

**Đường duy nhất cho full FT:** ≥80 GB (A100/H100 80 GB), hoặc sharding 2×40 GB với ZeRO-3 (khối `deepspeed` trong `training_full.yaml` đang **comment**, và trainer **chưa** tiêu thụ nó — `ĐỌC TỪ FILE`). Không được trình bày full FT là "sắp chạy được".

---

## 8. LoRA — trạng thái

### 8.1 Phát hiện target module (đường code thật, meta device)

`configs/training_lora.yaml` đặt `target_modules: null` → `discover_target_modules` chọn theo kiến trúc. Chạy qua `configure_lora` trên model meta-device dựng từ checkpoint phát hành:

| hạng mục | giá trị | nguồn |
|---|---|---|
| ứng viên | 17 suffix = 9 `ATTENTION_MLP_SUFFIXES` + 8 `HYBRID_SUFFIXES` | `ĐỌC TỪ FILE` `lora_finetuning.py:29-42` |
| **suffix resolve được** | **12**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | `ĐÃ CHẠY` |
| module bị wrap | **248** (`down` 32, `gate` 32, `up` 32, `in_proj_*` 24 each, `out_proj` 24, `q/k/v/o_proj` 8 each) | `ĐÃ CHẠY` |
| **trainable** | **32.464.896** (r=16, alpha=32, dropout=0,05, bias none) | `ĐÃ CHẠY` |
| đối chiếu peft | `peft.num_parameters(only_trainable=True)` = **32.464.896** | `ĐÃ CHẠY` |
| đối chiếu giải tích | `Σ r × (in + out)` trên 248 module = **32.464.896**, khớp cả report lẫn peft | `ĐÃ CHẠY` |
| adapter tensor | 496 | `ĐÃ CHẠY` |
| tổng tham số sau wrap | 4.238.216.192 | `ĐÃ CHẠY` |

`127` candidate suffix không tồn tại (`qkv_proj`, `gate_up_proj`, `in_proj`, `x_proj`, `dt_proj`) — đúng, vì kiến trúc này tách rời chúng.

### 8.2 Sửa lỗi "21.028.756"

Con số cũ **21.028.756** đúng bằng `int(4.205.751.296 × 0,005)` — **placeholder 0,5%** của preflight, bị trích dẫn như số đo. Nó **không** phải số adapter thật. Chênh: 32.464.896 / 21.028.756 = **1,544×** (thiếu 11,4 M tham số).

Đã sửa: `src/lifemate/training/memory_preflight.py:75-89` `_trainable_from_model(model)` đếm tensor có `requires_grad`; `:161-166` chọn `trainable_source ∈ {"measured", "placeholder_0.5pct"}` và **in ra nguồn**. Đo lại:

```
tuning_mode = lora
trainable_params = 32464896
trainable_source = measured
```

`ĐÃ CHẠY`. Test ghim: `tests/test_training.py` (`test_memory_preflight_math_is_exact` + các test LoRA); bất kỳ báo cáo nào in `trainable_params` mà không kèm `trainable_source` đều là thiếu bằng chứng.

### 8.3 Ước lượng bộ nhớ (đường đo được)

`ĐÃ CHẠY` cùng preflight, `seq_len=8192`, `gradient_checkpointing=true`:

| hạng mục | GB | ghi chú |
|---|---|---|
| weights (bf16, đóng băng) | **8,48** | 73,7% của tổng |
| gradients (chỉ adapter) | **0,06** | 32,46 M × 2 B |
| optimizer (chỉ adapter) | **0,26** | 32,46 M × 8 B |
| activations | **2,68** | k=2, giống full |
| **TOTAL** | **11,49** | |
| checkpoint | 8,54 | |
| verdict | **phụ thuộc thời điểm** — xem §6.1b | |

**11,49 GB là nhu cầu, không phải verdict.** "Có vừa hay không" là hàm của VRAM trống tại lúc launch; VRAM trống đã dao động **15,68 / 11,82 / 7,72 / 3,03 GB** trong cùng một ngày, hai mốc cuối cách nhau 78 giây (§6.1b):

| mốc trống | 11,49 GB chiếm | kết luận |
|---|---|---|
| **15,68 GB** (852 MiB used, 0 process ngoài, đo lúc viết §6) | **73,3%**, headroom 4,19 GB | **vừa** (`fits`, ngưỡng `tight` 90%) |
| **11,82 GB** (audit trước, tenant ngoài) | **97,2%** | `tight` — vừa về số học nhưng **không đủ biên**; một tenant lớn hơn là fail |
| **7,72 GB** (15:07:40, pid 200020 `ilearn39` giữ 7582 MiB) | **148,8%** | **KHÔNG vừa** — thiếu 3,77 GB |
| **3,03 GB** (15:08:58, cùng pid, đã tăng lên 12062 MiB) | **379,7%** | **KHÔNG vừa** — thiếu 8,46 GB |

Phân biệt **11,30 vs 11,49**: comment trong `configs/training_lora.yaml` ghi "1 keeps 11,30 GB" là ước lượng **trước** khi sửa `trainable_source` (dùng 21,03 M adapter, grads 0,04 + optim 0,17 = 0,21 GB). Số đo được hiện tại là **11,49 GB**. Chênh **0,19 GB**; phần riêng do adapter là **0,11 GB = 1,0%** → **verdict không đổi** dù đi đường nào (`SUY RA`: 0,32 − 0,21 = 0,11).

### 8.4 Độ nhạy

| biến | tác động | verdict |
|---|---|---|
| microbatch 2 | ~13,99 GB | `tight` ở mốc trống cao nhất từng đo (89,2% của 15,68); `does_not_fit` ở ba mốc còn lại — **không** nên |
| tắt gradient checkpointing | ~35,47 GB | không vừa (checkpointing load-bearing) |
| `r` 16 → 32 | +0,26 GB | không đổi verdict (chênh 0,26 GB so với biên 3,77 GB đang thiếu) |
| tenant ngoài chiếm 3,6 GB | trống còn 11,82 GB → 97,2% | `tight`, không đủ biên |
| tenant ngoài chiếm 7,6 GB (đo được) | trống còn 7,72 GB → 148,8% | **không vừa** |
| tenant ngoài chiếm 12,1 GB (đo được, 78 giây sau) | trống còn 3,03 GB → 379,7% | **không vừa** |
| `max_seq_length` → 4096 | ~10,1 GB | vừa, nhưng mất 846/3305 khối schema (§2.3) — **không được hạ** |

### 8.5 Những gì LoRA **chưa** chứng minh

| tuyên bố | trạng thái |
|---|---|
| adapter đúng chỗ (248 module, 12 suffix) | **ĐÃ ĐO** trên kiến trúc thật |
| VRAM thật khi train | **CHƯA ĐO** — chưa có `torch.cuda.max_memory_allocated()` |
| loss giảm / hội tụ | **CHƯA ĐO** — 0 bước trên model thật |
| chất lượng sau SFT | **CHƯA ĐO** — không checkpoint |
| tốc độ (giây/bước) | **CHƯA ĐO** |

### 8.6 Ngân sách bước

4368 record / microbatch 1 / accum 8 → **546** bước-optimizer mỗi epoch (`SUY RA`); 3 epoch → **1638** bước. Thời gian **CHƯA ĐO** (không có throughput nào trên GPU này).

### 8.7 Độ trễ suy luận — phần đo được và phần bị chặn

Đo trên **30 mẫu `public_test` phân tầng** (Gipformer thật, tokenizer thật `Qwen/Qwen3.5-4B`, `ToolEnv` thật, không mock). Máy chịu tải cao (`loadavg1` 28-33) nên các số dưới đây là **cận trên** của máy này.

| thành phần | median / mẫu | nguồn |
|---|---|---|
| ASR request cuối | **71,3-101,7 ms** tuỳ pass (5 pass: 100,79 / 83,60 / 71,30 / 94,82 / 90,48); median của median-từng-mẫu **110,3 ms**; `asr_final` đo riêng **128,7 ms** | `ĐÃ CHẠY` `/tmp/lat/m5a.json`, `m1.json`, `asr_rows.json` |
| load audio | **1,35 ms** | `ĐÃ CHẠY` `m3.json` |
| history load + apply (amortized) | **0,10 ms** lần đầu; bulk 634 mẫu **0,0044 ms/mẫu** | `ĐÃ CHẠY` `m2.json` |
| build prompt ×3 | 0,231 + 0,205 + 0,201 = **0,64 ms** | `ĐÃ CHẠY` `m1.json` |
| chat-template + encode ×3 | 4,19 + 4,21 (+ step2) ≈ **12,6 ms** | `ĐÃ CHẠY` `m5b.json`, `m2.json` |
| parse + validate ×3 | **0,006-0,008 ms**/lần | `ĐÃ CHẠY` `m4.json` |
| env session build | **0,71 ms** | `ĐÃ CHẠY` `m4.json` |
| env call (read/write) | 0,036 / 0,016 ms | `ĐÃ CHẠY` `m4.json` |
| state hash | **0,155 ms** | `ĐÃ CHẠY` |
| **subtotal đo được** | **~125-135 ms** | `SUY RA` từ các dòng trên |
| ASR decoder init | 0,84-1,20 s, **một lần mỗi process** | `ĐÃ CHẠY` (`m4`: 841,5 ms; `m1`: 1198,4 ms) |
| tokenizer load | **305-343 ms**, một lần | `ĐÃ CHẠY` `m5b.json`/`m2.json` |
| **LLM prefill + decode** | **BLOCKED** — không có trọng số | — |
| **tổng / mẫu** | **KHÔNG TÍNH ĐƯỢC** | — |

**CHƯA ĐO:** brief nêu "ASR final 102,1 ms = 87% subtotal, subtotal 118 ms". Không tái tạo được 102,1 ms ở bất kỳ file nào trong `/tmp/lat`; đo được 71-129 ms tuỳ cách tổng hợp. Báo cáo **không** dùng 118 ms; dùng khoảng **~125-135 ms** (`SUY RA`), LLM là phần chưa biết.

**Đây là chỗ rủi ro thật:** prompt tăng theo bước — step 0 median **4471** / p95 6421 / max **6665**; step 1 median 4616 / p95 6943,75 / max 7190; delta **320,5 token/bước**; **20/30** prompt step-0 đã > 4096 (`ĐÃ CHẠY` `m2.json`,`m5b.json`). Với `max_steps: 8` và `max_new_tokens: 512`, prompt có thể vượt **8192** ở các bước cuối — ngân sách huấn luyện 8192 không đủ cho suy luận dài, và **chưa có phép đo** (BLOCKED vì không có model).

**Giới hạn thời gian của BTC:** repo **không** ghi. Con số 120 s duy nhất là guard nội bộ (`configs/inference.yaml:20`, `stopping_policy` áp) — **không** phải luật thi. `CHƯA ĐO`.

---

## 9. Test thực sự đã chạy

Số hiện tại — ba interpreter, ba con số **khác nhau có chủ đích**. Đo lại **2026-11-13** với `WANDB_BASE_URL=http://127.0.0.1:9` (sinkhole: không lần chạy nào có thể chạm server wandb thật):

| lệnh | kết quả | thời gian | nguồn |
|---|---|---|---|
| `python3 tests/run_tests.py` | **`passed=479 failed=0 skipped(needs fixtures)=13`**; skip modules `test_tiny_e2e.py`, `test_tiny_generation.py` | 5,22 s | `ĐÃ CHẠY` |
| `.venv/bin/python -m pytest -q` | **`485 passed, 7 skipped`**, exit 0 | 8,96 s | `ĐÃ CHẠY` |
| `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q` | **`491 passed, 1 skipped`**, exit 0 | 17,53 s | `ĐÃ CHẠY` |

Ba bộ là **479 / 485 / 491**, skip **13 / 7 / 1**. Vì sao ba số khác nhau, theo đúng interpreter nào thu cái gì:

- **stdlib (`python3`)**: không có `torch`/`transformers`/`peft` nên **6** test tiny bị skip (`HAVE_DEPS = False` trong `test_tiny_e2e.py` 5 test + `test_tiny_generation.py` 1 test), cộng **7** test trong `test_sandbox_credentials.py` nhận tham số fixture của pytest mà runner stdlib không bind được (`_needs_args` → skip). 6 + 7 = **13**.
- **`.venv` pytest**: `.venv` **không cài `torch`** (kiểm lại: `import torch` → `ModuleNotFoundError`) nên 6 test tiny vẫn skip, cộng `tests/test_wandb_tracking.py` skip vì `wandb` không có trong `.venv` → **7**.
- **`/tmp/cudaenv` pytest**: có `torch 2.15.0+cu132` + `transformers 5.22.2` + `peft 0.21.2` + `wandb 0.31.0`, nên thu **toàn bộ**, chỉ skip **1** test: `tests/test_asr.py:376` (torch có mặt nên nhánh "thiếu package" không thể chạy).

**`/tmp/torchenv` đã bị xoá trong phiên 2026-11-13** (`ls -d /tmp/torchenv` → `No such file or directory`), nên **`/tmp/cudaenv` là interpreter torch của bản ghi**; mọi con số `455`/`473` trong tài liệu này và trong các tài liệu khác là **số lịch sử** đo trên đường dẫn không còn tồn tại. Bộ ba cũ **468 / 468 / 473** cũng là lịch sử: bốn file test đã được thêm trong phiên 2026-11-13 (`test_wandb_tracking`, `test_sandbox_probe`, `test_sandbox_credentials`, `test_sandbox_measured`).

**Không chạy ở chế độ ghi:** `scripts/verify_all.py` **không** được chạy mà không có `--no-write`, vì nó sẽ ghi đè artifact đã có stamp `2026-11-08T13:25:54+0700`.

### 9.1 Gate run trên cây hiện tại

`ĐÃ CHẠY` phiên này: `scripts/verify_all.py --no-write` trên **cây hiện tại** (sau mọi thay đổi của phiên), **gate Q bị loại trừ có chủ đích** — xem ghi chú dưới bảng.

| gate | nội dung | trạng thái | bằng chứng (một dòng) |
|---|---|---|---|
| A | install / pyproject / console scripts | **PASS** | metadata + `--help` của mọi console script parse được |
| B | import package, CLI, mọi module `src/lifemate` | **PASS** | 0 lỗi import trên toàn bộ module |
| C | config mặc định + YAML + unknown-key | **PASS** | `tuning_mode: full`, 3 YAML parse, unknown key bị từ chối |
| D | dataset validation (fixtures) | **PASS** | `inspect_data` exit 0 trên fixtures, exit 2 + đúng thông điệp trên root rỗng |
| E | synthetic EDA | **PASS** | `run_eda` exit 0, report mang banner `SYNTHETIC` |
| F | causal ordering + label leak | **PASS** | `prepare_dataset` trên fixtures, thứ tự nhân quả nghiêm ngặt, 0 leak |
| G | loss masking + gradient anti-leak | **PASS** | stdlib: "test_training.py masking tests pass (gradient anti-leak torch-gated)"; torch: "gradient anti-leak test verified (torch active)" — torch **mạnh hơn**, cả hai PASS |
| H | full FT là mặc định | **PASS** | dry-run preflight báo `tuning_mode: full` |
| I | LoRA vẫn chọn được | **PASS** | in adapter plan |
| J | agent loop nhiều bước + read-before-write + failure recovery | **PASS** | mock env, 3 kịch bản |
| K | missing-info qua meta-tool + out-of-scope qua `finish` | **PASS** | 2 nhánh |
| L | submission end-to-end (`--use-mock-env` + scripted generator + validate) | **PASS** | 0 schema error |
| M | quét tĩnh truy cập bị cấm + runtime guard | **PASS** | 0 vi phạm tĩnh, guard chặn đường dẫn thật |
| N | docs/CLI agreement | **PASS** | **17 docs**, **354 flag** kiểm, **0 lỗi, 0 flag bị nêu**; `--self-test` exit 0 và phân biệt được cả unknown flag lẫn unknown gate letter |
| O | placeholder / `pass` trần / TODO-FIXME (AST) | **PASS** | 0 hit trong `src/lifemate`, `scripts`, `cloud` |
| R | tiny-model CPU e2e | **PASS** | cả hai interpreter PASS; mỗi bên chọn tiny runner riêng, **6 test pass** như nhau |
| P | regression (cả hai runner) | **PASS** | đo lại 2026-11-13: stdlib `python3 tests/run_tests.py`: `passed=479 skipped=13`; `/tmp/cudaenv/bin/python tests/run_tests.py`: `passed=484 skipped=8`; pytest `485 passed, 7 skipped` (`.venv`) / `491 passed, 1 skipped` (`/tmp/cudaenv`). Số gate cũ `468/473` là lịch sử |

Hai lần chạy, cả hai exit 0:

| interpreter | timestamp | tổng |
|---|---|---|
| `python3` (stdlib) | `2026-11-08T15:18:59+0700` | 17 gate chọn (A-O, R, P) — **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 0** |
| `/tmp/torchenv/bin/python` (**lịch sử** — venv đã xoá 2026-11-13; `/tmp/cudaenv` là interpreter torch của bản ghi) | `2026-11-08T15:20:40+0700` | 17 gate chọn (A-O, R, P) — **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 0** |

**Artifact trên đĩa cho lượt chạy này** (bốn file mới; hai lượt 15:18/15:20 ở trên là số kể lại, bốn file dưới đây là bản ghi đóng dấu):

| artifact | `meta.timestamp` | `totals` | sha256 (prefix) |
|---|---|---|---|
| `outputs/verification/verify_all_current_tree_stdlib.json` | `2026-11-08T16:02:16+0700`, python 3.12.3 | **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 0** | `910c21b33f3d6854…` |
| `outputs/verification/verify_all_current_tree_torch.json` | `2026-11-08T16:04:09+0700`, python 3.12.11 | **PASS 17 / FAIL 0 / SKIP 0 / BLOCKED 0** | `5442c0cebee4046c…` |
| `outputs/verification/verify_all_current_tree_stdlib.txt` | bảng render của file stdlib (cùng `totals`) | 17 dòng gate, **tất cả PASS** | `1b8230a1aeaca43a…` |
| `outputs/verification/verify_all_current_tree_torch.txt` | bảng render của file torch (cùng `totals`) | 17 dòng gate, **tất cả PASS** | `982fe52bf3e88fc0…` |

**Vì sao phải có artifact trên đĩa:** lúc **15:54** và **15:56** gate N **FAIL**. Nguyên nhân: một lần sửa tài liệu đã dán `--gates …` — ký tự U+2026 (ellipsis) — vào một bảng, và `check_documented_commands` từ chối nó như một chữ cái gate không tồn tại (`unknown gate letter`). Dòng gây lỗi là `docs/FINAL_REAL_DATASET_AND_TRAINING_AUDIT.md:699`, nay là danh sách 17 chữ cái viết đúng. Trình tự đúng, nói thẳng: tuyên bố "17/17" là **đúng khi đo lần đầu** (15:18:59 / 15:20:40), sau đó **cây bị hồi quy** bởi chính lần sửa tài liệu đó, và hai lượt **16:02:16 / 16:04:09** là **lần chạy lại đã sửa**. Đây chính xác là lý do cần một artifact trên đĩa thay cho con số kể lại.

**Giới hạn của bốn artifact này:** vì gate Q bị loại trừ, footer của cả hai file `.txt` ghi nguyên văn

> `Source: CODE COMPLETE (this repo) / SYNTHETIC TESTED (fixtures) / REAL DATASET TESTED: not probed in this run / OFFICIAL SDK: not probed in this run / REAL MODEL TESTED: no / GPU TRAINING TESTED: not probed in this run / COMPETITION VALIDATED: no`

— nghĩa là bốn file này **chỉ chứng minh 17 gate không-Q**, và **KHÔNG** được dẫn như bằng chứng cho kiểm chứng dữ liệu thật, official SDK hay GPU.

**Gate Q (`blocked_real`) KHÔNG được chạy: operator đã tạm dừng toàn bộ công việc GPU**, nên gate môi trường-thật/GPU **vẫn chưa được đo** trong lần chạy này — tài liệu này **không** khẳng định bất kỳ kiểm chứng GPU nào. Registry có **18 gate tổng cộng; gate Q là gate duy nhất bị loại**, nên 18 − 1 = **17 gate được chọn** ở bảng trên — không gate nào biến mất.

Không phá huỷ artifact: sha256 của `outputs/verification/verify_all.json` **không đổi** (`138cfd4df553e1748d2d3a6043e7ec90643acd655143b62bf52ab3a645fd3f9b`) và `meta.timestamp` của nó vẫn là `2026-11-08T13:25:54+0700`.

### 9.1a Artifact gate cũ — **PRE-CHANGE**, không phải số hiện tại

`outputs/verification/verify_all.json` (18 gate, stamp `2026-11-08T13:25:54+0700`, `regression` ghi `passed=457`) được giữ **chỉ như bản ghi lịch sử**: nó được đóng dấu **trước mọi thay đổi trong §10**, nên các con số của nó — kể cả `passed=457` — **không** được trình bày như trạng thái hiện tại. Trạng thái hiện tại là §9.1 ở trên (17 gate, PASS 17/17, hai interpreter).

### 9.2 Audit "dữ liệu thực sự được tiêu thụ" (không chỉ tồn tại)

`ĐÃ CHẠY` phiên này: nạp **cả 4850** record qua `build_supervised_example` + `SupervisedCollator`:

| lớp kiểm tra | kết quả |
|---|---|
| record kiểm | **4850** |
| thiếu khoá | **0** |
| `len(input_ids) != len(labels)` | **0** |
| hàng all-`-100` (NaN loss) | **0** |
| prompt bị supervise | **0** |
| target bị mask | **0** |
| decode span ≠ target | **0** |
| **tổng token được supervise** | **159.449** |

Batch/padding/seed:

| hạng mục | giá trị | nguồn |
|---|---|---|
| `pad_token_id` | **248044** (tokenizer thật, `padding_side: right`) | `ĐÃ CHẠY` |
| pad label | `-100` | `ĐÃ CHẠY` |
| `pad_to_multiple_of` | 8 (1 hàng 3857 → 4120) | `ĐÃ CHẠY` |
| effective batch | 1 × 8 × 1 = **8** | `ĐỌC TỪ FILE` |
| seed | 42 | `ĐỌC TỪ FILE` |
| val **không** shuffle | `order_source = list(range(len(train_examples)))` chỉ cho train (`trainer.py:442,478`); `evaluate(val_examples)` đi tuần tự | `ĐỌC TỪ FILE` |
| loader phân giải dataset thật | đọc `outputs/datasets/train/train.jsonl`; thiếu → `DatasetNotFoundError` → `EXIT_NO_DATA = 2` (không thay bằng fixture) | `ĐỌC TỪ FILE` `scripts/train.py:98,323` |

---

## 10. Bug đã sửa trong phiên này

| # | bug | file:line | trước | sau | test ghim |
|---|---|---|---|---|---|
| 1 | **Rule observability sai cửa sổ + sai loại** | `src/lifemate/data/trajectory_builder.py:371-467` | `_visible_text` tìm trong **toàn bộ** history; `x-match` bị coi là tham chiếu; bỏ **cả mẫu** | dùng `history_limit` của renderer + **action của các bước trước**; `x-match` loại; bỏ **đúng decision point**; thêm `unobservable_ref_args` + `dropped_decision_points.unobservable_ref_target` | `tests/test_trajectory.py:372` |
| 2 | **Catalog bị cắt giữa** | `src/lifemate/training/collator.py:161,179-193,329-352` | 3/3324 prompt mất khối schema của chính tool đích (cả 3 `contacts.search`; mẫu số 3324 là **lịch sử**, build 4869 record) — nay **0/3305** | `TOOLS_MARKER` + `_split_at_tools_section`: catalog ghim vào tail, chỉ cắt phần trước nó | `tests/test_training.py:218` |
| 3 | **Supervise ID mà prompt không còn chứa** | `src/lifemate/training/collator.py:197-252,355-368` | không có guard (lỗ tiềm ẩn; ở 4096 sẽ là 122 ca) | `_target_values` + `_values_hidden_by_truncation` → `reason="target_value_truncated"` | `tests/test_training.py:263` (quét mọi ngân sách) |
| 4 | **History rỗng là suy giảm âm thầm** | `scripts/infer.py:130-135,577-583` | thiếu mount → 1299 lượt user rỗng, vẫn chạy tiếp | `EXIT_DATASET` **trước khi nạp trọng số** + `--allow-missing-history` (chẩn đoán) | `tests/test_inference.py:1162`, `:1192` |
| 5 | **`NO_THINK_SUFFIX` là chữ trần** | `src/lifemate/training/collator.py:63` | literal đọc thành `"think\n\n\n\n"` — **không có dấu ngoặc nhọn**, không phải think block | `"\x3cthink\x3e\n\n\x3c/think\x3e\n\n"` (escape để ngoặc là **một phần của chuỗi**, không thành markup) | `tests/test_training.py:426` `test_no_think_suffix_is_a_real_closed_think_block` |
| 6 | **Preflight bịa số adapter** | `src/lifemate/training/memory_preflight.py:75-89,161-166,191` | `int(params × 0,005)` = 21.028.756, in ra như số đo | đếm `requires_grad` thật (32.464.896); thêm `trainable_source ∈ {measured, placeholder_0.5pct}` và in kèm | `test_memory_preflight_math_is_exact` + các test LoRA |
| 7 | **Docker/docs seam** | `docker/Dockerfile:122-126`, `docker/entrypoint.sh:35,67-71`, `docker/README.md:65,81,95-104`, `cloud/sandbox_runner.py:121,254,272`, `docs/INFERENCE_GUIDE.md:86-87` | image không `ENTRYPOINT`; mount thứ 4 không được tài liệu hoá; biến set mà không đọc được thì im lặng | thêm `ENTRYPOINT`; bảng env var; `exit 2` khi set-mà-không-đọc-được; mount read-only; note nêu 419/634 | `test_training_command_points_at_prepared_dataset_dir`, `test_spec_has_no_uncovered_container_paths`, `test_entrypoint_wires_history_transcripts` |
| 8 | **Test không cô lập với môi trường** | `tests/test_model_loading.py:73` (`test_is_local_or_offline`) | `is_local_or_offline("Qwen/…") is False` gọi **không** truyền `env={}`, nên kết quả phụ thuộc `HF_HUB_OFFLINE` của caller: dưới `HF_HUB_OFFLINE=1` suite torch báo `472 passed, 1 failed` dù code đúng | truyền `env={}` cho lời gọi đó; hành vi "offline lật đáp án" vẫn được phủ bởi các assertion `env={"HF_HUB_OFFLINE": "1"/"0"}`, `env={"TRANSFORMERS_OFFLINE": "true"}` ngay cạnh | cùng test — nay cả hai ambient setting cho cùng bộ đếm **479 / 485 / 491** (đo 2026-11-13; bộ ba `468/468/473` là lịch sử, đo trên `/tmp/torchenv` đã xoá) |

Điểm chung của #2, #3, #5: **cùng một lớp lỗi** — prompt/model được huấn luyện trên byte khác với byte lúc suy luận, hoặc trên một câu hỏi không trả lời được. #1 và #6 là lớp lỗi thứ hai — **số liệu tự tin nhưng sai nguồn**. #4 và #7 là lớp thứ ba — **suy giảm im lặng**. #8 là lớp thứ tư — **test không cô lập với môi trường**: nó xanh hay đỏ tuỳ biến của caller chứ không tuỳ hành vi của code, nên bộ đếm `473` chỉ đáng tin sau khi sửa.

---

## 11. Blocker còn lại

Chỉ **B1** là blocker cứng (thiếu vật chất, không gỡ được bằng thời gian). **B2** là tranh chấp tài nguyên tạm thời — phải đo tại chỗ, không phải điều kiện đã thoả hay đã hỏng. B3-B11 là các khoảng trống riêng của từng đường (container, cloud, thước đo) không chặn thí nghiệm LoRA cục bộ.

| # | blocker | số đo / trạng thái | ảnh hưởng | gỡ bằng |
|---|---|---|---|---|
| B1 | **Không có trọng số** Qwen3.5-4B | index khai **738 tensor / 2 shard / 9.319.737.856 B = 8,68 GiB**; cache chỉ có `config.json`, index, tokenizer, `chat_template.jinja` | mọi thứ về model: **BLOCKED** | lệnh tải §12 bước 2. **NOT RUN** (policy không cho tải trong phiên này) |
| B2 | **Tranh chấp VRAM tạm thời — phải kiểm tra lúc launch, KHÔNG phải blocker vật lý** | 11,49 GB là nhu cầu cố định; trống đã đo **4** mốc trong ngày: **15,68 / 11,82 / 7,72 / 3,03 GB**, hai mốc cuối cách nhau 78 giây → verdict lật giữa *vừa* và *không vừa* (§6.1b). Tenant ngoài chiếm 3,6-12,1 GB | nếu launch lúc trống < 12 GB, LoRA fail ngay ở step 1 | đo `nvidia-smi` ngay trước launch; nếu free < 12 GB thì chờ tenant thoát rồi đo lại. **Không** gỡ được bằng code; **không** hạ microbatch dưới 1; **không** tắt gradient checkpointing |
| B3 | **Dockerfile từng ghim cu124** (nay đã sửa sang `cu128`/CUDA 12.8.1) | `cu124` không có kernel `sm_120` → image cũ **không thể** chạy GPU trên Blackwell ở mức pin; pin hiện tại nhắm `sm_120` nhưng **chưa build, chưa chạy** nên **UNVERIFIED** | image theo pin cũ build ra **không chạy GPU**; pin mới chưa có bằng chứng chạy | sửa pin (§6.3) — **đã áp dụng vào file**; còn lại phải build + chạy khi có daemon. **NOT RUN / UNVERIFIED** |
| B4 | **Không có docker** | `which docker` → 127; `docker info` → 127 | không build/không chạy container được | cài docker + daemon (ngoài repo). **NOT RUN** |
| B5 | **Không có scorer final-state của BTC** | `evaluator.py:487,663` → `{"available": False, "reason": "official scoring tooling unavailable"}` | không tự chấm được phần `final_state` | chờ BTC phát hành; **không** được điền số thay thế |
| B6 | **`120 s` là guard của ta, không phải của BTC** | `configs/inference.yaml:20` | có thể cắt sớm hoặc muộn so với luật thi | **CHƯA ĐO** — không có tài liệu BTC trong repo; phải hỏi BTC |
| B7 | **Prompt phình theo bước** | step 0 median 4471 / p95 6421 / max 6665 token; +320,5/bước; 20/30 mẫu đã > 4096 | nguy cơ vượt 8192 ở các bước cuối của `max_steps: 8` | **CHƯA ĐO** khi có model; nếu vượt thì phải cắt history hoặc giảm `max_steps` |
| B8 | **Không có phép đo chất lượng nào** | 0 checkpoint, 0 điểm | không biết SFT có tốt lên không | chạy §12 rồi đo loss/val_token_accuracy |
| B9 | **Full FT không vừa máy nào < 80 GB** | 69,98 GB | đường full FT **không khả thi** ở đây | ≥80 GB, hoặc ZeRO-3 2×40 GB (trainer chưa tiêu thụ DeepSpeed) |
| B10 | **`private_test` vắng mặt** | `data/` chỉ có `train`, `public_test` + `README.md` + `VERSION` | không thể kiểm tra trên tập chấm thật | đúng thiết kế (BTC phát sau khi nộp); **không** phải lỗi |
| B11 | **~~Không môi trường nào có cả `torch` CUDA và `sherpa_onnx`~~ — ĐÃ ĐÓNG** | trước: `/tmp/cudaenv` thiếu `sherpa_onnx`; `/tmp/asrenv` thiếu `torch`/`transformers`; `.venv`, `/tmp/torchenv` (đã xoá 2026-11-13) thiếu `sherpa_onnx`. Nay `/tmp/cudaenv` có **cả hai** (`sherpa_onnx 2.0.1` + `torch 2.15.0+cu132`) | không còn — bước 5 và 6 chạy trong **một** môi trường | `uv pip install --python /tmp/cudaenv/bin/python --offline sherpa-onnx` → `+ sherpa-onnx==2.0.1`, `+ sherpa-onnx-core==2.0.1` (`ĐÃ CHẠY`) |

**Không phải blocker, dù dễ bị hiểu nhầm:**

- `public_test` **không có nhãn** (634/634) → **không tồn tại** điểm trên đó. Mọi con số "1,0" trong repo là self-check của evaluator trên gold.
- 19 decision point bị bỏ là **quyết định đúng**, không phải mất mát dữ liệu cần "sửa".
- `visibility_limited` = 0, `n_dropped` (mẫu) = 0.

---

## 12. Lệnh chính xác cho thí nghiệm GPU thật đầu tiên

Chạy theo thứ tự. Mỗi bước ghi rõ trạng thái và điều kiện tiên quyết.

### Bước 1 — kiểm tra môi trường CUDA (`ĐÃ CHẠY`, giữ nguyên)

```bash
cd ~/data/zaloai
/tmp/cudaenv/bin/python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list(), torch.cuda.is_bf16_supported())
print(torch.cuda.mem_get_info())"
```

Kỳ vọng (đã đo): `2.15.0+cu132`, `True`, `(12, 0)`, arch list chứa `sm_120`, bf16 `True`.
Không có `sm_120` → dựng lại: `python3 -m venv /tmp/cudaenv2 && /tmp/cudaenv2/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132` (cần mạng).

### Bước 2 — tải trọng số (`NOT RUN`, cần mạng + ~8,7 GiB)

```bash
cd ~/data/zaloai && mkdir -p models
/tmp/cudaenv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3.5-4B', local_dir='models/Qwen3.5-4B',
      allow_patterns=['*.safetensors','*.json','*.jinja','*.txt','*.model']))"
python3 -c "
import json; i=json.load(open('models/Qwen3.5-4B/model.safetensors.index.json'))
print(len(i['weight_map']), len(set(i['weight_map'].values())), i['metadata']['total_size'])"
```

Kỳ vọng: **738 tensor, 2 shard, 9.319.737.856 B = 8,68 GiB**. Không có `hf`/`huggingface-cli` trên PATH (đã kiểm) nên dùng Python API.

### Bước 3 — smoke test LoRA 5 bước (`NOT RUN`; điều kiện: B1 + B2)

```bash
cd ~/data/zaloai
HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/train.py \
  --config configs/training_lora.yaml \
  --tuning lora \
  --model models/Qwen3.5-4B \
  --dataset-dir outputs/datasets/train \
  --output-dir /tmp/smoke_lora \
  --max-steps 5 \
  --device cuda
```

Điều kiện tiên quyết: `nvidia-smi` báo trống **≥ 12 GB** (11,49 GB + biên); `outputs/datasets/train/train.jsonl` tồn tại; `trainable_source` in ra phải là **`measured`**.
CLI gọi preflight **hai lần**: lần đầu (`scripts/train.py:343`, trước khi nạp trọng số) chỉ có thể in `placeholder_0.5pct`/10,88 GB; lần thứ hai (`scripts/train.py:430`, sau khi adapter đã gắn) in `measured` và **đó** mới là con số tin được.
Kỳ vọng: 5 optimizer step, `train_loss` hữu hạn, checkpoint ở `/tmp/smoke_lora`. **Không** ghi vào `outputs/checkpoints`.

### Bước 4 — đo VRAM thật (thay ước lượng bằng số đo)

Chèn **tạm** (không commit) vào `src/lifemate/training/trainer.py`, ngay sau `self.optimizer.step()` ở dòng **459** (trong `optimizer_step()`):

```python
            self.optimizer.step()
            if step == 0 and str(self.device).startswith("cuda"):   # tạm, chỉ để đo
                import torch as _t
                print("PEAK_ALLOC_GB", _t.cuda.max_memory_allocated() / 1e9,
                      "PEAK_RESERVED_GB", _t.cuda.max_memory_reserved() / 1e9, flush=True)
```

Vị trí chèn: ngay **sau** `self.optimizer.step()` ở dòng **459**, **trước** `step += 1` ở dòng **464** — nên `step == 0` đúng là bước optimizer đầu tiên (`ĐỌC TỪ FILE` `trainer.py:459,464`). Điều kiện device dùng `str(self.device).startswith("cuda")` theo đúng idiom của repo (`trainer.py:398`), không dùng `== "cuda"` — `--device cuda:0` sẽ trượt so sánh bằng.

Chạy lại Bước 3. Ghi lại hai số, so với **11,49 GB** (`SUY RA` hiện tại). Sau khi có số:
- peak ≤ 12,0 GB → giữ `per_device_train_batch_size: 1`.
- peak trong (12,0; 13,9] GB → có thể thử microbatch 2 nhưng chỉ khi trống ≥ 15,68 GB đo tại chỗ (microbatch 2 ước lượng 13,99 GB = 89,2%, sát ngưỡng `tight` 90%). Không dùng con số 15,68 GB của §6.1b như một bảo đảm — nó là một lần đo.
- peak > 13,9 GB → điều tra (ước lượng sai) **trước khi** chạy dài.

Sau khi đo, **xoá** khối chèn (đây là scaffolding, không phải tính năng).

### Bước 5 — chạy LoRA đầy đủ (`NOT RUN`)

```bash
cd ~/data/zaloai
HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/train.py \
  --config configs/training_lora.yaml \
  --tuning lora \
  --model models/Qwen3.5-4B \
  --dataset-dir outputs/datasets/train \
  --output-dir outputs/checkpoints \
  --device cuda \
  --run-name lora-v1
```

`--output-dir` là **thư mục cha**, không phải thư mục run. Layout thật là `<output_dir>/<run_name>/step-<N>` (`ĐỌC TỪ FILE` `trainer.py:170` `self.root = self.output_dir / self.run_name`, `:172-173` `step_dir` → `self.root / f"{tag}-{step}"`). Vì vậy `--output-dir outputs/checkpoints --run-name lora-v1` ghi ra `outputs/checkpoints/lora-v1/step-N`. Bản trước của tài liệu đặt `--output-dir outputs/checkpoints/lora-v1` **và** `--run-name lora-v1`, tức lồng đôi thành `…/lora-v1/lora-v1/step-N` — đã sửa. `ĐÃ CHẠY` đối chứng bằng `--smoke --tuning lora --output-dir /tmp/smoke_proof2 --run-name lora-v1` → `checkpoints : /tmp/smoke_proof2/lora-v1/step-2`.

Kỳ vọng (`SUY RA`): **1638** optimizer step (546/epoch × 3), effective batch 8, seed 42, val 482 record chấm cuối mỗi epoch. Thời gian: **CHƯA ĐO** — không có throughput nào trên GPU này.

### Bước 6 — suy luận + validate (`NOT RUN`; điều kiện: B5 cho phần `final_state`)

```bash
cd ~/data/zaloai
HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/infer.py \
  --split public_test \
  --data-root ./data \
  --model-mode lora \
  --checkpoint outputs/checkpoints/lora-v1/step-1638 \
  --model models/Qwen3.5-4B \
  --asr gipformer \
  --history-transcripts outputs/asr/history_transcripts.jsonl \
  --output outputs/submission_lora_v1.jsonl \
  --device cuda
```

Bắt buộc có `--history-transcripts`: không có thì run **dừng exit 2** (§4.3). `stderr` phải in `history transcripts: … filled=1299 existing=0 missing=0`.

**Điều kiện tiên quyết (đã gỡ):** cả `/tmp/cudaenv` và `/tmp/torchenv` (venv CPU này **đã bị xoá 2026-11-13**; mọi câu về nó dưới đây là bản ghi lịch sử) ban đầu **không có** `sherpa_onnx`, nên `--asr gipformer` dừng ở `ERROR: GipformerBackend requires 'sherpa-onnx' …` exit 2 (`ĐÃ CHẠY`). Đã cài **offline** từ uv cache vào `/tmp/cudaenv`:

```bash
uv pip install --python /tmp/cudaenv/bin/python --offline sherpa-onnx
# + sherpa-onnx==2.0.1
# + sherpa-onnx-core==2.0.1
```

Lệnh cài này chỉ chạy được vì wheel đã có sẵn trong uv cache; máy khác cần mạng (`pip install "sherpa-onnx>=1.10" "sherpa-onnx-core>=1.10"`).

**Bẫy thao tác 1 — `--checkpoint` phải trỏ vào thư mục `step-N`, không phải run parent.** Thư mục `step-N` là nơi chứa `adapter_config.json` + `adapter_model.safetensors` + tokenizer; `outputs/checkpoints/lora-v1` (parent) **không** có adapter. `ĐÃ CHẠY` đối chứng trên output smoke: `lora-v1/` không có `adapter_config.json` và `_has_tokenizer_files()` = **False**; `lora-v1/step-2/` có cả hai = **True**. Trỏ vào thư mục cha thì `_attach_lora_adapter` nhận một đường dẫn không có adapter và sai tokenizer source (`loader.py:231,392-410`). Số `1638` trong lệnh là tổng bước kỳ vọng — sau lần chạy thật phải `ls outputs/checkpoints/lora-v1/` để lấy đúng `step-N` (`save_strategy: epoch` → thực tế sẽ là `step-546`, `step-1092`, `step-1638`).

**Bẫy thao tác 2 — một môi trường cho cả bước 5 và 6.** Trước đây phải tráo venv (`torch` ở một chỗ, `sherpa_onnx` ở chỗ khác). Nay `/tmp/cudaenv` chạy được **cả hai** bước, đã xác nhận cùng process có:

```
sherpa_onnx 2.0.1
torch 2.15.0+cu132  cuda available True
```

`/tmp/asrenv` **không** dùng được cho suy luận dù có `sherpa_onnx 2.0.1`: nó **không có** `torch` lẫn `transformers`. `.venv` và `/tmp/torchenv` (venv CPU này **đã xoá 2026-11-13**) không có `sherpa_onnx`. Vì vậy bước 5 và bước 6 chạy trong **cùng một** môi trường, không cần tráo venv.

Validate bằng CLI chính thức. Interface thật là `--validate SUBMISSION` của `python -m zaic_tools` (`ĐỌC TỪ FILE` `data/public_test/zaic_tools/__main__.py:8-30`) — **không** có subcommand `check`, **không** có cờ `--submission`:

```bash
cd ~/data/zaloai
PYTHONPATH=data/public_test python3 -m zaic_tools \
  --split public_test --root data --validate outputs/submission_lora_v1.jsonl
```

`--root data` là bắt buộc trên thực tế: validator dựng `ToolEnv(split, row["id"], root)` cho **từng dòng** (`__main__.py:22`), và `ToolEnv` là **thứ duy nhất** được phép đọc trạng thái mẫu; nó cần root của split để tự định vị. Không có đường nào khác được dùng để đọc trạng thái, cố ý. Bỏ `--root` thì default `DATA_ROOT` vẫn trỏ đúng vì package nằm ngay trong thư mục split (`__init__.py:28-30`) — cả hai dạng đã chạy được (`ĐÃ CHẠY`).

Kỳ vọng:

| đầu ra | nghĩa |
|---|---|
| `634 lines checked, 0 schema errors` | đúng số dòng của `public_test`, **không** call nào bị từ chối |
| `N lines checked, M schema errors` với **M > 0** | call bị từ chối theo đúng luật của scorer; mỗi ca in `id call i tool: lỗi` |
| `id: no state in split public_test` | id không có trong split / sai `--root` — đó là lỗi đường dẫn, không phải lỗi schema |

`ĐÃ CHẠY` (kiểm chứng trên 5 dòng hợp lệ → `5 lines checked, 0 schema errors`; 1 dòng cố tình sai → `1 lines checked, 1 schema errors` kèm `tool 'alarm.create' không thuộc app đã cài`). **Exit code luôn 0 kể cả khi `bad > 0`** — phải đọc con số `M`, không được dựa vào `$?`.

Đây là kiểm tra **định dạng**, **không** phải điểm: không có nhãn trên `public_test` nên không tồn tại metric.

### Bước 7 — full FT (chỉ khi có máy ≥80 GB; `NOT RUN / BLOCKED` trên máy này)

```bash
cd ~/data/zaloai
HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/train.py \
  --config configs/training_full.yaml --model models/Qwen3.5-4B \
  --dataset-dir outputs/datasets/train \
  --output-dir outputs/checkpoints --run-name full-v1 --device cuda
```

Cùng quy ước `<output_dir>/<run_name>/step-N` → `outputs/checkpoints/full-v1/step-N`. Bản trước đặt `--output-dir outputs/checkpoints/full-v1` **không** kèm `--run-name`, nên sẽ ghi vào `…/full-v1/full/step-N` (vì `run_name` mặc định là `tuning_mode`) — đã sửa. Bản trước cũng gọi `python` trần (interpreter của system, **không có** `torch`); nay dùng `/tmp/cudaenv/bin/python` như bước 1-3, 5-6 — full FT cần CUDA torch chứ không chỉ CPU wheel.

### Trạng thái các bước

| bước | trạng thái | điều kiện tiên quyết | kích thước / thời gian |
|---|---|---|---|
| 1 verify CUDA | **ĐÃ CHẠY** | — | ~3 s |
| 2 tải trọng số | **NOT RUN** | mạng | **8,68 GiB** |
| 3 smoke 5 bước | **NOT RUN** | 1 + 2 + ≥12 GB trống | **CHƯA ĐO** |
| 4 đo VRAM | **NOT RUN** | 3 | **CHƯA ĐO** |
| 5 LoRA đầy đủ | **NOT RUN** | 3 + 4 | **CHƯA ĐO** (1638 bước) |
| 6 infer + validate | **NOT RUN** | 5 + B5 (`sherpa-onnx` đã có trong `/tmp/cudaenv`) | **CHƯA ĐO** |
| 7 full FT | **NOT RUN / BLOCKED** | card ≥80 GB | 69,98 GB |

### 12.1 Cheat-check: mọi cờ đã được đối chiếu với parser

Mỗi cờ dưới đây được đọc từ `parser.add_argument` của script tương ứng **và** được chấp nhận bởi một lần chạy thật (dry-run hoặc đến tận bước preflight).

**`scripts/train.py`** — 14 cờ, đọc tại `:46-73`:

| cờ trong §12 | dòng parser | chấp nhận bởi lần chạy thật |
|---|---|---|
| `--config` | `:46` | `--dry-run` exit 0 |
| `--tuning` | `:48` | `--dry-run` exit 0 (`choices=TUNING_MODES`) |
| `--dataset-dir` | `:53` | `--dry-run` exit 0, `n_train_records 4368 / n_val_records 482` |
| `--model` | `:54` | `--dry-run` exit 0 |
| `--output-dir` | `:55` | `--dry-run` exit 0 |
| `--max-steps` | `:57` | `--dry-run` exit 0 |
| `--device` | `:73` | `--dry-run` exit 0 |
| `--run-name` | `:68` | `--dry-run` exit 0 |
| `--seed` | `:66` | `--dry-run` exit 0 |
| `--seq-len` | `:72` | `--dry-run` exit 0 |
| `--dry-run` | `:58` | chính nó |
| `--resume` | `:56` (không dùng trong §12) | — |
| `--param-count` | `:60` (không dùng trong §12) | — |
| `--smoke` | `:65` (không dùng trong §12) | — |

Không có cờ nào trong §12 nằm ngoài danh sách trên. `ĐÃ CHẠY` hai lệnh dry-run (bước 3 và bước 5) — cả hai exit 0, in plan với `tuning_mode: lora`, `max_seq_length: 8192`, `n_train_records: 4368`, `group_hash` khớp manifest.

**`scripts/infer.py`** — 25 cờ, đọc tại `:80-136`:

| cờ trong §12 | dòng parser | chấp nhận bởi lần chạy thật |
|---|---|---|
| `--split` | `:80` | run thật đọc 634 mẫu |
| `--data-root` | `:81` | run thật |
| `--output` | `:82` | run thật |
| `--model-mode` | `:83-88` (`choices=("base","full","lora")`) | run thật |
| `--checkpoint` | `:89` | run thật |
| `--model` | `:102` | run thật |
| `--asr` | `:93` (`choices=(gipformer, oracle, whisper, none)`) | run thật với `none` |
| `--history-transcripts` | `:123-129` | run thật trên `public_test` in `filled=1299 existing=0 missing=0 (samples with history=419, samples receiving text=419)`; `--limit 1` in `filled=0 … samples with history=0` |
| `--device` | `:101` | run thật (`ERROR: device='cuda' … torch.cuda.is_available() is False`, exit 3 trên `/tmp/torchenv` — **lịch sử**, venv đã xoá 2026-11-13) |
| `--limit` | `:92` | run thật |
| `--max-steps` | `:97` (không dùng trong §12) | — |
| `--time-budget` | `:98` (không dùng) | — |
| `--allow-missing-history` | `:130-135` (không dùng — chỉ chẩn đoán) | — |

`ĐÃ CHẠY` phiên này trên ba môi trường: `/tmp/torchenv` (CPU torch — **lịch sử**, venv đã xoá 2026-11-13) đi qua `--history-transcripts` rồi dừng ở `torch.cuda.is_available() is False` (exit 3); `/tmp/cudaenv` **trước** khi cài `sherpa-onnx` dừng ở `GipformerBackend requires 'sherpa-onnx'` (exit 2), **sau** khi cài đi tiếp đến bước nạp tokenizer/trọng số và dừng ở thiếu cache trọng số. **Không lần chạy nào in `unrecognized arguments`** (`grep -c` = 0) — tức toàn bộ cờ được parser chấp nhận.

**`python -m zaic_tools`** — 6 cờ, đọc tại `__main__.py:10-14`:

| cờ | dòng | chấp nhận bởi lần chạy thật |
|---|---|---|
| `--split` | `:11` | `--apps` → `36 apps, 122 tools, 0 not implemented` |
| `--root` | `:11` | `--validate` với `--root data` → ok |
| `--validate` | `:13` | `5 lines checked, 0 schema errors` |
| `--apps` | `:12` | chính nó |
| `--id` / `--tool` / `--args` | `:14` | không dùng trong §12 |

**Ba cờ mà bản trước của tài liệu bịa ra và nay không còn:** subcommand `check`, `--submission`, và việc chạy từ trong `data/public_test` như *cách duy nhất*. Tất cả đã bị xoá.

---

## 13. Trả lời thẳng

### **"Có thể bắt đầu thí nghiệm SFT Qwen3.5-4B thật đầu tiên ngay bây giờ không?"**

# **KHÔNG.**

Không phải vì code — code đã sẵn sàng (479/485/491 test pass, đo 2026-11-13; 0 lỗi masking trên 4850 record, 0/3305 prompt mất schema, prompt train == prompt infer **0 lệch / 1534**, 3305/3305 prompt tự xác định đáp án). Mà vì **đúng một điều kiện cứng chưa thoả**, và một điều kiện mềm phải kiểm tra tại chỗ:

| # | điều kiện | trạng thái | lệnh gỡ | kích thước / thời gian |
|---|---|---|---|---|
| **1** | **Không có byte trọng số** | **THIẾU — blocker CỨNG duy nhất** | §12 bước 2 (`snapshot_download`) | **738 tensor / 2 shard / 8,68 GiB**; thời gian **CHƯA ĐO** (phụ thuộc mạng) |
| 2 | VRAM trống | **BIẾN ĐỘNG — đo được 15,68 / 11,82 / 7,72 / 3,03 GB trong cùng ngày**; hai mốc cuối cách 78 giây, mốc mới nhất **KHÔNG đủ**. Không phải blocker vật lý, là tranh chấp tenant | đo `nvidia-smi` **ngay trước** launch; nếu free < 12 GB thì chờ tenant thoát và đo lại | 11,49 GB cần; ở 3,03 GB là **379,7%** |
| 3 | Môi trường CUDA | **ĐỦ** — `/tmp/cudaenv` `2.15.0+cu132`, `sm_120` (VRAM không liên quan tới mức này) | — | — |

**Nếu** tải xong trọng số (bước 2) **và** `nvidia-smi` ngay lúc đó báo trống ≥ 12 GB, câu trả lời đổi thành **CÓ** — chạy §12 bước 3 (smoke 5 bước) rồi bước 5. Nếu lúc đó trống < 12 GB, câu trả lời là **CHƯA** cho tới khi tenant ngoài thoát.

**Vì sao không ghi một con số VRAM duy nhất ở đây:** trên cùng card, cùng driver, trong cùng ngày, free VRAM đã đo được **15,68 → 3,03 GB** (biên độ **12,65 GB**, lớn hơn cả 11,49 GB mà run cần), trong đó hai mốc cuối chỉ cách nhau 78 giây. Bất kỳ con số nào in ở đây cũng hết hạn trong vòng một giờ. Bảng có timestamp nằm ở §6.1b; B2 (§11) được đổi thành "tranh chấp tạm thời — kiểm tra lúc launch", **không** còn là blocker vật lý. Blocker cứng vẫn chỉ có trọng số.

**Điều câu trả lời này KHÔNG nói:** rằng chạy xong sẽ có kết quả tốt. Không có phép đo chất lượng nào tồn tại: 0 checkpoint, 0 điểm, `public_test` không nhãn, scorer final-state không có trong gói phát hành. Thí nghiệm đầu tiên sẽ trả lời "loss có giảm không", không trả lời "điểm bao nhiêu".

### Ba mức sẵn sàng

| mức | kết luận | căn cứ |
|---|---|---|
| **DATA READY** | **CÓ** | 4368 + 482 record, 0 drop, 0 leak, 0 lỗi mask, 159.449 token supervise, 0/3305 prompt thiếu schema, 19 điểm bỏ có chủ đích và có test ghim |
| **CODE READY (train path)** | **CÓ** | 479/485/491 test (đo 2026-11-13; bộ ba cũ 468/468/473 là lịch sử, đo trên `/tmp/torchenv` đã xoá); 17/17 gate PASS trên cây hiện tại (§9.1, gate Q không chạy vì GPU bị tạm dừng); prompt train/infer `0 lệch / 1534`; loader từ chối fixture; guard truncation/observability/history đều có test |
| **GPU READY (môi trường)** | **CÓ** | `/tmp/cudaenv` = `2.15.0+cu132`, `sm_120`, bf16 matmul+backward finite. VRAM trống không thuộc mức này — xem §6.1b |
| **TRAINING READY** | **KHÔNG** | thiếu trọng số (8,68 GiB) — blocker cứng duy nhất; VRAM phải đo lại lúc launch (đã thấy 3,03-15,68 GB); 0 optimizer step trên model thật; throughput **CHƯA ĐO** |
| **COMPETITION READY** | **KHÔNG** | không checkpoint, không điểm, không scorer final-state, docker chưa chạy được (không daemon); pin Dockerfile đã sửa sang `cu128`/CUDA 12.8.1 cho `sm_120` nhưng image **chưa từng build, chưa từng chạy** → **UNVERIFIED** |

---

## Phụ lục A — Artifact dùng làm bằng chứng

| artifact / file | nội dung |
|---|---|
| `outputs/datasets/train/{train.jsonl,val.jsonl,manifest.json,stats.json}` | 4368 / 482 record; count, drop, truncation, `group_hash` |
| `outputs/asr/{train_request_transcripts,history_transcripts}.jsonl` | 1545 / 4790 dòng, 0 error |
| `src/lifemate/data/trajectory_builder.py:371-467,661-703` · `training/collator.py:63,161,179-193,197-252,329-368` · `training/memory_preflight.py:75-89,151-200` · `scripts/infer.py:130-135,538-583` · `docker/entrypoint.sh:35,67-71` | code của các bản sửa ở §10 |
| `outputs/verification/verify_all.json` | 18 gate @ `2026-11-08T13:25:54+0700` — **PRE-CHANGE**, chỉ giữ làm bản ghi lịch sử (`regression` ghi `passed=457`, trước mọi thay đổi §10); trạng thái hiện tại là §9.1 (17 gate, PASS 17/17) |
| `/tmp/lora_recon.txt` | 12 suffix / 248 module / 32.464.896, đối chiếu peft + giải tích |
| `/tmp/rbaudit/{promptlevel.json,unsat496.json,final.txt}` | phân loại 496 mẫu, đối chiếu cờ cũ |
| `/tmp/numan/{final.json,s.txt,table2.txt,absent.txt,gates.py,rulemeasure.py}` | ASR span số + materiality |
| `/tmp/lat/{m1,m2,m3,m4,m5a,m5b,m7_hist}.json` | latency từng thành phần |

## Phụ lục B — Những con số trong brief KHÔNG xác minh được

Mọi mục ghi `CHƯA ĐO` và **không** được dùng làm căn cứ; chỗ cần thì đã thay bằng số đo được.

| con số trong brief | trạng thái | số thay thế | lý do |
|---|---|---|---|
| cờ cũ "báo thừa ~91%" | **CHƯA ĐO** | **98,1%** (chỉ tính references proper) hoặc **18,2%** (tính cả `x-match`) | thử 5 định nghĩa (mẫu/arg/điểm × proper/all) đều không ra 91% |
| "943 decision point mang cờ" | **sai đơn vị** | **378** decision point mang cờ (train); 943 = tổng **gold call** của 319 mẫu | `ĐÃ CHẠY` cả hai phép đếm |
| rank chuẩn hoá `0,4643` | **CHƯA ĐO** | **0,4434** (`i/(n-1)`, 3305 prompt) | thử 4 định nghĩa: 0,4434/0,4296/0,4449/0,4588 |
| "2 of 35 material span làm call schema-invalid" | **CHƯA ĐO** (chỉ xác minh 1) | **1** (`train_1271` `threshold_kwh` required) | 6 ca còn lại đều `schema_after_drop_ok=True` |
| history material "19 span / 0,54%" | **lệch** | **20 span / 20 mẫu = 0,57%** | chênh đúng ca `cardinal` + `material_schema_invalid` |
| phone gate "0 false positive trên reference" | **lệch** | **3** reference phone có 8 hoặc 6 chữ số (histogram `{11:14, 8:1, 6:2}`) | gate theo độ dài sẽ cờ oan 3 |
| latency "ASR 102,1 ms / subtotal 118 ms" | **CHƯA ĐO** | ASR **71,3-128,7 ms** tuỳ cách tổng hợp; subtotal **~125-135 ms** (`SUY RA`) | không file nào trong `/tmp/lat` chứa 102,1 ms |
| "history load+apply 0,37 ms amortized" | **lệch** | **0,10 ms** lần đầu; **0,0044 ms/mẫu** bulk 634 | `m2.json` |
| "env calls ×3 0,08 ms" | **lệch nhỏ** | read **0,036** / write **0,016** ms | `m4.json` |
| "prompt build ×3 0,86 ms" | **lệch nhỏ** | **0,64 ms** (0,231+0,205+0,201) | `m1.json` |
| "11,30 GB cho LoRA" (comment config) | **cũ** | **11,49 GB** | đường measured (§8.3); 11,30 ứng với placeholder 21,03 M |
| "free VRAM 11,26 GB" (bản trước của **tài liệu này**, không phải brief) | **sai đơn vị** | 11276 MiB = **11,82 GB** decimal (11,01 GiB). VRAM trống còn đổi tiếp (15,68 / 11,82 / 7,72 / 3,03 GB — bảng §6.1b); "11,26" không xuất hiện ở lần đo nào | `nvidia-smi` bốn lần đo |

Không có con số nào trong tài liệu này được tạo ra để khớp với brief. Chỗ lệch, báo cáo nêu cả hai và nói rõ cái nào đo được.

**Quan hệ với phần thân:** mọi dòng ở trên là **số thay thế đã đo**, và phần thân đã dùng đúng số thay thế đó — không có chỗ nào trong thân còn khẳng định con số trong brief. Cụ thể: `rank 0,4434` (§3.5), `20 span / 0,57%` (§5.3), `1 ca schema-invalid` (§5.3), `subtotal ~125-135 ms` (§8.7), `11,49 GB` (§8.3), `378 decision point` (§1.4). Bảng này tồn tại để nói rõ **vì sao** các con số brief bị thay, không phải để giữ chúng song song.

---

## Phụ lục C — Trạng thái wandb và W&B Sandbox (đo 2026-11-13)

Phụ lục này ghi lại hai hạng mục **mới được đo trong phiên 2026-11-13** và **không** thay đổi verdict ở §13: vẫn **KHÔNG** chạy được thí nghiệm huấn luyện thật, vì lý do duy nhất vẫn là thiếu byte trọng số Qwen3.5-4B. Chi tiết thao tác của sandbox nằm ở `cloud/README.md`; phụ lục này chỉ ghi kết luận và bằng chứng, không chép lại.

### C.1 Đường tracking wandb đã hoạt động (offline)

**Config.** `configs/training_lora_wandb.yaml` là một **bản sao** của `configs/training_lora.yaml`, không phải include: `lifemate.config.load_config()` (`src/lifemate/config.py:547`) chỉ áp **một** tài liệu YAML cộng overrides, không có `include`/`extends`. Đo bằng cách nạp cả hai config rồi so từng khoá sau khi resolve (`ĐÃ CHẠY`), **đúng ba** khoá khác nhau:

| khoá | `training_lora.yaml` | `training_lora_wandb.yaml` |
|---|---|---|
| `training.report_to` | `[]` | `["wandb"]` |
| `wandb.enabled` | `false` | `true` |
| `wandb.mode` | `'online'` | `'offline'` |

Mọi khoá còn lại (kể cả toàn bộ khối `lora`, `model`, và 21 khoá `training`) **giống hệt**. `configs/training_lora.yaml` **không bị sửa**: nó vẫn là config ship mặc định không-wandb, `wandb.enabled: false`, `report_to: []`.

**Cần cả hai cờ, không phải một.** `WandbTracker.requested` (`src/lifemate/tracking/wandb_tracker.py:185`) chỉ đọc `wandb.enabled`; `training.report_to: ["wandb"]` **một mình không bật** tracking. Đo (`ĐÃ CHẠY`): `report_to=["wandb"]` + `wandb.enabled: false` → `requested=False`, `disabled_reason="wandb.enabled is false (no-op tracker)"`. Ngược lại, `_build_tracker()` trong `scripts/train.py:548` lại dùng `enabled = wandb.enabled OR "wandb" in report_to`, tức hai đường khác nhau trong cùng một luồng. **Docstring của `_build_tracker` ghi `"wandb"` (hoặc `wandb.enabled: true`) là đủ — câu đó SAI** so với hành vi thật của tracker; đây là lỗi tài liệu trong code, chưa sửa (ngoài phạm vi phiên này), và config ship kèm đặt **cả hai** nên không lộ ra khi chạy.

**`WANDB_MODE` thắng config.** `WandbTracker.__init__` lấy `self.mode = os.environ.get("WANDB_MODE") or wandb.mode`. Đo (`ĐÃ CHẠY`): config `mode: offline`, env `WANDB_MODE=disabled` → `mode='disabled'`, `enabled=False`, `disabled_reason="WANDB_MODE=disabled"`; bỏ env đi thì `mode='offline'`, `enabled=True`. `--smoke` **cố ý bỏ qua** tracking: `_build_tracker` trả `NullTracker()` trước mọi nhánh khác.

**Bề mặt gọi của trainer.** Trainer chỉ gọi đúng hai thứ trên tracker: `log(dict)` (mỗi cửa sổ logging một lần ở `trainer.py:583`, cộng một lần sau mỗi lần evaluate ở `:471`/`:528`) và `finish()` (`:539`, trong `finally`). `log_artifact` **không bao giờ** được gọi từ đường huấn luyện — nó chỉ tồn tại trong `lifemate/tracking/artifact_manager.py:157-163` và chỉ dùng cho `cloud/sandbox_runner.py download`.

**Chứng minh redaction (đo lại phiên này).** Chạy `WandbTracker` thật ở `mode=offline` với một payload chứa **5 giá trị bí mật/PII** (`wandb.api_key`, `wandb.token`, `user_profile.name`, `messages[0].content`, `email`) cộng 2 bước `log()` có thêm `api_key`/`password`/`phone`/`email` (`ĐÃ CHẠY`, `WANDB_MODE=offline`, `WANDB_BASE_URL` không đặt, `WANDB_API_KEY` không có trong env). Kết quả wandb in ra: `api_key [REDACTED]`, `password [REDACTED]`, `phone [DROPPED_PII]`, `email [DROPPED_PII]`. Quét **toàn bộ byte** wandb ghi xuống đĩa (8 file, **15.813 B**): **0 hit** cho cả 5 giá trị. Quy tắc đang chạy: key khớp `api_key|token|secret|password|passwd|authorization|credential` → `[REDACTED]`; key thuộc `PII_KEYS` (`user_profile`, `messages`, `audio_text`, `transcript`, `conversation`, `history`, `content`, `raw_text`, `phone`, `email`, `address`, `full_name`, `name`) → `[DROPPED_PII]`; đệ quy vào dict/list.

### C.2 `describe_environment()` đã từng rò rỉ — nay là allowlist

Bản trước gửi **`hostname`** và **`user`** (tên đăng nhập của operator, và một hostname *chứa* tên đó) lên server bên thứ ba cùng mỗi run. Đó là **rò rỉ thật**, không phải rủi ro lý thuyết: định danh máy + tài khoản là dữ liệu cá nhân và không cần cho việc đọc biểu đồ loss.

Bản hiện tại (`src/lifemate/tracking/wandb_tracker.py:108`) là **allowlist**, không phải bộ lọc: `platform` (nay chỉ là `sys.platform`), `python`, `git_commit`, `git_dirty`, `torch`, `cuda`, `cuda_available`, `gpu` (nay là **lớp capability** `cuda-sm_120`, không phải tên card), `transformers`. Đo lại trên đĩa (`ĐÃ CHẠY`, đọc thẳng file `run-*.wandb`): khối `environment` đúng 9 khoá trên, **không** có `hostname`, **không** có `user`.

**Điều allowlist này KHÔNG che:** bản thân wandb vẫn ghi hostname của máy vào record run (trường `host` của `RunRecord`) và ghi đường dẫn tuyệt đối của interpreter vào metadata. Đó là hành vi của SDK, ngoài `describe_environment`; muốn chặn phải đặt `WANDB_HOST`/`WANDB_*` hoặc tự dựng `wandb.Settings`, và phiên này **không** làm.

### C.3 Credential: `~/.netrc`, không phải env

Trên máy này credential wandb đến từ **`~/.netrc`** (`machine api.wandb.ai`), **không** từ `WANDB_API_KEY`. wandb map host đó sang một instance Forge tự vận hành (`api.forge.coreweave.com`). `cloud/sandbox_runner.py` đã được sửa để **chấp nhận một trong hai nguồn**: `resolve_credentials()` báo `source` là `env:WANDB_API_KEY`, `netrc`, hoặc `absent`, kèm **host** — **không bao giờ** in giá trị. Preflight vì vậy exit `0` trên máy này. (Tên và host đã kiểm; giá trị không đọc ra ngoài.)

**Cảnh báo thật đã xảy ra trong phiên:** một test của worker **vô tình tạo 4 run online** trên instance Forge đó. Cả 4 đã bị xoá và project được kiểm lại là rỗng. Cơ chế gây ra nó đáng ghi lại: một run có `wandb.enabled: true` **và không có `WANDB_MODE`** sẽ đi **online im lặng** — không prompt, không xác nhận — vì `mode` mặc định lấy từ config, và config mặc định của repo là `'online'`. Với một operator *có* credential trong `~/.netrc`, đó là đường ghi dữ liệu ra ngoài không có bước duyệt nào. Vì vậy: chạy thử phải đặt `WANDB_MODE=offline` **hoặc** dùng `configs/training_lora_wandb.yaml` (đã offline), và test không được bật `wandb.enabled` mà không ép mode.

### C.4 W&B Sandbox: CPU chạy được, GPU bị từ chối ở tầng entitlement

Probe thật, `cwsandbox` 1.28.0, **một** sandbox CPU, cap 60 s, đã xoá (`ĐÃ CHẠY`):

| hạng mục | kết quả |
|---|---|
| image | `python:3.12-slim`, `resources` không yêu cầu (CPU) |
| create → accepted | **3,16 s**; `runtime_class cw-kata-qemu`; `resource_gpu None` |
| `mounted_files` | **chạy**: file 2030 B mount vào `/app/`, import OK |
| egress PyPI | **HTTP 200** (`pip download six` thành công) |
| teardown | `Sandbox.delete(id)` → `completed`; `list()` → **0** |

**GPU: bị từ chối, không phải "chậm" hay "hết hàng".** Cả **7** shape đều trả **cùng một** lỗi:

```
CWSandboxAuthenticationError: Permission denied: this organization is not entitled to create GPU sandboxes
```

Bảy shape: `{"count": 1}` **không** `type`, rồi `A100`, `A100-80GB`, `RTX-A6000`, `L40`, `H100`, `T4`. Vì lỗi giống nhau ở mọi `type`, **loại GPU là không liên quan** — entitlement của org mới là thứ thiếu, và đó là thiết lập control-plane, không phải thay đổi code.

**Hệ quả phải nói thẳng:** `nvidia-smi` **chưa từng chạy** trong sandbox, và **không có lần huấn luyện nào xảy ra**. Probe CPU ở trên **không phải** training; nó chỉ chứng minh đường compute chạm được.

**Chi phí:** **0 GPU-second**. Bảy lần create GPU bị từ chối **trước** cấp phát (~1,3 s mỗi lần, không pull image, không container). Một sandbox CPU ~8 s trong cap 60 s — không phải compute GPU.

**Ba lỗ hổng giao hàng, không phải một:**

1. **Code — có đường.** `mounted_files` chở được 2,3 MB của `src/scripts/configs/docker`. Không cần registry, không cần docker daemon.
2. **Dữ liệu — không có đường.** `data/` là **1,1 GB**: quá lớn cho `mounted_files`, cần object storage có signed URL, hoặc URL dataset công khai, hoặc bake vào image. **Chưa có kế hoạch nào.**
3. **Trọng số — không có đường.** Qwen checkpoint **8,68 GiB**: cùng vấn đề, cộng thêm egress HF **chưa được kiểm** bên trong sandbox (egress mới chỉ chứng minh cho PyPI).

**Xung đột mô hình thực thi:** job spec của repo dùng **host bind-mount** (`/data` ro, `/models/ckpt` ro, `/outputs` rw), trong khi API sandbox chỉ nhận `container_image` + `mounted_files` (byte file, read-only) — serverless không bind được thư mục host. Nên **kể cả** khi đã có code trong sandbox, spec hiện tại **vẫn không chạy được** ở đó. `docker/Dockerfile` **chưa từng được build** và máy này **không có binary `docker`**.

**Kết luận C.4:** sandbox compute là **reachable và authenticated**; GPU là **bị chặn bởi org**; dữ liệu và trọng số **chưa có đường vào**. Không có tổ hợp nào ở trên tạo ra một bước huấn luyện, và phụ lục này không khẳng định bất kỳ kết quả GPU nào.

### C.5 Ảnh hưởng lên verdict §13

**Không đổi.** `TRAINING READY` vẫn **KHÔNG**, `COMPETITION READY` vẫn **KHÔNG**. Điều kiện cứng vẫn là **byte trọng số** (§12 bước 2). Hai hạng mục mới chỉ đóng thêm hai câu hỏi phụ: *"log được không"* → **được, offline, đã redact**; *"mượn GPU qua sandbox được không"* → **không, org không có entitlement**. Cả hai đều không thay thế được một lần chạy thật.
