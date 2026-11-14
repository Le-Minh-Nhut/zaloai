# Hướng dẫn fine-tune Qwen3.5-4B trên Google Colab

Notebook: [`finetune_qwen35_colab.ipynb`](./finetune_qwen35_colab.ipynb)

Notebook này **không** tự viết lại vòng lặp train. Nó gọi code trong repo
`Le-Minh-Nhut/zaloai` (`scripts/prepare_dataset.py`, `scripts/train.py`,
`scripts/evaluate.py`, `scripts/export_model.py`), nên mọi thay đổi về sau trong
repo vẫn áp dụng được.

> **Chưa có lần chạy Colab nào.** Mọi con số thời gian / Compute Unit / VRAM
> dưới đây là **ESTIMATE** (ước lượng), trừ khi được ghi rõ là **TESTED**. Những gì
> đã kiểm tra được ở máy local: notebook là JSON hợp lệ, mọi code cell `ast.parse`
> thành công, và mọi flag CLI notebook dùng đều tồn tại trong `--help` của script
> tương ứng tại commit hiện tại.
>
> Ba giới hạn thật của môi trường này, và chỉ ba điều đó:
>
> 1. **Không truy cập được runtime Google Colab** từ máy này, nên không có bước nào
>    (load `Qwen/Qwen3.5-4B`, train, eval, export) được chạy trên Colab.
> 2. **Không có weights `Qwen/Qwen3.5-4B`** trên đĩa — chỉ có `models/MiniCPM5-2B`.
> 3. **Không có `bitsandbytes`**, nên QLoRA chưa được thực thi ở bất kỳ đâu.
>
> Phần *đã* chạy thật trên máy này: có GPU **RTX 5070 Ti (16.57 GB)**, và
> `models/MiniCPM5-2B` đã được load thật, forward + backward thật và generate thật
> (số đo ở README gốc). Nhưng đó **không** phải Qwen3.5-4B, **không** phải QLoRA và
> **không** phải trên Colab.

---

## 1. Cách mở notebook

1. Mở link Colab:
   `https://colab.research.google.com/github/Le-Minh-Nhut/zaloai/blob/main/notebooks/finetune_qwen35_colab.ipynb`
   (Colab sẽ hỏi có mở bản sao vào Drive hay không — chọn **Save a copy in Drive**).
2. Hoặc: tải file `.ipynb` lên `File → Upload notebook`.
3. Notebook **tự clone repo** ở mục 3, nên không cần clone tay. Ô `BRANCH_OR_COMMIT`
   cho phép ghim một commit cụ thể (`main` = đầu branch).

## 2. Chọn GPU Colab

`Runtime → Change runtime type → Hardware accelerator → GPU`.

| GPU | VRAM | bf16? | Notebook làm gì |
| --- | --- | --- | --- |
| T4 | 16 GB | **Không** (sm_75) | tự chọn `float16`; mục 2b tự suy ra `TRAINING_METHOD=qlora` (LoRA 16-bit chỉ vừa tới seq 3840) |
| L4 | 24 GB | Có | `bfloat16`; LoRA 16-bit chạy được |
| A100 | 40 / 80 GB | Có | `bfloat16`; dư VRAM, có thể tăng `MAX_SEQ_LENGTH` |

Mục 1 của notebook in ra tên GPU, VRAM, phiên bản CUDA/torch và kết quả một phép
nhân ma trận thật trên device. Nếu GPU < 14 GB VRAM hoặc compute capability < 7.0,
notebook **dừng** với thông báo cụ thể thay vì OOM giữa chừng.

## 3. Kết nối Google Drive

Đặt `USE_GOOGLE_DRIVE = True` ở mục 2. Notebook mount Drive và dùng:

```text
/content/drive/MyDrive/lifemate_colab/
├── data/          # dữ liệu LifeMate (chỉ split train được dùng)
├── datasets/      # SFT records đã build
├── checkpoints/   # checkpoint đã sync, mỗi thư mục có SYNC_COMPLETE.json
├── outputs/       # export adapter, report đánh giá, run_report.json
└── logs/          # stdout của train
```

Checkpoint được ghi **nguyên tử và có thể rollback**:

```text
copy -> <dst>.incomplete-<ts>            # bản copy đang chạy, không bao giờ là checkpoint
     -> fsync cả cây
     -> SYNC_COMPLETE.json (size + sha256 của TỪNG file, tổng, source, schema=2)
     -> nếu <dst> đã tồn tại: rename <dst> -> <dst>.prev-<n>   (KHÔNG xoá)
     -> rename staging -> <dst>
     -> verify lại <dst>, rồi mới prune các .prev-* cũ (giữ KEEP_PREVIOUS=1)
```

Không có bước nào `rmtree` bản duy nhất còn sống: crash ở bất kỳ điểm nào cũng để lại
một thư mục **đầy đủ** dưới `<dst>` hoặc `<dst>.prev-<n>` (test giả lập crash giữa hai
lần rename và kiểm tra đúng điều đó). Thư mục staging còn sót mang tên `.incomplete-*`
và bị `checkpoint_ready()` từ chối thẳng.

Resume **verify** marker chứ không chỉ nhìn sự tồn tại của nó: mọi file được liệt kê đều
được băm lại (size + sha256), nên một bản copy bị cắt cụt mà vẫn còn marker cũ sẽ bị từ
chối kèm **tên file** sai. Marker do phiên bản notebook cũ ghi (không có `sha256`) được
báo là `legacy marker, cannot be verified` và **không** dùng để resume — phải sync lại
thư mục đó. Các kiểm tra cũ (`training_state.json` với `step > 0`, có weights, có
`optimizer_state.pt`) vẫn nguyên vẹn, và còn được siết thêm `run_name` khi resume từ Drive.

`USE_GOOGLE_DRIVE = False` vẫn chạy được, nhưng mất runtime là mất checkpoint.

## 4. Cung cấp dataset LifeMate

Chọn một trong ba cách (mục 4c của notebook tự dò theo thứ tự):

1. **Đã có trên Drive**: copy `data/` (chứa `train/train.jsonl`,
   `train/train_tools_catalog.json`, `train/audio/`) vào `lifemate_colab/data/`.
   Notebook sẽ copy split `train` sang vùng local.
2. **Zip trên Drive**: đặt đường dẫn vào `DRIVE_DATA_ZIP`.
3. **Upload trực tiếp**: để trống cả hai, chạy cell upload và chọn file `.zip`.

Lưu ý tuân thủ (`docs/COMPETITION_COMPLIANCE.md`):

* notebook chỉ đọc split `train`, **không bao giờ** đọc `public_test` / `private_test`;
* mọi đường dẫn chứa thư mục `*_environment` bị chặn ngay ở tầng notebook (khi giải
  nén, khi copy, khi sync) — không list, không mở, không upload;
* nhãn test không bao giờ được dùng để train hay để chấm.

## 5. Chọn LoRA hay QLoRA

`TRAINING_METHOD` ở mục 2:

| | `lora` | `qlora` |
| --- | --- | --- |
| Base | bf16/fp16 (đóng băng) | 4-bit NF4 (đóng băng) |
| VRAM base 4B | ~8.8 GB | ~2.4 GB (ước lượng) |
| Artifact | adapter LoRA | adapter LoRA (giống nhau) |
| Cần | peft | peft + **bitsandbytes** |
| Hợp với | L4 / A100 | **T4** (16 GB), hoặc seq dài |

Notebook kiểm tra `--tuning {full,lora,qlora}` từ chính `--help` của `train.py`.
Nếu checkout không có `qlora`, notebook **dừng** với thông báo (không âm thầm train
16-bit rồi dán nhãn QLoRA). Khi `qlora` khả dụng, notebook gọi đúng đường của repo:
loader với `quantization="nf4"` + `configure_qlora()` (nội bộ gọi
`prepare_model_for_kbit_training`).

Target module LoRA **luôn** được dò từ kiến trúc thật
(`discover_target_modules`), gồm cả projection hybrid của Qwen — không hard-code
`q/k/v/o/gate/up/down`.

Cấu hình khởi đầu khuyến nghị (**ESTIMATE**, trừ khi ghi TESTED):

| | T4 16 GB | L4 24 GB | A100 40/80 GB |
| --- | --- | --- | --- |
| `TRAINING_METHOD` | `qlora` (bắt buộc ở 8192) | `lora` (tới 8192) | `lora` (tới 16384 / 32768) |
| `MAX_SEQ_LENGTH` | 8192 với `qlora`, tối đa 3840 với `lora` | 8192 với `lora`, tới 12288 với `qlora` | 8192 (A100-40 có thể 16384; A100-80 tới 32768) |
| `MICRO_BATCH_SIZE` | 1 | 1 | 1–2 |
| `GRADIENT_ACCUMULATION_STEPS` | 8 | 8 | 8 (hoặc 4 nếu micro=2) |
| `NUM_EPOCHS` | 3 | 3 | 3 |
| `LEARNING_RATE` | 2e-4 | 2e-4 | 2e-4 |
| VRAM dự kiến (đỉnh) | 13.5 GB — **ESTIMATE** | 19.8 GB — **ESTIMATE** | 29.3 GB (A100-40, seq 16384) / 48.2 GB (A100-80, seq 32768) — **ESTIMATE** |
| Thời gian / epoch (4368 records, ~546 step) | 1.3–2.3 h — **ESTIMATE** | 0.8–1.3 h — **ESTIMATE** | 0.4–0.8 h — **ESTIMATE** |

**GPU quyết định cả METHOD, không chỉ sequence length.** Qwen3.5-4B là
`Qwen3_5ForConditionalGeneration` với phần text ~4B tham số (hidden 2560, 32 lớp):
weights fp16 đóng băng ~8.8 GB, cộng ~5.4 GB activation, ~4.1 GB logits **bf16**
(`[seq, vocab]` ở 8192 token) và ~1.0 GB cross-entropy fp32 (chunked) =
**19.8 GB ESTIMATE**, trong khi T4 chỉ dùng được ~14.9 GB. Vì vậy trên tier
`T4-16GB`, `lora` @8192 bị suy ra thành `qlora` (**13.5 GB ESTIMATE**) kèm phép
tính in ra, và `lora` bị chặn ở **3840** (**14.8 GB ESTIMATE**, ~100% usable →
"tight"). Mọi con số đó là **ESTIMATE** (heuristic, in kèm nhãn).

**Vocabulary không còn hard-code.** Ô 2b lấy `vocab_size` theo thứ tự: (a) `config.json`
của model nếu có đường dẫn local (kể cả `text_config.vocab_size`) → (b) `preflight_vocab`
từ `src/lifemate/models/profiles.py` → (c) **raise**, không đoán. Qwen3.5-4B khai báo
**248320**; con số 151936 của Qwen2/Qwen2.5 cũ đã bị xoá vì nó đánh giá thấp hạng tử
logits **1.63 lần**. Số dùng được in ngay cạnh ước lượng, kèm nguồn.

**Hai hạng tử logits, không phải một.** Đường chunked mà trainer thực chạy giữ *hai*
tensor cùng lúc: (1) `logits_compute_gb` — output đầy đủ của causal-LM head
`[micro_batch, seq, vocab]` ở dtype compute (bf16/fp16, 2 B/logit; fp32 nếu chạy fp32).
Vì `loss_chunk_tokens > 0` nên trainer gọi model **không** kèm `labels`, head trả logits
thay vì loss, và tensor đó sống suốt `backward()` vì chunked CE đọc từng slice của nó:
8192 × 248320 × 2 B = **4.1 GB**; và (2) `logits_chunked_gb` — bản upcast fp32 của *một*
slice, `min(loss_chunk_tokens, seq) * micro_batch * vocab * 4` (mặc định 1024 →
**1.0 GB**). `logits_gb` = (1) + (2). Bảng 2b in thêm `logits_unchunked` để đối chiếu:
**8.1 GB** nếu upcast fp32 cả sequence (8192 × 248320 × 4 B) — đúng khoản allocation mà
`loss_chunk_tokens` tồn tại để tránh. Chỉ tính (2), như bản trước, đánh giá thấp mỗi lần
chạy chunked đúng bằng khoản (1).

`MAX_SEQ_LENGTH` bạn đặt **không bao giờ bị hạ âm thầm**: khi nó không vừa tier, mục 2b
in `!! MAX_SEQ_LENGTH <cũ> -> <mới>` kèm ước lượng đã buộc phải chặn và cách giữ giá
trị cũ (sequence nhỏ hơn, hoặc runtime lớn hơn).

`MAX_SEQ_LENGTH = 0` = tự suy từ phân bố token **đo được** ở mục 5 (tokenize thật
300 record trên CPU). Mặc định trong ô param là 8192, khớp với cấu hình repo đã
đo: 47/4368 record bị cắt giữa (1.08%), 0 record bị drop — **TESTED trên dataset
local**, không phải trên Colab.

## 6. Bắt đầu train

Thứ tự chạy: mục 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.

* Mục 7 chạy **probe 2 step thật** rồi báo đỉnh VRAM đo được (không phải ước lượng).
  Nếu đỉnh > 92% VRAM, notebook dừng trước khi train dài.
* Mục 8 in epoch/step/loss/val loss/LR/VRAM mỗi lần log; checkpoint xong là sync
  Drive ngay trong lúc train.
* W&B là tuỳ chọn, mặc định tắt (`training.report_to: []`).
* Trên GPU không có bf16 (T4) mục 1 và mục 6 in cảnh báo `float16`: fp16 cần loss
  scaling, repo trainer cấp nó bằng `torch.amp.GradScaler`
  (`src/lifemate/training/trainer.py::_make_grad_scaler`), và đường fp16 + scaler đó
  **chưa được kiểm chứng** ở đây. Trên T4 nên dùng `TRAINING_METHOD=qlora`.
* Mục 1 export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` vào `os.environ`
  trước khi gọi `scripts/train.py` (mục 8), nên tiến trình con thừa hưởng: giảm phân
  mảnh allocator ở sequence dài. Biến này đọc lúc process tạo CUDA context, nên phải
  đặt trước khi spawn.

Compute Unit: chỉ tính theo thời gian runtime được cấp phát. Runtime để không vẫn
bị tính tiền — xong việc thì `Runtime → Disconnect and delete runtime`.

## 7. Resume sau khi mất kết nối

1. Mở lại notebook, chạy lại mục 1 → 7 (mục 7 có thể bỏ qua nếu đã biết cấu hình
   vừa khớp VRAM).
2. Ở mục 2, đặt:
   `RESUME_FROM_CHECKPOINT = "/content/drive/MyDrive/lifemate_colab/checkpoints/<RUN_NAME>/step-<N>"`
3. Chạy mục 8.

Notebook kiểm tra trước khi resume:

* nếu checkpoint đến từ Drive: `SYNC_COMPLETE.json` phải **verify được** — mọi file
  trong marker được băm lại (size + sha256); thiếu file, thừa file, sai size, sai
  sha256, hay marker `legacy` (không có sha256) đều bị **từ chối** kèm tên file;
* marker (nếu có ghi) phải mang đúng `run_name` của session này;
* có `training_state.json` với `step > 0`,
* có weights **và** `optimizer_state.pt`,
* `config_hash` của checkpoint khớp `config_hash` hiện tại (lr, seed, seq len,
  batch, epoch, precision, tuning mode) — lệch là dừng, vì trainer cũng sẽ từ chối.

### 7.1 Mục 9: đọc (luôn luôn) và chứng minh resume (tuỳ chọn)

Mục 9 có hai nửa:

1. **Luôn chạy, chỉ đọc**: in nội dung thư mục checkpoint, chạy `checkpoint_ready`,
   so `config_hash` với session hiện tại. Không subprocess, không tạo `step-<n>` mới,
   không ghi gì lên Drive.
2. **Chỉ khi `RUN_RESUME_PROOF = True`** (ô param ở mục 2, mặc định **`False`**): chạy
   thêm 2 optimizer step từ checkpoint và khẳng định số step tăng lên, in dòng
   `resumed ... at step N` của trainer. Bước này **làm thay đổi experiment**: nó in ra
   tên các `step-<n>` mới được tạo, sync chúng lên Drive, và ghi `resume_proof.json` với
   `"isolated": false` + `from_step` / `to_step` / `new_step_dirs`. Vì vậy nó không bao
   giờ chạy ngầm: một experiment đã xong không bị nối dài mà không ai yêu cầu.

**Chỉ load `adapter_model.safetensors` là "adapter-only loading"**, không phải resume
optimizer — notebook không bao giờ gọi đó là resume.

## 8. Export adapter

Mục 11 xuất:

```text
export/<RUN_NAME>-step<N>/
├── adapter/adapter_model.safetensors   # trọng số LoRA
├── adapter/adapter_config.json
├── adapter/tokenizer*.json, chat_template.jinja
├── effective_config.yaml               # cấu hình đã dùng
└── export_meta.json                    # model id, config hash, dataset hash, git commit, libs
```

Merge vào weights đầy đủ là **tuỳ chọn** (`MERGE_LORA = True`, cần ~20 GB VRAM
rảnh và ~9 GB disk). Mặc định giữ adapter: nhỏ, và repo merge lúc load.

Tải từ Drive về máy, hoặc copy sang máy khác.

## 9. Dùng adapter trong repo

```bash
# chấm offline trên split validation (teacher-forced)
python scripts/evaluate.py --validation-only \
  --checkpoint <đường-dẫn>/adapter --model-mode lora \
  --predictions outputs/datasets/train/val.jsonl \
  --config configs/training_lora.yaml

# sinh submission trên public_test (cần ASR + SDK chính thức)
python scripts/infer.py --model-mode lora --checkpoint <đường-dẫn>/adapter \
  --split public_test --data-root ./data --output submission.jsonl \
  --model Qwen/Qwen3.5-4B
```

Luôn truyền `--model Qwen/Qwen3.5-4B` (hoặc `model.name` trong config): repo mặc
định có thể là model khác (`openbmb/MiniCPM5-2B`), và adapter chỉ có nghĩa với base
model đã train nó.

Mục 12 của notebook là smoke test: load adapter → generate thật → đưa output qua
`parse_model_output` + validator + một episode agent (mock env). **Load thành công
không phải là đánh giá tool-calling thành công** — đó chỉ là kiểm tra đường ống.

## Ghi chú về độ tin cậy

* **TESTED** (chạy thật ở máy local): JSON notebook hợp lệ; mọi code cell `ast.parse`
  OK; mọi flag CLI notebook dùng có trong `--help` của `prepare_dataset.py`,
  `train.py`, `evaluate.py`, `export_model.py` tại commit hiện tại; thống kê
  truncation 47/4368 @8192 lấy từ `manifest.json` của repo; và — không phải trên
  notebook này — `models/MiniCPM5-2B` đã được load, forward/backward và generate thật
  trên GPU RTX 5070 Ti của máy này (README gốc).
* **TESTED ở mức code, không ở mức phần cứng**: `derive_run_config` / `estimate_vram_gb`
  / `revalidate_tier_caps` được gọi thật với input tier tổng hợp (T4/L4/A100-40/A100-80)
  từ `tests/test_colab_notebook.py` — tách hàm ra khỏi `.ipynb` bằng `ast`, không cần GPU.
  Đó là kiểm tra logic, không phải kiểm tra VRAM thật. Cùng file đó chạy thật
  `atomic_sync`/`verify_sync_marker`/`checkpoint_ready` trên thư mục tạm (stdlib), gồm
  cả một lần **giả lập crash** giữa hai lần rename, và `release_model_refs` với một
  "model" giả để chứng minh tham chiếu bị xoá thật (weakref chết) chứ không chỉ bị gán
  `None`.
* **TESTED ở mức code, không cài gì**: cổng phiên bản ở mục 1c (tách hàm bằng `ast` ra
  `/tmp`, thay `importlib.metadata.version` + `pip` bằng bản giả) — `transformers` 4.44.2
  với floor `>=5.22` báo `upgrade`, đúng `5.22` báo `none`, gói vắng báo `install`, và
  trường hợp vẫn dưới floor sau khi cài thì raise. Mục 1c gate theo **phiên bản** chứ
  không theo "đã cài hay chưa": floor khớp `pyproject.toml` (`transformers>=5.22`,
  `peft>=0.21`, `accelerate>=0.33`, `pyyaml>=6`). `torch` chỉ **kiểm tra** (dưới 2.15 thì
  bảo chọn runtime mới hơn), không pip cài, vì torch của Colab gắn với CUDA của runtime.
* **ESTIMATE**: mọi con số VRAM, thời gian, Compute Unit, và toàn bộ bảng cấu hình
  khuyến nghị. Chưa có lần chạy Colab nào. Các con số trong mục 5 (`13.5 / 19.8 / 14.8 /
  18.2 / 29.3 / 48.2 GB`) là output thật của `estimate_vram_gb` ở vocab **248320** với
  **cả hai** hạng tử logits của đường CE chunked (`logits_compute` bf16 `[seq, vocab]` +
  `logits_chunked` fp32 ở `loss_chunk_tokens=1024`), không phải số chép tay; bảng đầy đủ
  (kèm `lg_compute` / `lg_fp32_chunk` / `lg_unchunked`) do chính ô 2b in ra khi chạy.
* **Chưa kiểm chứng được**: load `Qwen/Qwen3.5-4B` trên GPU (không có weights ở máy
  này), hiệu quả của QLoRA (không có `bitsandbytes` ở đây), đường fp16 +
  `torch.amp.GradScaler` trên T4, chấm điểm closed-loop với `zaic_tools`, độ chính
  xác tool-calling sau train, và mọi lần chạy Colab (không truy cập được runtime).
