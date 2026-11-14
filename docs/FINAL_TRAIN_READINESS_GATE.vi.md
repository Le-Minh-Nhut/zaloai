# Cổng sẵn sàng huấn luyện MiniCPM LifeMate

## A. Kết luận

**TRAIN_READY** cho baseline SFT + LoRA trên máy hiện tại, với lệnh ở mục K. Không chạy full training trong lần kiểm tra này. Sẵn sàng huấn luyện không đồng nghĩa sẵn sàng nộp bài hoặc bảo đảm chất lượng mô hình.

## B. Trạng thái ban đầu

Nhánh `main`, commit đầu `668c2283b3659ba962368a405a837b9e5a23a96a`, trùng `origin/main`. Workspace đã có sửa `src/lifemate/evaluation/metrics.py` và các tệp chưa theo dõi `agy-9router/`, `configs/inference_submission.yaml`, `docs/HISTORY_AUDIO.md`, `docs/OFFICIAL_REQUIREMENTS_CHECK.md`, `docs/RUNTIME_BUDGET.md`, `docs/SCORING.md`, `tests/test_p1_hardening.py`, `work/`; giữ nguyên và không commit. Dataset/model/checkpoint thử nghiệm không commit.

## C. Lỗi đã sửa

| ID | Nguyên nhân và sửa | Kiểm thử |
|---|---|---|
| TRAIN-OOM | Baseline 8192 token bị OOM ở backward: LM head vẫn tạo logits cho toàn prompt dù nhãn prompt đều `-100`. `training/trainer.py` dùng `logits_to_keep` khi kiến trúc hỗ trợ, giữ token ngay trước target để giữ đúng causal shift; model khác vẫn dùng đường cũ. Không đổi rank, LR, chiều dài hoặc objective. | `tests/test_train_target_logits.py`: so sánh loss và gradient với logits đầy đủ; chạy real 8192 sau sửa. |
| TRAIN-FINITE | Clip gradient trước đây nuốt mọi exception; BF16 có thể tiếp tục với loss/gradient không hữu hạn. Nay từ chối loss không hữu hạn trước backward và gradient không hữu hạn trước optimizer, giữ cơ chế GradScaler cho FP16. | Test gradient NaN bị từ chối; real BF16 588 gradient hữu hạn. |
| ASR-EMPTY | ASR lượt hiện tại thất bại vẫn có thể gọi agent. `scripts/infer.py` ghi hàng no-action từ env, không chạy agent, tiếp tục mẫu khác. | Test hai mẫu và smoke ToolEnv chính thức với lỗi ASR được tiêm: 2 hàng, agent chỉ chạy mẫu hợp lệ. |
| EVAL-HISTORY | Lỗi history tạo hash rỗng. `evaluation/evaluator.py` tạo agent/env một lần, lấy calls/hash từ env, ghi lỗi và không chạy agent; lỗi đường dẫn cấm vẫn ném ra. | Test history lỗi và env hash. |

Dataset cũ `outputs/datasets/train` dùng tokenizer Qwen nên không dùng làm bằng chứng cho MiniCPM; tạo bộ riêng dưới đây. N-best, ASR cache, trainer resume và prompt builder đã có sẵn được giữ nguyên ngoài các sửa có căn cứ trên.

## D. Dataset

Lệnh chuẩn bị ở K chỉ đọc split `data/train`, không dùng nhãn public/private hay trực tiếp đọc hidden state. Split builder nhóm theo hash request/history trước khi chia, và builder tạo prompt từ các bước đã xảy ra trước target, có guard final-state và test chống future observation. Hai lần chạy chuẩn bị cùng seed 42 cho bốn file giống byte theo SHA-256.

- 1.545 source sample: 1.389 train, 156 validation; 1.544 nhóm, giao nhóm/ID giữa hai split = 0.
- File train có 4.368 decision point, validation 482. Trainer giữ tương ứng **4.298 / 477** sau fit bằng tokenizer MiniCPM. Builder bỏ 17 / 2 decision point `unobservable_ref_target`; bước fit bỏ thêm 70 / 5 vì `target_value_truncated`.
- Fit ghi 488/4.368 prompt train (11,17%) và 41/482 validation (8,51%) bị cắt giữa; các số này tính trên tập trước khi loại fit. Target được giữ nguyên; cắt prompt vẫn có thể làm mất ngữ cảnh khác ngoài các giá trị tham chiếu được kiểm tra, nên đây là giới hạn chất lượng được ghi nhận.
- Quét toàn bộ 4.850 record bằng `parse_action` và `parse_model_output` với catalog/installed apps theo sample: target rỗng 0, parser lỗi 0, schema lỗi 0, ID ngoài nguồn train 0. Train 3.170 call/1.198 finish; val 352 call/130 finish. Quét 4.775 record được giữ: lỗi mask/target token/terminator/attention 0.
- SHA-256: manifest `123f4abcc3e3e965807d4f87da45cca18e912fc2831b4c924b086c9fce311fa2`; train JSONL `0162f94aaea87e09f84bac066bbf28bd714b721b7e02071fb9539f3b595a61cf`; val JSONL `633ed31fd0ee34efa97975ddb9b609a156cf0130cd61920a4913e0b5e966bd2f`; group hash `53a8c3536ac6eee4ed245f0fb33c06a54585e9e4919cd6b13f6492ae9d2f4a9e`.

## E–F. Tokenizer, loss và cấu hình hiệu lực

Tokenizer cục bộ `models/MiniCPM5-2B`, chat template MiniCPM, max 8192. Collator mask prompt và padding `-100`, supervise đủ action và terminator; target quá dài hoặc mất giá trị tham chiếu bị loại có đếm, không cắt target. Prompt train và inference dùng cùng `build_decision_prompt`; parser train target dùng cùng protocol/registry như inference.

Model gốc `openbmb/MiniCPM5-2B` qua đường dẫn cục bộ `models/MiniCPM5-2B`; SHA-256 file weights `14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c`, config `59613157e5357d62bcb121e04cb90f2170b43d789b142380c12c3d4277d95180`. BF16, CUDA, AdamW, cosine, LR `2e-4`, weight decay `0.01`, warmup `0.03`, 3 epoch, seed 42, clip 1.0. LoRA rank 16, alpha 32, dropout 0.05, target `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`; 25.116.672 tham số trainable / 2.541.873.152 tổng. Batch 1 × accumulation 8 = effective batch 8, gradient checkpointing ON, tính loss target chunk 1024. Các feature thử nghiệm OFF. Số optimizer step theo số record thực giữ: `ceil(4298 / 8) × 3 = 1614`.

## G. GPU probe thật

RTX 5070 Ti 16 GiB, ~31 GiB RAM hệ thống; các process desktop đang chiếm khoảng 1,4 GiB VRAM. Probe đầu với cấu hình nguyên gốc và một record thực dài **8192 token**, lặp 8 microbatch, OOM backward, exit 1 (PyTorch đã cấp phát 11,39 GiB, còn 1,96 GiB; yêu cầu thêm 1,99 GiB). Đây là bằng chứng cho sửa TRAIN-OOM, không phải CUDA OOM sau sửa.

Sau sửa, lệnh lõi (monitor ngoài dừng nếu RSS vượt 12 GiB hoặc quá 600 giây):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/train.py --config configs/training_lora.yaml \
  --dataset-dir /tmp/lifemate_long_realdata --model models/MiniCPM5-2B \
  --output-dir /tmp/lifemate_final_probe --max-steps 1 --planned-steps 2
```

`/tmp/lifemate_long_realdata` chứa 8 bản của một record train thật dài 8192 và một validation record thật, chỉ dùng thử nghiệm. Bước 1 forward/backward/optimizer hoàn tất, loss `0.05911946`, validation loss `0.036672`, exit 0, peak RSS 2.167.844 KiB, peak VRAM process 11.890 MiB. Lần resume 8192 có instrument ghi peak torch allocated 8.611.201.024 byte, reserved 12.062.818.304 byte, 588 gradient hữu hạn, 0 tham số base trainable, LoRA đổi sau optimizer; loss `0.04117418`.

Probe bổ sung chạy **toàn bộ dataset chuẩn bị thật** nhưng dừng sau một optimizer step; config tạm chỉ đổi eval sang `no` và save sang `steps` để tránh đánh giá toàn validation trong smoke. Các thông số tối ưu hóa/model không đổi. Step 1 loss `0.33898777`, RSS peak 3.561.624 KiB, VRAM process peak 9.142 MiB, exit 0. Đây là bounded smoke, không phải huấn luyện đầy đủ. Chỉ số preflight sau khi model đã nạp có thể báo `does_not_fit`: ước lượng cộng trọng số model vào số free đã trừ model, và còn giả định full logits; phép đo thật nêu trên đáng tin hơn cho cấu hình đã thử. Không có bảo đảm nếu GPU bị process khác chiếm thêm bộ nhớ.

## H. Checkpoint và resume

Checkpoint thật `/tmp/lifemate_full_dataset_probe/lora/step-1` chứa `adapter_config.json`, `adapter_model.safetensors`, `optimizer_state.pt`, tokenizer, `lifemate_meta.json`, `training_state.json` (scheduler/RNG/sampler). Dữ liệu huấn luyện được fingerprint group và thứ tự token. Resume với cùng config smoke/dataset, `--resume .../step-1 --max-steps 2`, khôi phục exact từ bước 1, epoch 1, microbatch kế tiếp 8, scheduler horizon **1614**, RNG Python/CPU/CUDA; hoàn tất bước 2, loss `0.18030547`, 588 gradient hữu hạn, 0 base trainable, một tensor LoRA thực sự đổi, LR `1.25e-05`, scheduler epoch 2; RSS peak 4.047.860 KiB, torch CUDA allocated 8.522.014.208 byte, reserved 9.336.520.704 byte, exit 0. Probe đầy đủ dùng `/tmp/lifemate_full_probe.yaml` chỉ khác cadence đánh giá/lưu so với baseline, không phải checkpoint sản xuất.

`PYTHONPATH=src /tmp/cudaenv/bin/python scripts/verify_training_resume.py --tiny --steps 12 --interrupt-at 5 --accum 3`: EQUIVALENT; weights, optimizer, scheduler, RNG, sample order và loss giống nhau. Bit-exact hai run MiniCPM đầy đủ chưa thử, không cần thiết cho smoke resume này.

## I. Còn lại

**Training blockers:** không có blocker được quan sát theo các gate dưới đây. Cần theo dõi thực tế vì việc kiểm tra hữu hạn không bảo đảm không gặp OOM/NaN về sau; full multi-epoch chưa chạy.

**Submission verification gaps:** Docker, A30, full private-test, điểm chính thức và chất lượng final state chưa được xác minh; không chặn SFT. Chưa tải checkpoint ASR tùy chọn hoặc chạy benchmark ASR.

## J. Bằng chứng T01–T32

Ký hiệu lệnh: `P` = lệnh prepare ở K chạy hai lần; `A` = `/tmp/lifemate_train_audit.py` quét toàn bộ file với tokenizer thật, parser, registry và `build_supervised_example`; `L` = lệnh probe dài ở G; `R` = probe/resume full dataset ở H; `T` = lệnh tiny resume ở H; `S` = full pytest ở K; `D` = dry-run ở K; `C` = `git diff --check`, quét staged và source guard trong suite. Các script `/tmp` là artifact cục bộ để kiểm chứng, không phải dữ liệu dùng nộp.

| Gate | Status | Lệnh / kết quả thực tế |
|---|---|---|
| T01 | PASS | P/A: split train, 1.545 sample hợp lệ. |
| T02 | PASS | P/A: 0 nhóm/ID giao nhau, seed 42. |
| T03 | PASS | A: 4.850 target, 0 lỗi schema/target rỗng. |
| T04 | PASS | P/S: builder chỉ append quan sát sau khi sinh target; guard final state và test future observation. |
| T05 | PASS | A: 0 target parser/registry invalid. |
| T06 | PASS | A: 0 prompt mask sai. |
| T07 | PASS | A: 0 target label gap, token ID hoặc terminator sai. |
| T08 | PASS | A: 0 target bị cắt trong 4.775 record giữ. |
| T09 | PASS | P/A: counts khớp manifest; 4 hash trùng hai lần chuẩn bị. |
| T10 | PASS | P/A: tokenizer MiniCPM cục bộ; manifest nhận diện đúng. |
| T11 | PASS | D/L/R: r16, alpha32, dropout .05, 7 suffix. |
| T12 | PASS | R: 0 nonadapter trainable. |
| T13 | PASS | R: 25.116.672 trainable; 588 tensor có gradient. |
| T14 | PASS | L/R: forward real model/data thành công. |
| T15 | PASS | L/R: loss hữu hạn. |
| T16 | PASS | L/R: backward hoàn tất. |
| T17 | PASS | R: 588 gradient, nonfinite 0. |
| T18 | PASS | R: weight_changed=True sau optimizer. |
| T19 | PASS | R: scheduler 1 → 2, horizon 1614. |
| T20 | PASS | L: 8192 token, rank16, accum8 hoàn thành; peak process 11.890 MiB. |
| T21 | PASS | L/R sau sửa: exit 0; OOM trước sửa được nêu tại G. |
| T22 | PASS | R: adapter, optimizer và training state tại step-1. |
| T23 | PASS | R: checkpoint được load, exact. |
| T24 | PASS | R: resume step 1 → 2. |
| T25 | PASS | R: optimizer/scheduler khôi phục và step 2 cập nhật; scheduler horizon 1614. |
| T26 | PASS | R/T: RNG khôi phục, cursor 8; tiny test xác nhận thứ tự và RNG bit-identical. |
| T27 | PASS | T: EQUIVALENT. |
| T28 | PASS | S/L: cờ mặc định OFF, model/config baseline giữ nguyên, so sánh loss/gradient. |
| T29 | PASS | `pytest -q tests/test_train_target_logits.py tests/test_submission_readiness_repairs.py`: 13 passed. |
| T30 | PASS | S: 973 passed, 1 skipped, 1 warning. |
| T31 | PASS | D/L/R: CLI, đường dẫn, cấu hình và horizon đã kiểm tra; full multi-epoch chưa chạy. |
| T32 | PASS | C: diff check, source compliance suite; không stage dataset, weights, checkpoint hoặc credentials. |

## K. Lệnh thực thi

Chuẩn bị dataset (bộ hiện đã tồn tại; chạy lại cùng đầu vào cho hash ở D):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/prepare_dataset.py --data-root ./data --split train \
  --output outputs/datasets/minicpm_baseline --tokenizer models/MiniCPM5-2B --max-seq-length 8192
```

Dry-run thực tế:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/train.py --config configs/training_lora.yaml \
  --dataset-dir outputs/datasets/minicpm_baseline/train --model models/MiniCPM5-2B \
  --output-dir outputs/checkpoints/lifemate_baseline --dry-run
```

**Huấn luyện đầy đủ — người dùng tự chạy**, chỉ khi thư mục đích chưa tồn tại:

```bash
set -o pipefail
[ ! -e outputs/checkpoints/lifemate_baseline ] && \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/train.py --config configs/training_lora.yaml \
  --dataset-dir outputs/datasets/minicpm_baseline/train --model models/MiniCPM5-2B \
  --output-dir outputs/checkpoints/lifemate_baseline 2>&1 | tee outputs/lifemate_baseline_train.log
```

Resume đúng checkpoint đã lưu, không đổi horizon (điền đường dẫn step có thật):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/train.py --config configs/training_lora.yaml \
  --dataset-dir outputs/datasets/minicpm_baseline/train --model models/MiniCPM5-2B \
  --output-dir outputs/checkpoints/lifemate_baseline \
  --resume outputs/checkpoints/lifemate_baseline/lora/step-538
```

`step-538` là checkpoint kỳ vọng sau epoch đầu, không tồn tại trước khi training tạo ra; chọn đúng step hiện có nếu ngắt ở bước khác. Không thêm `--planned-steps`, `--extend-horizon` hoặc `--best-effort-resume` vào resume bình thường. Theo dõi bằng `watch -n 2 nvidia-smi` và `tail -f outputs/lifemate_baseline_train.log`. Cấu trúc dự kiến `outputs/checkpoints/lifemate_baseline/lora/step-<n>/` gồm adapter, optimizer state, tokenizer, metadata và training state.

Full regression đã chạy sau sửa mã:

```bash
WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q
```

Kết quả 973 passed, 1 skipped, 1 warning; bao gồm những thay đổi người dùng tồn tại sẵn trong workspace. Không suy diễn kết quả đó thành điểm cuộc thi.

## L. Git handover

Commit mã và báo cáo được xác định bằng `git log -1 --format=%H -- docs/FINAL_TRAIN_READINESS_GATE.vi.md`; kiểm tra HEAD/remote bằng `git rev-parse HEAD` và `git ls-remote origin refs/heads/main`. Những file được commit: trainer, inference/evaluation, regression tests và báo cáo này; weights, dataset và checkpoint không thuộc commit.
