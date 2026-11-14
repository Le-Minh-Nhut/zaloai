# Kiểm tra cuối trước khi chạy SFT

**TRAIN_READY.** Không chạy full training trong phiên này. Mã xuất phát `74a929f877fb66d92eb3a37e8126f376ab8eee00`; các thay đổi workspace có sẵn của người dùng được giữ ngoài commit.

- `SFTTrainer.evaluate()` dùng chung lựa chọn target-only logits với training, giữ token trước target để dịch causal chính xác. Model không hỗ trợ `logits_to_keep` giữ đường logits đầy đủ. Test Llama nhỏ có padding xác nhận loss, accuracy và số token bằng đường đầy đủ.
- Validation MiniCPM5-2B thật, BF16, LoRA rank 16 trên **record validation dài nhất được giữ: 8.192 token**: 477 record hợp lệ, 5 record bị loại theo fit; target dự kiến và đo được đều 37 token; loss `0.318383`, accuracy `0.945946`, exit 0. Peak `torch.cuda.max_memory_allocated()` **5.792.846.848 byte**, reserved **6.268.387.328 byte**, tính cả model đã nạp. Không gặp OOM. Lệnh kiểm tra cục bộ: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src /tmp/cudaenv/bin/python /tmp/lifemate_longest_validation.py`.
- `--save-steps 100` chuyển strategy sang `steps`; test mô phỏng bước 100/200/300 xác nhận cadence và retention hai checkpoint mới nhất. Probe resume thực với `--save-steps 100` dùng checkpoint bước 1 của dataset thật, khôi phục exact horizon 1614/RNG/cursor, tiếp tục bước 2 và ghi checkpoint (exit 0, loss 0,18030547). Config probe chỉ tắt validation để giới hạn thời gian; các tham số tối ưu hóa giống baseline. Không chạy 100 bước GPU chỉ để kiểm tra lịch lưu.
- Hash train/val/manifest/stats khớp `docs/FINAL_TRAIN_READINESS_GATE.vi.md`; 0 nhóm train/validation giao nhau, 0 ID ngoài train. Dataset hiện có 4.368/482 record thô; 4.298/477 được giữ sau fit. MiniCPM weights cục bộ có sẵn, model/LoRA/config gốc giữ nguyên, feature thử nghiệm OFF, đích checkpoint sản xuất chưa tồn tại. GPU còn khoảng 14,5 GiB trước probe.
- Kiểm thử tập trung: 5 passed, 71 deselected. Full regression `WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q`: 975 passed, 1 skipped, 1 warning, 29,66 giây; bao gồm các thay đổi workspace sẵn có. Không có training blocker đã quan sát; đây không phải xác nhận submission.

Lệnh khởi chạy:

```bash
set -o pipefail
mkdir -p outputs
[ ! -e outputs/checkpoints/lifemate_baseline ] && \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/train.py \
  --config configs/training_lora.yaml \
  --dataset-dir outputs/datasets/minicpm_baseline/train \
  --model models/MiniCPM5-2B \
  --output-dir outputs/checkpoints/lifemate_baseline \
  --save-steps 100 \
  2>&1 | tee outputs/lifemate_baseline_train.log
```

Resume bằng đúng config/cadence và checkpoint hiện có; ví dụ `step-100` chỉ có sau khi tạo:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
/tmp/cudaenv/bin/python scripts/train.py \
  --config configs/training_lora.yaml \
  --dataset-dir outputs/datasets/minicpm_baseline/train \
  --model models/MiniCPM5-2B \
  --output-dir outputs/checkpoints/lifemate_baseline \
  --save-steps 100 \
  --resume outputs/checkpoints/lifemate_baseline/lora/step-100
```

Không truyền `--planned-steps` hay `--extend-horizon`; horizon được suy ra hoặc khôi phục là 1.614 bước. Theo dõi bằng `watch -n 2 nvidia-smi` và `tail -f outputs/lifemate_baseline_train.log`. Commit chứa báo cáo xác định bằng `git log -1 --format=%H -- docs/FINAL_TRAIN_LAUNCH_CHECK.vi.md`; đối chiếu remote bằng `git ls-remote origin refs/heads/main`.
