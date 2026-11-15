# Multi-ASR Top-1 bốn model

Đây là chuẩn bị mã, **chưa xác minh tương thích thực tế với bốn checkpoint**. Không chạy inference thật hoặc download weights trong lượt triển khai.

- Gipformer 68M: dùng backend có sẵn; baseline khi tắt feature giữ đường cũ.
- Gipformer 1.5: tên ONNX int8/fp32, tokens, config 16 kHz và revision `dd9227dcd8705c13f33bdbe59728d546ab94480f` đối chiếu [upstream](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt/tree/main). Thiếu weights ở môi trường này; chưa load/decode thật. Thư mục mặc định theo model ID, không rơi về weights 68M.
- PhoWhisper-small: dùng Whisper backend hiện có, greedy Top-1; cần weights/tokenizer local. Chưa chạy thật trong lượt này.
- ChunkFormer RNNT: adapter riêng theo [Python API upstream](https://huggingface.co/khanhld/chunkformer-rnnt-large-vie), gọi `ChunkFormerModel.from_pretrained(local_dir)` và `endless_decode(..., return_timestamps=False)`. Cần package `chunkformer`, các dependency upstream và `config.yaml`, `global_cmvn`, `pytorch_model.pt`, `vocab.txt` local. Chưa cài, load hoặc kiểm tra tương thích torch/transformers thực tế. Không tự download.

Danh sách `features.asr.multi_asr_models` chọn một đến bốn model theo thứ tự ưu tiên. Mỗi model cho một transcript thật; kết quả thô, latency, lỗi và cấu hình được giữ riêng. Sau đó gộp transcript tương đương, giữ provenance, tối đa ba ứng viên cho prompt. Các ứng viên này không được gọi là decoder N-best; không so sánh điểm âm học giữa model. Không bật đồng thời decoder N-best và chế độ danh sách Top-1 mới. Legacy primary/secondary vẫn dùng đường cũ.

RMS/VAD xử lý một lần trước các backend, không sửa audio gốc. ITN tác động lên text đưa vào prompt và có cả raw/normalized trong metadata. Cache transcript thô theo model dùng SHA-256 audio đầu vào, identity model/artifact, decoder và sample ID; đổi audio/tiền xử lý, artifact hoặc decoder tạo key mới. Cache tổng hợp có cả metadata cho evaluation. `transcript_records_by_model` tạo record từng model tương thích loader Noisy-ASR; chưa tạo dataset hay label.

Chạy **sau này**, sau khi chuẩn bị weights offline và chọn checkpoint LLM, ví dụ giới hạn 5 mẫu:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python scripts/infer.py --config configs/multi_asr_four_top1.yaml \
  --split public_test --data-root data --model models/MiniCPM5-2B \
  --model-mode lora --checkpoint outputs/overnight/candidates/step-1614 \
  --override agent.time_budget_seconds=180 --limit 5 \
  --output outputs/asr/four_top1_probe.jsonl
```

Đây là cấu hình đo thử, không phải xác nhận tuân thủ thời gian cuộc thi. `multi_asr_budget_seconds` chỉ kiểm tra giữa các backend và ghi nhận overrun sau decode: nó **không ngắt cứng một lần decode đang chạy**. CPU mặc định tránh bốn model tranh VRAM, nhưng thời gian/RAM/VRAM thực tế, offline loading, accuracy/final-state và ảnh hưởng prompt cần đo trên weights thật. Lỗi/missing model được ghi; policy `strict` dừng, `primary_only` cho phép kết quả từng nguồn còn lại và đánh dấu degraded.

Kiểm thử CPU mock/synthetic: nhóm multi-ASR/feature/spec 164 passed, 1 deselected; nhóm Gipformer 30 passed, 1 deselected. Hai test bỏ qua phụ thuộc weights baseline trong worktree tách biệt. Không suy ra độ chính xác hay khả năng chạy thật từ mock.
