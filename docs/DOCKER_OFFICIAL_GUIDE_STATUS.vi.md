# Đối chiếu Docker Guide LifeMate — 15/11/2026

Nguồn chính thức: [Submission Guidance](https://challenge.zalo.ai/portal/lifemate) và [Docker Guide](https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf), được lấy trực tiếp từ đường dẫn ở Submission Guidance. Guide yêu cầu offline; `/code/predict.sh` đọc `/data/<split>`, dùng `zaic_tools` của split, ghi `/result/submission.jsonl` với `id`, `tool_calls`, `state_hash`, `time` (ms); Jupyter và notebook ghi `/result/jupyter_submission.jsonl`; mã/data huấn luyện và README; weights inference trong image. Guide mô tả GPU RTX 5090 32 GB, driver 595.58.03 và CUDA không cao hơn 13.2; một đoạn khác nói bắt buộc CUDA 13.2, cần làm rõ với BTC.

## Đã thực hiện

- `docker/predict.sh` chọn split và gọi entrypoint hiện có; `docker/export_timed_submission.py` xuất trường `time` từ latency thực tế trong journal đã kiểm checksum, từ chối thời gian thiếu hoặc kết quả khác journal. Cấu trúc mặc định của inference hiện tại được giữ nguyên.
- `docker/Dockerfile.competition` thêm giao diện `/code`, Jupyter, mã training và bundle mô hình đã chuẩn bị sẵn; model inference không được tải khi chạy. `docker/start_jupyter.sh` và notebook có sẵn.
- Kiểm thử exporter bảo đảm không đổi call, state hash, thứ tự và từ chối journal hỏng. Khói SDK chính thức dùng một ToolEnv train và `python -m zaic_tools ... --validate`: 1 dòng, 0 lỗi schema. Đây không phải kiểm thử chất lượng agent.
- Thông số `time` đo từ trước khi xử lý history/audio đến sau lấy kết quả; thời gian load model và ghi journal/output toàn cục không tính vào từng mẫu.

## Chưa được xác minh / blocker

- Máy không có lệnh Docker; chưa build, chạy offline, thử GPU, hoặc đối chiếu kết quả inference/notebook. Không tuyên bố submission ready.
- Chưa có adapter đã huấn luyện được chọn; chưa tạo `submission_models/` chứa base, adapter, ASR. Phải kiểm tra license, model identity và bảo đảm weights dùng thực sự nằm trong image.
- Notebook hiện gọi inference trong subprocess: các bước load model và đọc/infer không được trình bày tách biệt như ví dụ guide. Phải điều chỉnh/kiểm chứng notebook thực tế trước khi nộp; không xem notebook này đã đạt yêu cầu đo thời gian.
- README training chỉ mô tả nguồn tổng quát; phải bổ sung URL bất biến và fingerprint chính xác của dataset/model, checkpoint được chọn và lệnh tái lập trước deadline.
- Runner đang chạy bản commit `42789c3` ở worktree riêng, nên candidate JSONL của runner chưa tự có `time`. Không thay source đang huấn luyện; cần xuất lại có kiểm chứng hoặc chạy candidate qua image mới sau khi huấn luyện.
- Hướng dẫn yêu cầu notebook và predict cho điểm giống nhau; chưa có bằng chứng. Việc SDK validate chỉ kiểm schema tool, không xác minh state hash, chất lượng, thời gian hay tính đúng của lời giải.

## Chuẩn bị khi có artifact và Docker

Tạo `submission_models/base`, `submission_models/adapter`, `submission_models/asr` từ artifacts đã được chọn và xác minh; không commit chúng. Sau đó:

```bash
docker build -f docker/Dockerfile -t lifemate-runtime:latest .
docker build -f docker/Dockerfile.competition -t lifemate-competition:local .
docker run --rm --gpus all --network none -v "$PWD/data:/data:ro" -v "$PWD/outputs/docker-result:/result" lifemate-competition:local
```

Các lệnh trên là kế hoạch chưa chạy; chỉ mount split phù hợp và bảo đảm `/result` ghi được. Kiểm thêm đúng sample coverage, SDK validator, notebook run-all, kết quả tương đương, offline và độ trễ thực tế trước khi nộp.
