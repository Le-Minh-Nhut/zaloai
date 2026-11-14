# Báo cáo bàn giao LifeMate

## Phạm vi và kết quả

Theo chỉ dẫn cuối của người dùng, các thay đổi thật sự quan trọng và ảnh hưởng lớn được tạm hoãn để báo cáo; nhiệm vụ sửa toàn bộ pipeline chưa hoàn tất. Đã sửa một lỗi nhỏ ở cấu hình backend ASR phụ và chạy bộ kiểm thử hiện có. Workspace đã có thay đổi chưa commit trước khi bắt đầu; không quy các thay đổi đó cho lần thực hiện này.

## Nguồn chính thức

[Trang LifeMate của ban tổ chức](https://challenge.zalo.ai/portal/lifemate) mô tả agent nhận tiếng Việt, ngữ cảnh và ứng dụng đã kết nối, gọi công cụ, hỏi thông tin còn thiếu hoặc không hành động khi không có ứng dụng phù hợp. Nội dung chi tiết các tab không trích xuất được trong phiên này; chưa xác minh độc lập giới hạn thời gian, phần cứng, kích thước mô hình, quy tắc Docker, điểm số hoặc các ràng buộc khác. [README Gipformer upstream](https://github.com/ggroup-ai-lab/gipformer/blob/main/README.md) liệt kê `g-group-ai-lab/gipformer1.5-68M-rnnt`.

## Lỗi đã sửa

- **ASR-SECONDARY**, P0 theo yêu cầu: `_compose_multi_asr` truyền `model_name` cho mọi backend, trong khi Gipformer dùng `model_id`. `scripts/infer.py` chọn tên tham số theo backend; `tests/test_multi_asr.py` kiểm tra cả Whisper và Gipformer bằng backend giả. Kiểm thử đạt; chưa tải checkpoint hoặc xác minh suy luận thật.

## Vấn đề tác động lớn đang hoãn

- History audio chưa được xác minh có đường xử lý hoàn chỉnh cho private test; không khẳng định submission hợp lệ.
- Deadline toàn pipeline, ngắt generation, ToolEnv và độ trễ A30 chưa được xác minh thực tế.
- N-best và Multi-ASR vẫn có nhánh `if/elif` trong `scripts/infer.py`; chưa sửa tương tác hoặc bảo đảm hypotheses tới evaluation. Không tuyên bố vấn đề này đã giải quyết.
- Chưa xác minh đầy đủ cache, trạng thái feature, ITN, parity history, chế độ evaluation và ablation trong workspace hiện tại. `src/lifemate/evaluation/transcripts.py` và `metrics.py` đã có thay đổi từ trước và không được chỉnh sửa trong phiên này.
- Không xác nhận công thức điểm chính thức; LCS-F1 chỉ nên được xem là chỉ số chẩn đoán cho tới khi có đặc tả chính thức.
- Chưa xác minh baseline MiniCPM LoRA, Qwen LoRA/QLoRA, resume thực tế hoặc noisy-ASR bằng mô hình thật trong phiên này.
- Chưa xác minh các cổng G01–G29 một cách đầy đủ; không nên coi bộ kiểm thử giả lập là bằng chứng đạt các cổng runtime/submission.

## Kiểm thử

Trong môi trường `.venv`, với `WANDB_BASE_URL=http://127.0.0.1:9`:

- `.venv/bin/python -m pytest -q tests/test_multi_asr.py`: **19 passed**.
- `.venv/bin/python -m pytest -q`: **940 passed, 16 skipped**, 0 failed, 13.71 giây. Bộ này chạy trên workspace có sẵn thay đổi của người dùng, bao gồm các file chưa commit.
- `python3 -m py_compile scripts/infer.py tests/test_multi_asr.py`: thành công.
- `git diff --check`: thành công tại thời điểm kiểm tra.

Docker không có trong PATH: build và chạy container **BLOCKED**; offline inference, ToolEnv thật, định dạng submission và thời gian chạy **NOT TESTED**. Không có đo đạc A30 hay xác minh mô hình thật trong phiên này. Không khẳng định sẵn sàng nộp bài.

## Git

HEAD ban đầu và `origin/main` đều là `7037f1dfe6bdf4b3322bd4e58151caf759331203`. Commit cục bộ và push được báo trong phần bàn giao của cuộc hội thoại; báo cáo này không tự ghi SHA của chính commit chứa nó. Các thay đổi sẵn có được giữ lại và không đưa vào commit của phiên này.
