# Codex Research Supervisor

Supervisor quan sát `outputs/overnight/state.json` định kỳ, tạo event bền vững và chỉ gọi `codex exec --sandbox read-only --ephemeral` khi có sự kiện mới. Mỗi review có schema JSON, timeout 300 giây, tối đa hai lần thử, log/exit code, file lock một supervisor và không chạy lệnh do model đề xuất. Event chạy dở sau sự cố được phục hồi; PID/start-time tránh chạy reviewer trùng đang còn sống. Training và runner hiện tại không bị sửa, không bị khởi động lại. Review có thể bị giới hạn bởi namespace sandbox; phải kiểm tra PID trực tiếp trên host trước khi kết luận process mất.

Event: `BEFORE_TRAIN` hồi cứu khi phát hiện job đã bắt đầu; `AFTER_TRAIN`, `AFTER_EVAL`, `SUBMISSION_CHECK`, `ON_ANOMALY` cho lỗi, loss không hữu hạn hoặc bước không tiến triển trên 30 phút. Kết quả, log, queue: `outputs/overnight/codex_supervisor/`. Báo cáo buổi sáng liên kết nội dung `CODEX_REVIEWS.vi.md`. Không có truy cập/đọc hidden state có chủ ý, không tự nộp bài.

## Giới hạn quan trọng

**Đây là supervisor chỉ tư vấn, không phải hệ thống tự quyết hoàn chỉnh.** Worker hiện tại đang chạy mã commit `42789c3` trong worktree độc lập. Không có cơ chế đồng bộ giữa các job và supervisor, nên không thể chặn BEFORE_TRAIN, thay đổi thứ tự thí nghiệm, áp dụng quyết định hay tự sửa/re-evaluate an toàn trong phiên hiện tại. Runner tiếp tục theo logic xác định cũ; không tuyên bố gate AI đã bảo vệ các job. Khả năng phát hiện anomaly phụ thuộc các trường telemetry hiện có và trạng thái job; không phát hiện mọi gradient NaN/OOM/IO trước khi runner ghi nhận lỗi. Review không khởi chạy inference GPU hay khẳng định tính hợp lệ checkpoint nếu thiếu bằng chứng. Sửa lỗi phải được thực hiện trong worktree tách biệt, test, commit và dùng cho job tương lai. Đây là phần còn thiếu để đạt yêu cầu tự động điều khiển; không đánh dấu hoàn tất.

Sau khi runner hiện tại dừng, tích hợp một handshake có timeout, nguồn dữ liệu và commit cố định, kiểm thử phục hồi và giới hạn hành động trước khi cho phép AI quyết định. Không thay đổi training đang hoạt động để thêm handshake.

## Vận hành

Supervisor đã được khởi động dưới dạng process độc lập (xem `process.json` và kiểm tra PID thực tế), không phụ thuộc phiên Codex. Khởi động lại nếu process đã dừng:

```bash
nohup /tmp/cudaenv/bin/python scripts/codex_research_supervisor.py > outputs/overnight/codex_supervisor.log 2>&1 < /dev/null &
```

Lock ngăn hai supervisor cùng chạy. Nếu Codex không khả dụng, sự kiện được đánh dấu thất bại sau retry; runner vẫn tiếp tục. Không sửa state của runner bằng tay.
