# Các điểm cần bạn xem xét — LifeMate

Cập nhật sau commit sửa pipeline `5dee7213219332b7fed036a403bd5c0e68ee2204`.

## Quyết định đang chờ

**Không có quyết định thiết kế nào cần bạn đưa ra để tiếp tục sửa các lỗi đã xác định.** Không nên quyết định rằng bài nộp đã sẵn sàng chỉ dựa trên bộ kiểm thử hiện tại: xác minh container, thời gian, và điểm chính thức vẫn chưa hoàn thành.

## Thông tin hoặc tài nguyên có thể cần bạn cung cấp

Đây là các điều kiện để xác minh, không phải lựa chọn thiết kế bắt buộc:

1. **Quy định và hướng dẫn nộp chính thức**: nếu cổng cuộc thi yêu cầu đăng nhập hoặc có tài liệu riêng, cần nội dung chính thức về giới hạn thời gian, GPU, tham số/context, Docker, offline và công thức điểm. Profile 30 giây trong repo chỉ là mục tiêu thận trọng; không được coi là quy định đã xác minh.
2. **Docker và môi trường GPU tương đương máy chấm**: cần Docker và, nếu có, NVIDIA A30 để build/chạy image, xác minh offline, ToolEnv, đầu ra và phân phối độ trễ. Hiện chưa có bằng chứng container hoặc A30.
3. **Mô hình tùy chọn**: cần checkpoint Gipformer 1.5 và backend phụ tương thích cùng runtime được xác minh nếu muốn đánh giá các thí nghiệm này. Không cần chúng để tiếp tục kiểm tra baseline Gipformer 68M.
4. **Smoke training MiniCPM**: lần thử model thật kết thúc mã 137; cần môi trường đủ tài nguyên và điều tra nguyên nhân trước khi xác nhận resume với model thật.

## Công việc kỹ thuật còn lại, không cần quyết định của bạn

- Hoàn thiện timeout thực sự và tính thời gian ASR trong evaluation; đo latency end-to-end.
- Hoàn thiện thống kê thực thi cho mọi feature, kiểm tra parity prompt inference/evaluation, kiểm thử cache khi checkpoint thay nội dung tại cùng đường dẫn.
- Xác minh công thức điểm và tính đúng final state bằng scorer chính thức khi có thể; không dùng LCS-F1 hay mock làm điểm chính thức.
- Chạy container, kiểm thử submission thật và kiểm tra các trường hợp private-test khi dữ liệu/môi trường được cung cấp.

Bằng chứng, kết quả kiểm thử và trạng thái từng cổng ở [báo cáo cuối](FINAL_COMPETITION_FIX_REPORT.vi.md). Các tệp chưa commit sẵn có trong workspace không nằm trong commit sửa pipeline, ngoại trừ phần cache đã được tích hợp như báo cáo giải thích.
