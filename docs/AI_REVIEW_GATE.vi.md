# AI Review Gate — phiên bản thử nghiệm cho job tương lai

Không áp dụng hoặc hot-patch phiên baseline hiện tại. Runner dùng gate chỉ khi cấu hình `ai_review_gate.enabled=true`, như trong `configs/overnight_gated.json`. Trước job và sau khi job kết thúc, runner chờ `codex exec` chỉ đọc, schema xác định, timeout tối đa và lưu hồ sơ trong `<work_dir>/ai_gate/`. Reviewer lỗi, timeout, dữ liệu không đủ, kết luận BUG/BLOCKED/INCONCLUSIVE hoặc action không hợp lệ đều chặn nhánh phụ thuộc. Kết quả đánh giá kém không tự biến thành lỗi implementation. Không thực thi shell command từ response của model.

Gate gắn review với SHA nguồn, cấu hình, held-out, model/dataset và băm dataset/checkpoint trên command; kiểm tra lại artifact sau review. Checkpoint training phải được review xác nhận; retry training chỉ dùng checkpoint exact-resume tương thích horizon, tối đa một lần. GPU lock và kiểm tra tài nguyên vẫn do runner hiện có thực hiện. Review hoàn chỉnh được tái sử dụng khi danh tính hoàn toàn khớp. Lock một reviewer; record RUNNING được phục hồi có giới hạn khi reviewer cũ không còn. Nếu review bị từ chối, runner dừng `REVIEW_BLOCKED`, không mặc định cho phép; có thể điều tra record rồi dùng `--resume --retry-review <event-id>` để lưu record cũ vào `ai_gate/audit` và yêu cầu review mới.

## Đã kiểm thử và giới hạn

Các kiểm thử tổng hợp dùng process nhỏ, không GPU: thứ tự job và gate, review thành công, model underperformance, sai verdict/action, timeout, record dở dang và retry, bảo toàn identity, từ chối khi dataset đổi, runner dừng trước training nếu reviewer không kết luận được. Chưa chạy gated pipeline trên trọng số MiniCPM thật, chưa chạy review Codex thật trong gate mới, và chưa có bằng chứng khả năng review thực tế sẽ luôn đủ để quyết định. Supervisor Codex đang chạy hiện nay độc lập và vẫn chỉ tư vấn.

`scripts/ai_repair.py` có thể gọi Codex `workspace-write` tại worktree mới chỉ khi record có BUG_CONFIRMED; tắt GPU qua `CUDA_VISIBLE_DEVICES`, yêu cầu test regression được sửa/thêm và chạy suite cố định. Nó **chưa tự commit, chuyển runner sang commit sửa, hay tự chạy lại stage**; không được trình bày là khôi phục tự động hoàn chỉnh. Việc áp dụng repair và đánh giá sự tương đương khoa học cần kiểm tra sau khi tests qua. Những action ngoài allowlist gate bị chặn thay vì tự suy đoán hoặc tự thực thi.

## Tiếp tục từ baseline đã hoàn thành

Chỉ thực hiện khi worker cũ và toàn bộ child đã dừng, baseline có trạng thái COMPLETED/exit 0, và có candidate đã bảo vệ. Trong checkout commit gated sạch, nối các tài nguyên hiện có (thay đường dẫn nếu khác):

```bash
ln -s /home/heheboiz/data/zaloai/data data
ln -s /home/heheboiz/data/zaloai/models models
ln -s /home/heheboiz/data/zaloai/outputs outputs
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/import_completed_baseline.py \
  --old-run /home/heheboiz/data/zaloai/outputs/overnight \
  --config configs/overnight_gated.json \
  --stop-at 2026-11-16T23:30:00+07:00
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py \
  --config configs/overnight_gated.json --resume
```

Dùng thời điểm dừng tương lai phù hợp. Lệnh import từ chối khi worker cũ còn chạy, kiểm tra dataset/model/held-out/horizon và băm candidate, sao chép adapter sang run riêng, giữ baseline là COMPLETED. Job đánh giá được chạy lại dưới source mới; baseline không được train lại. Chức năng import đã được kiểm thử tổng hợp, chưa vận hành với run thật. Không chạy hai worker GPU đồng thời và không chỉnh các file của run cũ.
