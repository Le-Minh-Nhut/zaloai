# Thực thi E2 với ASR thật — 15/11/2026

Tại thời điểm ghi, E2 đang ở bước smoke và validation; không khẳng định run E2 đầy đủ đã hoàn tất. E3 chưa chạy do ba model ASR còn lại chưa được xác minh có sẵn. Không có điểm chính thức hoặc kết quả task-level.

- Source tạo transcript: `1e8ad53698eae3b62e34c44afee0a010d18594a4`. Source tạo dataset/train: nhánh `codex/e2-real-execution`, `888d83970eafc34f1c4c1427a828ecbf7f41f780`. Không thay source của job baseline hoặc ghi đè checkpoint cũ.
- Gipformer 68M int8 CPU: smoke 3/3 audio train; sau đó thực sự giải mã 5.036/5.036 lượt, gồm 1.545 lượt hiện tại và 3.491 lượt lịch sử. Thời gian backend trung bình 68,55 ms/lượt, tối đa 295 ms trong artifact; đây không phải benchmark end-to-end. Phiên smoke 3 mẫu báo trung bình 0,088 giây/mẫu.
- Artifact: `outputs/research/e2_gipformer_20261115/train_top1.jsonl`; SHA256 `d061c99f1c914ea215e3d48b7289d44c2387b38e2babd27738f26dc5d4b6e98d`. Manifest cùng tên có hậu tố `.manifest.json`.
- Dataset: `outputs/research/e2_gipformer_20261115/dataset/train`. 4.368 decision point sạch và 140 biến thể ASR được giữ, 482 record validation trước lọc tokenizer. Coverage: SUPPORTED 140, UNKNOWN 4.142, UNSUPPORTED 86; các trường hợp UNKNOWN/UNSUPPORTED không được đưa vào bản tăng cường. Manifest SHA256 `0c7c942b906a87840e2562c0e52e23475d6aabe13fc2150291d6b5651218ae6c`. Group hash `53a8c3536ac6eee4ed245f0fb33c06a54585e9e4919cd6b13f6492ae9d2f4a9e` khớp baseline. Model dùng tokenizer MiniCPM thật, max 8192; tokenizer báo 70 train và 5 validation record không fit target, 492 train và 41 validation prompt cần cắt giữa. Trainer giữ 4.438 train record sau lọc.
- Giới hạn khoa học: nhiều biến thể E2 được giữ nhờ ASR tương đương transcript sạch sau chuẩn hóa bề mặt bảo thủ; chỉ 140/4.368 decision point sạch có thêm biến thể. Chưa chứng minh cải thiện với lỗi ASR hoặc Multi-ASR thực.
- Một step optimizer MiniCPM5-2B BF16 LoRA thật đã tạo `outputs/research/e2_gipformer_20261115/smoke/lora/step-1`, gồm adapter, optimizer, RNG và trạng thái sampler. Đây là smoke, không phải kết quả so sánh. Smoke dùng horizon 1.692 tính theo số record trước lọc; không resume smoke sang run chính. Run chính tự tính horizon theo số record đã token hóa, dự kiến 1.665; cần xác nhận log/state thực tế.
- Hàng đợi `lifemate_e2_training_queue_20261115` chờ smoke/validation kết thúc; sau đó kiểm tra GPU và chạy E2 độc lập vào `outputs/checkpoints/lifemate_e2`, không resume từ baseline hoặc smoke. Nếu lỗi tiền điều kiện, script dừng; không tự động tuyên bố thành công.
- Log: `outputs/research/e2_gipformer_20261115/asr.log` và `pipeline.log`; script và config thực tế cùng thư mục. Xem `tmux ls`, `tail -f outputs/research/e2_gipformer_20261115/pipeline.log`, `nvidia-smi` để theo dõi. Không dừng run đang khỏe.
- Kiểm thử sửa coverage: 27 bài CPU PASS; chưa thay cho đánh giá model thật. Chưa chạy held-out ToolEnv, chưa có điểm chính thức, chưa xác minh submission.

Lệnh train chính được hàng đợi thực thi sau các kiểm tra (không chạy trùng nếu đã có tiến trình):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
/tmp/cudaenv/bin/python -u scripts/train.py \
  --config /home/heheboiz/data/zaloai/outputs/research/e2_gipformer_20261115/config.yaml \
  --dataset-dir /home/heheboiz/data/zaloai/outputs/research/e2_gipformer_20261115/dataset/train \
  --model /home/heheboiz/data/zaloai/models/MiniCPM5-2B \
  --output-dir /home/heheboiz/data/zaloai/outputs/checkpoints/lifemate_e2 \
  --save-steps 100
```

## Cập nhật lúc 16:41 GMT+7

Smoke hoàn tất sau 331,1 giây, 1 optimizer step, loss train `0,1168001`; validation teacher-forced trên 477 record giữ lại cho loss `0,306475`, token accuracy `0,933605` trên 16.447 token mục tiêu. Đây chưa phải so sánh task-level với baseline. Chưa kiểm tra nạp lại checkpoint smoke.

Run E2 chính đã thực sự bắt đầu lúc `2026-11-15 16:40:35 +07:00`, PID `689649`, qua `tmux` `lifemate_e2_training_queue_20261115`. GPU RTX 5070 Ti; checkpoint đầu của run chính chưa có lúc cập nhật. Log chứa dòng `planned_steps=1692` từ smoke tính trên số record trước lọc; lệnh train chính không truyền `--planned-steps` và tự tính theo 4.438 record sau lọc, dự kiến 1.665 step. Kiểm tra `training_state.json` của checkpoint chính khi có để xác nhận thực tế. Trạng thái và số bước tiếp tục thay đổi sau thời điểm báo cáo này; hãy xem log và tiến trình thực tế.
