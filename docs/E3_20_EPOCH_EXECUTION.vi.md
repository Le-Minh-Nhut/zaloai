# E3-WARMSTART — tiếp tục tới 20 epoch

## Trạng thái khi ghi

Đã khởi chạy train GPU trong `tmux` `lifemate_e3_20epochs_20261115`, PID `761530`, source `d63ac52892c13fad419211331265a62a5978006b`. Log xác nhận nạp checkpoint E3 step 100, khôi phục optimizer, sampler và RNG; scheduler được dựng lại từ horizon 100 sang **13.260 step**. Vì lịch learning rate thay đổi, đây không phải quỹ đạo tối ưu giống run cũ, dù trạng thái optimizer/RNG/sampler được khôi phục đầy đủ.

Checkpoint nguồn: `outputs/checkpoints/lifemate_e3_warmstart/lora/step-100`. Bản sao chỉ đọc bảo vệ: `outputs/research/e3_20epochs_20261115/protected_step_100`. SHA256 adapter `70b91492e68441e541c7399dd150f1829b1c9ee7e1b0b2147ca5060f19aa2b52`; optimizer và training state của bản sao đã đối chiếu hash với nguồn. Parent E0 step 1614 giữ nguyên.

Dataset sau tokenizer/cân bằng có **5.298 exposure/epoch**; microbatch 1, accumulation 8, flush nhóm cuối → **663 optimizer step/epoch**, **20 epoch = 13.260 step tổng**, bao gồm 100 step đã chạy. Resume từ epoch 1, microbatch kế tiếp 800. Dataset manifest SHA256 `28358f4ee17c4bde41c732da2e3ad06b918c23080c8a8707bce4d2765cbeaced`, ASR artifact SHA256 `b219aaef1feab1bc2313994ca602d0ca3cf94c94caea2aaa57257bd092354b00`.

Cấu hình giữ nguyên base MiniCPM5-2B, LoRA r16/alpha32, BF16, LR `2e-5`, K=3, sampling, seed và context 8192. Chỉ mở rộng `num_train_epochs` và horizon; validation/saving đặt theo epoch. Các checkpoint trong `outputs/checkpoints/lifemate_e3_20epochs/lora/` không bị prune; còn khoảng 119 GiB trống trước khi chạy, đủ cho khoảng 20 checkpoint theo kích thước hiện có.

`status.json` được cập nhật bởi `lifemate_e3_20epoch_monitor_20261115`, gồm loss train mới nhất và checkpoint validation-loss tốt nhất trong các checkpoint đã ghi xong. Không dừng khi validation loss tăng. Phiên `lifemate_e3_20epoch_evaluation_queue_20261115` chỉ đánh giá E0/best/final với ToolEnv thật, replay bốn ASR K=3 trên 12 ID held-out **sau khi checkpoint step 13260 tồn tại**; không tự chạy thêm training. Kết quả tốt nhất/final và điểm ToolEnv hiện NOT TESTED, điểm chính thức UNAVAILABLE.

Lệnh thực tế: `outputs/research/e3_20epochs_20261115/run.sh`; cấu hình cùng thư mục. Kiểm tra:

```bash
R=/home/heheboiz/data/zaloai/outputs/research/e3_20epochs_20261115
tail -f "$R/train.log"
cat "$R/status.json"
tmux list-sessions
nvidia-smi
```

Regression cho mở rộng epoch/horizon: 2 bài PASS (bao gồm từ chối thay LR hoặc thiếu `--extend-horizon`). Không dùng `--best-effort-resume`. Khi cần phục hồi từ checkpoint epoch mới, giữ source commit, config, dataset, E0 adapter, sampling và horizon 13.260; không quay về checkpoint E0 hay step 0. Không chạy lệnh phục hồi khi PID hiện tại còn sống.
