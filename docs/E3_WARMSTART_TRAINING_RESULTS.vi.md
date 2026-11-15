# E3-WARMSTART — kết quả thực thi

## Trạng thái tại thời điểm ghi

Run GPU thật đang hoạt động; chưa hoàn tất 100 step và chưa có kết quả E3 ToolEnv. Không khẳng định E3 cải thiện chất lượng. Tiến trình chính PID `748868`, phiên `tmux` `lifemate_e3_warmstart_20261115`, bắt đầu `2026-11-15 21:48:29 +07`. Smoke GPU độc lập đã chạy một optimizer step và ghi checkpoint `outputs/research/e3_multiasr_warmstart_20261115/smoke/lora/step-1`. Trong smoke, loss microbatch cuối `0,00244653`; so sánh với parent cho thấy cùng 588 tensor LoRA, 588 tensor có thay đổi sau update, max delta `6,672e-6`. Chưa có validation loss của smoke.

Run chính đã ghi ít nhất step 5; loss microbatch cuối tại step 5 là `0,001069`, 10/40 exposure đầu là conditioned. Số này thay đổi theo thời gian và không phải chất lượng task. Kiểm tra `outputs/checkpoints/lifemate_e3_warmstart/lora/metrics.jsonl` để biết tiến độ mới. Step 25/50/100 chưa có tại thời điểm ghi. Đánh giá được xếp trong `tmux` `lifemate_e3_eval_queue_20261115`, chỉ chạy sau khi training dừng và có step 25; không tự train tiếp.

## Dữ liệu

Nguồn bốn ASR SHA256 `b219aaef1feab1bc2313994ca602d0ca3cf94c94caea2aaa57257bd092354b00`; K=3, ITN tắt. Dataset `outputs/research/e3_multiasr_warmstart_20261115/dataset/train`, manifest SHA256 `28358f4ee17c4bde41c732da2e3ad06b918c23080c8a8707bce4d2765cbeaced`.

- 4.368 record sạch, 201 conditioned; SUPPORTED 201, UNKNOWN 4.155, UNSUPPORTED 12; UNKNOWN và UNSUPPORTED không được đưa vào augmentation.
- 482 record validation trước lọc; group train/val không giao nhau, cùng phân hoạch baseline. Tokenizer MiniCPM thật: 497/4.569 prompt train bị cắt giữa, 71 record train bị loại vì target không còn hợp lệ; validation tương ứng 41/482 và 5 bị loại. Trong 201 conditioned có 200 record giữ được, tất cả 200 kiểm tra mask target PASS; một record bị loại. Không giữ target bị cắt.
- Sampling lặp trong bộ nhớ tối đa 5 lần/record conditioned, giữ mọi record sạch; 804 exposure bổ sung, tỷ lệ kế hoạch 18,70% trước lọc, thấp hơn mục tiêu 20% để giữ giới hạn lặp. Metrics ghi exposure thực theo từng step. Không tạo target mới.

## Warm-start và cấu hình

Base `openbmb/MiniCPM5-2B`, adapter cha `outputs/overnight/candidates/step-1614`, SHA256 `789eb8389cb6e79ea0705cbf2ee616e7e29dcf6f39a50c7898cf38370c183860`, step cha 1614. Đây là E3-WARMSTART, không phải E3 khởi tạo độc lập từ base. `PeftModel.from_pretrained(..., is_trainable=True)` gắn một adapter sẵn có; base bị freeze; 25.116.672 tham số LoRA trainable. Optimizer/scheduler mới, không `--resume` E0. Source training `3179c629492a48b8bc6d646de03818888e5f6328`.

BF16, rank 16/alpha 32, microbatch 1, accumulation 8, sequence 8192, gradient checkpointing; learning rate `2e-5` (thấp hơn baseline để giảm mức thay đổi khi warm-start, chưa được chứng minh tối ưu), cosine horizon 100, hard stop 100, checkpoint mỗi 25 bước, giữ tối đa 4 checkpoint. Validation teacher-forced trong trainer tắt để tránh delay; evaluation ToolEnv riêng sau training. Không tự kéo dài run.

## Lệnh tái hiện

Dựng dataset vào đường dẫn **mới**:

```bash
cd /tmp/lifemate-e2-e3-pilot
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/prepare_dataset.py \
 --data-root /home/heheboiz/data/zaloai/data --split train \
 --config /home/heheboiz/data/zaloai/outputs/research/e3_multiasr_warmstart_20261115/config.yaml \
 --conditioned-multi-records /home/heheboiz/data/zaloai/outputs/research/e3_four_asr_full_20261115/four_asr_train.jsonl \
 --conditioned-mode multi --output /home/heheboiz/data/zaloai/outputs/research/e3_warmstart_rebuild_check --tokenizer /home/heheboiz/data/zaloai/models/MiniCPM5-2B --max-seq-length 8192
```

Lệnh run chính và kiểm tra smoke thực tế nằm tại `outputs/research/e3_multiasr_warmstart_20261115/run.sh`; evaluation tuần tự tại `.../evaluate_after.sh`. Không chạy lại khi PID hiện tại còn sống hoặc output đã tồn tại. Source code được cố định tại commit training nêu trên; nếu resume, dùng đúng source, config, dataset, parent và sampling identity của checkpoint, không resume E0.

```bash
tail -f /home/heheboiz/data/zaloai/outputs/checkpoints/lifemate_e3_warmstart/lora/metrics.jsonl
tail -f /home/heheboiz/data/zaloai/outputs/research/e3_multiasr_warmstart_20261115/training.log
tmux list-sessions
```

So sánh đã có trước đó: E0 4/12 exact match tool-call ở bốn ASR K=3 trên ToolEnv thật, nhưng run mới sẽ đánh giá lại E0 cùng cấu hình E3 trước khi so sánh. E3 accuracy/tool-name/latency: NOT TESTED tại thời điểm ghi; điểm chính thức UNAVAILABLE. Kết quả 12 mẫu là diagnostic, không phải leaderboard. Regression code: 27 tests warm-start/resume/loss PASS; warm-start riêng 2 tests PASS.

## Cập nhật checkpoint step 25

Run chính đã hoàn tất **25/100 optimizer step** và tạo `outputs/checkpoints/lifemate_e3_warmstart/lora/step-25`. Đã kiểm tra các file adapter, optimizer và `training_state.json`: `step=25`, scheduler horizon `100`, parent adapter SHA256 `789eb8389cb6e79ea0705cbf2ee616e7e29dcf6f39a50c7898cf38370c183860`, ASR artifact SHA256 `b219aaef1feab1bc2313994ca602d0ca3cf94c94caea2aaa57257bd092354b00`, source commit `3179c629492a48b8bc6d646de03818888e5f6328`. State ghi `exact_resume=true`, nhưng **chưa thử reload/resume checkpoint này**. Parent E0 vẫn giữ nguyên SHA256.

Tại step 25, log: loss microbatch cuối `0,004836`, 43 conditioned/200 exposure (**21,5%**). Loss không phải task metric. Training vẫn chạy; step 50/100 và evaluation chưa có kết quả tại thời điểm ghi. `watch_run.py` chạy riêng, kiểm tra định kỳ SHA dataset/ASR/parent, step, file checkpoint và GPU contention; nếu phát hiện vi phạm, nó gửi SIGTERM đúng PID training. Log: `outputs/research/e3_multiasr_warmstart_20261115/watch.log`. Sau training, `evaluate_after.sh` và `summarize.py` chỉ đánh giá và tổng hợp; không tự kéo dài training.


Kiểm tra bổ sung step 25: `optimizer_state.pt` có 588 mục optimizer, optimizer step 25 và scheduler `last_epoch=25`. Đây là kiểm tra cấu trúc/trạng thái trên CPU, chưa chứng minh reload inference hay exact resume thực tế.

Để tái hiện **một run độc lập mới** sau khi GPU rảnh, dùng đúng source commit `3179c629492a48b8bc6d646de03818888e5f6328`, chọn output chưa tồn tại và chạy:

```bash
cd /tmp/lifemate-e3-warmstart
R=/home/heheboiz/data/zaloai
W="$R/outputs/research/e3_multiasr_warmstart_20261115"
[ ! -e "$R/outputs/checkpoints/lifemate_e3_warmstart_repro" ] && \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
/tmp/cudaenv/bin/python scripts/train.py --config "$W/config.yaml" --dataset-dir "$W/dataset/train" \
 --model "$R/models/MiniCPM5-2B" --init-adapter "$R/outputs/overnight/candidates/step-1614" \
 --conditioned-exposure-ratio 0.2 --planned-steps 100 --max-steps 100 --save-steps 25 \
 --output-dir "$R/outputs/checkpoints/lifemate_e3_warmstart_repro"
```

Ví dụ nạp và đánh giá checkpoint step 25 sau khi GPU rảnh (output phải chưa tồn tại):

```bash
cd /tmp/lifemate-e3-warmstart
R=/home/heheboiz/data/zaloai
W="$R/outputs/research/e3_multiasr_warmstart_20261115"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/evaluate.py \
 --config "$W/config.yaml" --data-root "$R/data" --split train \
 --heldout-dataset "$R/outputs/datasets/minicpm_baseline/train" --model "$R/models/MiniCPM5-2B" \
 --model-mode lora --checkpoint "$R/outputs/checkpoints/lifemate_e3_warmstart/lora/step-25" \
 --evaluation-mode audio_to_action --closed-loop --replay-multi-asr \
 "$R/outputs/research/e3_four_asr_full_20261115/four_asr_train.jsonl" --replay-k 3 --limit 12 \
 --output "$W/manual_eval_25"
```
