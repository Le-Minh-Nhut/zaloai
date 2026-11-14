# Báo cáo sửa đường nộp LifeMate

## A–B. Kết quả và trạng thái repository

Bắt đầu từ `770e3f25b261ba1944dfcbc4beaded8af918cec4`, nhánh `main`, trùng `origin/main`. Đã sửa kiểm tra loại checkpoint, cách cô lập lỗi history, chế độ đo runtime không cache, thống kê thực thi feature, nhận diện thay đổi tham số ablation, JSON debug và nhận dạng artifact ASR trong cache. Đã chạy một mẫu thật qua audio, Gipformer, MiniCPM + adapter smoke và SDK chính thức; agent không hoàn tất yêu cầu. **Chưa thể kết luận submission sẵn sàng.**

Có các thay đổi của người dùng đã tồn tại trước phiên này, gồm `src/lifemate/evaluation/metrics.py` và các tệp/thư mục chưa theo dõi; chúng được giữ ngoài commit. SHA commit chứa báo cáo này lấy bằng `git log -1 --format=%H -- docs/FINAL_SUBMISSION_READINESS_REPAIR.vi.md`; SHA đầu nhánh lấy bằng `git rev-parse HEAD`, và remote bằng `git ls-remote origin refs/heads/main`. Không tự ghi SHA của chính commit vào nội dung commit vì SHA phụ thuộc chính nội dung đó.

## C. Tài liệu chính thức

Đã đối chiếu `data/README.md`, SDK trong `data/*/zaic_tools/README.md`, và [trang LifeMate](https://challenge.zalo.ai/portal/lifemate). Dataset quy định một ToolEnv mỗi mẫu, history user có audio, state đã phản ánh các thao tác trước, installed apps giới hạn tool, và output gồm `id`, `tool_calls`, `state_hash` từ env. Tool sai bị từ chối; thiếu thông tin có `sys.ask_missing_info`; yêu cầu ngoài phạm vi không hành động. Các tab quy định/nộp của cổng không đọc được đầy đủ trong phiên: giới hạn thời gian 30 giây, A30, tham số, context, offline, mount và công thức điểm **chưa được xác minh độc lập**. Cấu hình 30 giây chỉ là mục tiêu thận trọng. [Upstream Gipformer](https://github.com/ggroup-ai-lab/gipformer/blob/main/README.md) liệt kê checkpoint 1.5; spec/artifact runtime ở repo này vẫn chưa được xác minh.

## D–E. Phân loại và thay đổi

| ID | Mức độ / loại | Nguyên nhân và sửa thực tế | Bằng chứng / trạng thái |
|---|---|---|---|
| CKPT | P0 / CONFIRMED BUG | `scripts/infer.py` từ chối thư mục thiếu adapter hoặc adapter-only ở chế độ full trước khi tải model; `docker/entrypoint.sh` kiểm tra tương tự. | Test sai loại/thiếu checkpoint; adapter thực nạp qua inference. Full checkpoint thật và container chưa thử. |
| HIST | P0 / CONFIRMED BUG | Một lỗi audio history làm dừng cả tập. `data/history.py` ghi lỗi cục bộ; inference tạo hàng không gọi tool từ env cho mẫu thiếu history, cảnh báo và tiếp tục. Đường dẫn bị cấm vẫn lỗi cứng. | Test 10 mẫu, lỗi mẫu 4, đủ 10 hàng và hash mock env; test exception cấm. Chính sách no-action an toàn, không bảo đảm đúng ý người dùng. |
| TIME | P0 / DESIGN GAP | `evaluate.py --end-to-end-runtime` chạy history và ASR chính trong deadline từng mẫu; báo chất lượng cache ghi rõ loại trừ latency ASR. Agent kiểm tra trước generate và sau generate; inference không bắt đầu ASR chính sau deadline. | Test ASR trễ và đo một mẫu thật. `max_time` hợp tác, không ngắt cứng ASR/kernel; timing chưa gồm đầy đủ ghi file và chưa tách LLM/tool/prompt. |
| FEAT | P1 / CONFIRMED BUG | `evaluate.py` đếm requested/executed/degraded/blocked/not-observed cho VAD, RMS, 1.5, N-best, Multi-ASR dựa trên metadata; ablation không đồng nhất thiếu quan sát với degraded và không tính delta hợp lệ khi thiếu bằng chứng. | Test metadata mô phỏng; checkpoint 1.5 và secondary model thật chưa thử. |
| ABL | P1 / CONFIRMED BUG | `feature_ablation.py` so sánh thêm identity ASR/audio, phát hiện RMS/VAD và decoder/model thay đổi dù tập feature không đổi. | Test RMS −23 → −18 bắt buộc audio-to-action. Fairness hiện hữu giữ nguyên. |
| DEBUG | P1 / CONFIRMED BUG | `_sanitize` xử lý dataclass, mapping, list, tuple có giới hạn độ sâu, loại khóa nội dung cá nhân. | Test ASR hypothesis và nested metadata JSON. |
| CACHE | P1 / CONFIRMED BUG | Cache v3 thêm thống kê size/mtime/ctime các artifact cục bộ và các đường dẫn backend đã resolve, cùng backend thực/override; quét một lần mỗi cache key; giữ hash audio và tham số. Cache cũ bị vô hiệu. | Test thay file tại cùng đường dẫn. Đây là fingerprint metadata, không phải hash nội dung mật mã; artifact hub bị sửa tại chỗ bên ngoài thư mục cấu hình chưa được chứng minh bao phủ. |

Đã giữ các sửa trước về hypothesis, Multi-ASR, ITN và lựa chọn 1.5; test hồi quy vẫn chạy. 1.5 bị từ chối khi spec chưa xác minh, không âm thầm dùng 68M. N-best thật của runtime chưa chứng minh. Không sửa trainer hoặc notebook.

## F. MiniCPM training và exit 137

Lệnh chính xác và PID của lần 137 trước không còn được lưu trong báo cáo cũ, nên **không thể đối chiếu chắc chắn**. `journalctl -k` cho thấy ngày 14/11 lúc 20:56:26 kernel giết Python PID 456697 trong cgroup terminal hiện tại do **global OOM**, anonymous RSS 25.313.960 KiB; `memory.events` của cgroup ghi `oom_kill 1`, `memory.max=max`. Đây là bằng chứng OOM của một tiến trình cùng phiên, không đủ chứng minh PID đó chính là lệnh smoke trước. `dmesg` không có quyền đọc. Hiện máy có 31 GiB RAM, RTX 5070 Ti 16 GiB; số liệu không chứng minh tình trạng tại thời điểm cũ.

Smoke mới dùng `/tmp/cudaenv/bin/python` + transformers/PEFT, weights `models/MiniCPM5-2B` offline, BF16 trên CUDA, LoRA rank 4 tại q/v, batch 1 và câu ngắn: forward/backward/AdamW/scheduler, lưu adapter và state, giải phóng model, nạp lại, chạy bước thứ hai. LR khôi phục `9e-5`, scheduler epoch 1, sau bước tiếp `8.1e-5`; peak VRAM allocated 5.169.621.504 byte, peak RSS khoảng 5.586.124 KiB ở lần đầu, exit 0. Có giới hạn giám sát 12 GiB RSS và 240 giây; đây là script PEFT độc lập, **không phải** kiểm thử resume trainer thật của repo. Adapter từ smoke này được dùng thử pipeline, không phải checkpoint đã huấn luyện cạnh tranh. Không thay cấu hình huấn luyện mặc định; exact-resume tiny được kiểm tra riêng.

Ngoài ra đã chạy **trainer thật của repo** với dữ liệu tổng hợp 8 train/2 val, rank 4, sequence tối đa 256, batch 1, accumulation 1, gradient checkpointing, offline: `scripts/train.py --config /tmp/lifemate_repo_smoke.yaml --dataset-dir /tmp/lifemate_repo_smoke_data --model models/MiniCPM5-2B --output-dir /tmp/lifemate_repo_smoke_output --max-steps 1 --planned-steps 2 --save-steps 1 --seq-len 256`; exit 0, tạo `lora/step-1`, RSS cao nhất 2.025.928 KiB. Chạy lại cùng lệnh với `--resume /tmp/lifemate_repo_smoke_output/lora/step-1 --max-steps 2`: báo exact, khôi phục scheduler horizon 2 và RNG, hoàn thành step 2, RSS cao nhất 2.053.516 KiB. Không đo peak VRAM riêng cho lần trainer repo; so sánh bit-exact real-model chưa chạy. Các config/dữ liệu/checkpoint smoke nằm ngoài repo và không được commit.

## G. Hợp đồng checkpoint và lệnh nộp LoRA

`base` dùng base model; `full` dùng thư mục trọng số full; `lora` dùng model gốc cộng thư mục adapter PEFT chứa `adapter_config.json` và trọng số adapter. Tokenizer được ưu tiên từ checkpoint nếu có, nếu không từ base; profile lấy từ tên model. Ví dụ, thay hai đường dẫn bằng artifact thật đã được huấn luyện và kiểm tra, không dùng adapter smoke:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python scripts/infer.py --config configs/competition_inference.yaml \
  --split public_test --data-root data --model models/MiniCPM5-2B \
  --model-mode lora --checkpoint /models/trained-lora \
  --asr gipformer --output outputs/submission.jsonl
```

Docker entrypoint phải chỉ rõ `MODEL_MODE=lora`, `MODEL_NAME` trỏ base cục bộ và `CHECKPOINT` trỏ adapter; mặc định Docker vẫn là full, không tự suy đoán artifact. Chưa xác minh container/offline mounts.

## H–I. History và runtime

History âm thanh hợp lệ chưa có text được phiên âm theo thứ tự; text có sẵn giữ nguyên, không dùng nhãn private. Một audio thiếu/hỏng/rỗng hoặc hết thời gian dẫn tới no-action cho mẫu đó và cảnh báo; các mẫu khác tiếp tục. Vi phạm đường dẫn không được hạ cấp. `--allow-missing-history` chỉ dành chẩn đoán.

Đo `--evaluation-mode audio_to_action --end-to-end-runtime` trên **một mẫu public không có history**, RTX 5070 Ti, adapter smoke: history 0,000071 s, ASR hiện tại 0,06644 s, tổng 10,60677 s, timeout 0. Vì n=1, mean/P50/P95/max đều là giá trị duy nhất, không có ý nghĩa phân phối. LLM, ToolEnv và prompt riêng: UNAVAILABLE. Chưa đo A30, nhiều history, cold start của mô hình trong ngân sách mỗi mẫu hoặc ghi output đầy đủ. Không tuyên bố tuân thủ giới hạn chính thức.

## J. Evaluation và ablation

Chế độ chất lượng có thể dùng cache và ghi `quality_asr_latency_excluded`; chế độ `--end-to-end-runtime` không dùng cache transcript hiện tại, ghi `end_to_end_uncached`. Thời gian đo có hạn chế nêu trên. Số đếm feature dựa trên thông tin thực thi, không dựa đơn thuần trên cờ. Kết quả vắng dữ liệu là NOT OBSERVED, không phải thất bại được quan sát. ASR/audio thay đổi tham số buộc audio-to-action; metrics không có vẫn unavailable. LCS-F1 chỉ là chẩn đoán, không phải điểm chính thức; final-state score chính thức chưa khả dụng.

## K–L. Docker, ToolEnv và submission

`bash -n docker/entrypoint.sh` PASS. Docker executable không có: build BLOCKED, container BLOCKED, offline container NOT VERIFIED. Lệnh inference thật dùng một sample public, ASR thật, base MiniCPM và adapter smoke PEFT, ToolEnv chính thức: exit 0, một hàng, agent `repeated_failure` với 6 calls. Lặp lại với adapter `step-2` do trainer repo tạo: exit 0, agent `max_steps` với 8 calls, vòng mẫu 21,1 giây; SDK cũng báo `1 lines checked, 0 schema errors`. Adapter huấn luyện hai bước tổng hợp không đại diện model cạnh tranh. `PYTHONPATH=data/public_test /tmp/cudaenv/bin/python -m zaic_tools --split public_test --root data --validate /tmp/lifemate_real_submission.jsonl` trả `1 lines checked, 0 schema errors`. Kiểm tra schema một dòng không chứng minh đủ cả split, hành động đúng, final-state đúng hoặc điểm leaderboard.

## M. Kiểm thử

- Targeted: `.venv/bin/python -m pytest -q tests/test_submission_readiness_repairs.py tests/test_competition_pipeline_fixes.py tests/test_inference.py`: 48 passed trước test feature bổ sung; bộ test mới riêng: 9 passed. Lần kiểm tra sau sửa checkpoint: 44 passed cho hai file được chọn.
- Full `.venv` với `WANDB_BASE_URL=http://127.0.0.1:9`: 954 passed, 16 skipped, 13,50 giây; bao gồm các thay đổi người dùng sẵn có trong workspace.
- CUDA `/tmp/cudaenv/bin/python -m pytest -q`: 969 passed, 1 skipped, 1 warning, 27,90 giây sau sửa cuối.
- `scripts/verify_training_resume.py --tiny --steps 12 --interrupt-at 5 --accum 3` với `/tmp/cudaenv`: EQUIVALENT, weights/optimizer/scheduler/RNG/sample order/loss giống nhau.
- `scripts/verify_all.py --no-write --gates A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,R,P`: các gate của script PASS; phần lớn là fixture/mock, không là chứng nhận cuộc thi.
- `compileall`, `bash -n`, `git diff --check`: PASS. Số kết quả full CUDA ở phần bàn giao được cập nhật theo lần chạy thực tế.

## N–O. Blocker và mức sẵn sàng

**Có thể tạo submission thực bằng model đã huấn luyện? NOT VERIFIED.** Đường đi một mẫu với adapter thử nghiệm đã hoạt động, nhưng chưa có bằng chứng adapter SFT hoàn chỉnh, độ đúng hành động, toàn bộ split, Docker/offline và hardware chấm. Cần checkpoint huấn luyện thực, Docker và tài liệu chính thức; so sánh bit-exact real-model, A30, điểm chính thức và model 1.5 còn BLOCKED/NOT TESTED. Không có quyết định thiết kế nào đang chặn các sửa đã thực hiện.

## P. Gate bàn giao

| Gate | Kết quả | Gate | Kết quả | Gate | Kết quả | Gate | Kết quả |
|---|---|---|---|---|---|---|---|
| G01 | PASS (tài liệu khả dụng; còn thiếu quy định) | G02 | PASS (test) | G03 | PASS (điều tra; chưa gắn PID) | G04 | PASS (smoke trainer thật; chưa so sánh bit-exact) |
| G05 | NOT TESTED (full thật) | G06 | PASS (lựa chọn LoRA tường minh) | G07 | BLOCKED (container) | G08 | PASS (test) |
| G09 | PASS (10 mẫu mock) | G10 | PASS (test) | G11 | BLOCKED (timeout hợp tác) | G12 | BLOCKED (chưa đủ các giai đoạn) |
| G13 | PASS | G14 | PASS (agent test) | G15 | PASS (metadata test) | G16 | PASS (metadata test; 1.5 thật blocked) |
| G17 | PASS | G18 | PASS | G19 | PASS | G20 | BLOCKED (artifact hub mutable chưa bao phủ) |
| G21 | PASS (từ chối spec chưa xác minh) | G22 | PASS | G23 | PASS (regression) | G24 | PASS |
| G25 | PASS (một mẫu SDK thật) | G26 | PASS (schema một mẫu) | G27 | BLOCKED (Docker) | G28 | PASS (quét và test) |
| G29 | PASS | G30 | PASS (suite khả dụng) | G31 | PASS (diff review) | G32 | NOT TESTED (xác minh sau push bằng Git) |

Các gate PASS ở mức code/test không mở rộng thành khẳng định chất lượng hay hiệu năng trên bộ chấm. Danh sách file và commit chính xác kiểm tra bằng `git show --stat --oneline HEAD`; trạng thái push kiểm tra bằng `git ls-remote`.
