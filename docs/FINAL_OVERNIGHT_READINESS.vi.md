# Bàn giao runner qua đêm

**BASELINE_READY: PASS ở mức preflight và kiểm thử điều phối tổng hợp.** Lệnh thật, dataset, model và cấu hình đã được preflight; chưa chạy một lượt baseline đầy đủ trong phiên này. **OVERNIGHT_READY: NOT TESTED ở mức chạy pipeline thật**: các stage đã được nối và kiểm thử bằng job tổng hợp, nhưng chưa có checkpoint baseline mới để chứng minh chuỗi đánh giá và submission thật. Không khởi chạy full training trước lệnh GO.

## Phạm vi và sửa lỗi

- Bắt đầu tại `9bdca519fa169b194fc112d1dccfb9ce7dceaf19`, nhánh `main`. Thay đổi có sẵn của người dùng trong `src/lifemate/evaluation/metrics.py` và các file untracked được giữ nguyên, không đưa vào commit. Worker chạy từ git worktree tách biệt tại SHA đã ghi, liên kết tới data, model và outputs; không dùng thay đổi chưa commit.
- CLI worker nằm trong nhóm argparse hợp lệ; nguyên nhân cũ khiến không chạy là khóa release. Bỏ khóa, thêm handshake xác nhận PID đã vào worker, khóa launch/worker/GPU, xác minh danh tính process và từ chối chạy trùng.
- Horizon lấy từ số record fit, batch, accumulation, epoch của từng dataset; checkpoint không tương thích bị từ chối. Nếu dừng, resume chọn checkpoint hợp lệ cùng horizon, không tự chạy lại từ bước 0. Lưu checkpoint mỗi 100 bước.
- Candidate bảo vệ là bản **adapter-only**, copy atomic và kiểm SHA-256, không giả làm checkpoint exact-resume. Giữ các adapter mới gần đây và candidate tốt; checkpoint optimizer để trainer quản lý riêng. Kiểm tra dung lượng trước khi ghi.
- Candidate so sánh trên cùng ID held-out gốc, dùng checkpoint LoRA thật trong lệnh evaluate. Episode model thất bại nhận điểm 0; lỗi hạ tầng, thiếu ID, kết quả hỏng hoặc sai thứ tự làm đánh giá vô hiệu. Exact-call là diagnostic cục bộ; **official score UNAVAILABLE**, không phải độ đúng final state.
- Nối submission public-test với Gipformer, MiniCPM LoRA và official ToolEnv; file tên riêng, journal hỗ trợ resume và kiểm schema/thứ tự/coverage trước khi hoàn tất. Đánh giá audio lỗi ASR không gọi agent với transcript rỗng. Runtime sidecar ghi thống kê, số timeout và tình trạng thiếu đo; đo mỗi sample không bao gồm khởi tạo model, hash đầu vào toàn cục và ghi journal/output.
- Nối thí nghiệm inference và SFT độc lập vào pipeline tuần tự; chỉ ghi delta inference khi có bằng chứng thực thi feature. Guardrail/constrained có thể `NOT_OBSERVED` nếu evaluator không cung cấp bằng chứng. Các job ghi state atomic, lệnh, log, kết quả và lý do skip; lỗi không được báo thành công. Bật log JSONL local chỉ trong worker, không đổi mặc định trainer.

## Dataset, thứ tự và tài nguyên

Baseline dùng `outputs/datasets/minicpm_baseline/train`, `models/MiniCPM5-2B`, `configs/training_lora.yaml`: BF16, LoRA r16/alpha32, LR 2e-4, 8192 token, microbatch 1, accumulation 8, 3 epoch, horizon 1614. Preflight đọc manifest và đối chiếu held-out; 20 ID chọn có SHA-256 `494932f4343fa89ceaca509ab880bd637423746b73d0116acd3aa3bea52da5c2`. Không dùng training group làm validation. Đĩa trống quan sát 131 GiB, RTX 5070 Ti trống 14621 MiB tại thời điểm kiểm tra; runner kiểm tra lại GPU, RAM, disk trước job. Các giá trị này thay đổi theo thời gian.

Thứ tự: baseline → bảo vệ adapter → chọn trên held-out → submission candidate nếu còn đủ thời gian theo cận dưới đo được → audio baseline và ITN/guardrail/constrained/VAD/RMS → SFT độc lập khi thời gian và tài nguyên đủ. Mỗi variant có dataset, checkpoint, log và horizon riêng và bắt đầu từ base. Chuẩn bị thử trên dữ liệu train thật, không train: slot masking tạo 119 biến thể, 119 qua lọc tokenizer, horizon 1659; tool dropout tạo 4368 biến thể, 4308 qua lọc, horizon 3228; cùng held-out. Runner tự tạo lại vào thư mục experiment, kiểm hash và không ghi đè. Noisy-ASR hiện **SKIPPED_MISSING_ASSET**: chưa cấu hình transcript train và provenance có hash. Không giả lập transcript.

Kết quả, log và state dự kiến nằm trong `outputs/overnight/`: `state.json`, `worker.log`, `logs/`, `baseline/checkpoints/lora/`, `candidates/`, `evaluation/`, `variants/`, `submissions/`, `FINAL_RESULTS.vi.md`. Theo dõi tốc độ/ETA chỉ khi có dữ liệu đo; không tự bịa thời lượng. Hết thời gian hoặc `--stop` gửi SIGINT hợp tác và đợi tiến trình; nếu không có checkpoint hợp lệ để resume thì dừng và báo lỗi. Lỗi bất ngờ không được retry vô hạn; một số công việc không phải training bị gián đoạn yêu cầu kiểm tra thủ công thay vì ghi đè.

## Xác minh

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --dry-run`: PASS; xác nhận lệnh baseline và held-out, không train.
- `/tmp/cudaenv/bin/python -m pytest -q tests/test_overnight_runner_integration.py tests/test_overnight_safety.py tests/test_submission_readiness_repairs.py`: **31 passed**. Các job tổng hợp sử dụng tiến trình tách rời thật, checkpoint giả và evaluator giả; kiểm tra start/duplicate/status, chuỗi tuần tự, episode model thất bại, candidate adapter-only, stop/resume, exit 7, khóa GPU, SFT riêng horizon khác và báo cáo. Đây không phải bằng chứng huấn luyện MiniCPM thật.
- Kiểm thử journal/inference/evaluation liên quan: **61 passed**. Không chạy lại smoke MiniCPM 8192 hoặc full training theo yêu cầu; kết quả ở báo cáo trước không được tính là kết quả mới.
- Full regression tại worktree sạch từ commit mã `faa7fed`, với `WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q`: **983 passed, 1 skipped, 1 warning**, 79,72 giây. Lượt đầu có 3 test thất bại vì manifest ghi đường dẫn tuyệt đối của checkout; test nay xác nhận đường dẫn đúng rồi chuẩn hóa trường đó khi so digest, và 12 test dataset đã qua. Commit báo cáo sau đó chỉ thay tài liệu; code được test không đổi. Không commit model, dataset, checkpoint, log hoặc credentials.
- Docker, full submission và điểm chính thức **NOT TESTED** trong phiên; không tuyên bố submission-ready. Candidate thật và chất lượng task chỉ xác định sau khi chạy thực tế.

## Gate G01–G30

| Gate | Status | Bằng chứng / giới hạn |
|---|---|---|
| G01 | NOT TESTED | Không lặp lại audit toàn repository. |
| G02 | PASS | Config baseline được preflight, regression chạy với code thay đổi. |
| G03 | PASS | Manifest và held-out được kiểm khi preflight. |
| G04 | NOT TESTED | Không chạy lại xác minh action format trên toàn dataset trong phiên. |
| G05 | NOT TESTED | Lệnh chọn LoRA được kiểm; chưa nạp adapter mới thật trong phiên. |
| G06 | PASS | `select_heldout` kiểm ID gốc; test và dry-run. |
| G07 | PASS | Guard group và held-out hoạt động trong preflight. |
| G08 | PASS | Lệnh mỗi arm truyền model mode và checkpoint; test lệnh. |
| G09 | PASS | SFT variant dùng config baseline cùng feature override; test horizon riêng. |
| G10 | PASS | Hai dataset train thử thật tạo biến thể và kiểm retained. |
| G11 | PASS | Không quan sát được thì `NOT_OBSERVED`, không tính delta. |
| G12 | PASS | Test candidate adapter-only, SHA, retention và best. |
| G13 | NOT TESTED | Test stop/resume tổng hợp; không chạy lại real exact resume. |
| G14 | PASS | Test journal bị ngắt và resume; không đại diện full inference thật. |
| G15 | NOT TESTED | Đường lệnh dùng SDK chính thức; không chạy full pipeline thật. |
| G16 | PASS | Guard đường dẫn và test truy cập cấm; không chủ ý đọc hidden state. |
| G17 | PASS | Thư mục variant/candidate riêng được test tổng hợp. |
| G18 | PASS | Test khóa GPU toàn cục và từ chối worker thứ hai. |
| G19 | PASS | Test exit code 7 được ghi FAILED. |
| G20 | PASS | Test stop/resume worker với checkpoint tổng hợp; real training chưa chạy. |
| G21 | NOT TESTED | Nhánh xử lý thiếu noisy-ASR asset đã nối, chưa chạy integration thực cho nhánh này. |
| G22 | PASS | So sánh cùng held-out, episode model thất bại tính 0; không phải official score. |
| G23 | PASS | Tên submission theo SHA adapter; test luồng tổng hợp. |
| G24 | NOT TESTED | Kiểm validator được nối; chưa tạo submission full thật. |
| G25 | NOT TESTED | Không chạy Docker. |
| G26 | PASS | Dry-run thật xuất lệnh baseline và evaluate. |
| G27 | PASS | 31 test tập trung và 61 test liên quan như trên. |
| G28 | PASS | 983 passed, 1 skipped, 1 warning ở worktree sạch commit mã `faa7fed`. |
| G29 | PASS | Test start/status/stop/resume và báo cáo với tiến trình tổng hợp. |
| G30 | PASS | Không khởi chạy full training trước GO. |

## Lệnh sau khi bạn phát GO

Chọn thời điểm dừng tương lai, có timezone và còn tối thiểu 120 phút. Chỉ chạy `--start` sau GO; runner từ chối nếu `outputs/overnight/state.json` đã tồn tại. Không xóa state/checkpoint để chạy lại.

```bash
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --dry-run
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --start --stop-at 2026-11-16T07:00:00+07:00
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --status
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --stop
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --resume --stop-at 2026-11-16T07:00:00+07:00
```

`--resume` chỉ dùng cho run đã ngắt và checkpoint hợp lệ; phải chọn deadline mới nếu deadline cũ đã hết. Kiểm tra `state.json` và log trước khi xử lý FAILED, không xóa kết quả để tự động retry.

SHA commit chứa báo cáo: `git log -1 --format=%H -- docs/FINAL_OVERNIGHT_READINESS.vi.md`; SHA hiện hành và remote: `git rev-parse HEAD` và `git ls-remote origin refs/heads/main`.
