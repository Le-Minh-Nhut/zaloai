# Báo cáo chuẩn bị thí nghiệm qua đêm

**TRẠNG THÁI: BLOCKED — CHƯA RELEASE RUNNER.** Không có full training nào được khởi chạy trong phiên. Đây là các sửa và bằng chứng đã hoàn tất, không phải xác nhận rằng toàn bộ yêu cầu đã xong. `--start`, `--resume` và worker bị khóa có chủ ý vì các giai đoạn bắt buộc còn thiếu. Không gửi GO để chạy runner này cho đến khi các blocker được xử lý.

## Phạm vi và nguồn chính thức

Bắt đầu tại `57cb1bec250f968ffa3297fb3542e57a96a68531`, nhánh `main`; so sánh mã hiện hành ở các điểm checkpoint, held-out, inference, ablation, trainer và ToolEnv. **Chưa hoàn thành audit toàn diện mọi dependency** như yêu cầu G01. [Mô tả LifeMate chính thức](https://challenge.zalo.ai/portal/lifemate) xác nhận audio, context/history/apps, tool calls, `sys.ask_missing_info`, no-action và state hash từ environment. [Trang timeline chính thức](https://challenge.zalo.ai/) hiện hiển thị hạn nộp solution 16/11/2026 23:59 GMT+7; cần theo dõi cập nhật chính thức. Các chi tiết giới hạn phần cứng/thời gian/parameter, Docker interface và công thức chấm không trích xuất được đầy đủ: UNVERIFIED. LCS-F1 và exact match ở đây là **chẩn đoán cục bộ**, không phải điểm chính thức.

## Phát hiện và sửa thực tế

| Phân loại | Vấn đề và thay đổi | Bằng chứng |
|---|---|---|
| CONFIRMED BUG | `feature_ablation.py` không truyền checkpoint/model mode; nay nhận `--checkpoint`, `--model`, `--heldout-dataset`, truyền LoRA adapter cho từng arm, ghi SHA-256 và từ chối real ablation thiếu adapter/held-out. | Test argv và hai arm thật trên một mẫu held-out dùng cùng adapter; cả hai mẫu lỗi lặp, không chứng minh chất lượng. |
| CONFIRMED BUG | Closed-loop train lấy các mẫu đầu thay vì held-out; nay `--heldout-dataset` đọc ID từ JSONL chuẩn bị, đối chiếu hash manifest, tập ID, group và split trước khi dùng sample gốc/SDK. Real train evaluation mặc định yêu cầu held-out; có opt-in rõ ràng cho train diagnostics. | 156 source ID held-out, không giao nhóm; một mẫu dùng adapter thật và SDK thật. |
| CONFIRMED BUG | Inference trước đây chỉ ghi khi kết thúc. Nay journal riêng `.progress.json` được ghi atomic, checksum, gắn config, dataset/audio/history/model/adapter identity; `--resume` kiểm tra thứ tự, hash, status lỗi và không chạy lại các hàng đã hoàn tất. File submission chính chỉ được finalize sau vòng lặp. | Test ngắt sau 2/3 mẫu, resume chỉ chạy mẫu còn lại; test journal hỏng và sai identity. Hash model cục bộ dùng metadata cho weights lớn, không phải xác thực mật mã toàn bộ base. |
| CONFIRMED BUG | Các arm có lỗi sample vẫn có thể nhận delta; nay đánh dấu INCOMPLETE và từ chối delta khi có lỗi, feature degraded hoặc không được quan sát. Ghi nhận ITN thực thi; các feature khác thiếu bằng chứng không được tính effective. | Test episode lỗi và feature status. Hai arm thật trước sửa đều `repeated_failure`; không dùng số đo đó làm kết luận cải thiện. |
| CONFIRMED DESIGN GAP | Candidate tốt có thể bị pruning trước khi được đánh giá. Prototype nay theo dõi và copy checkpoint hoàn chỉnh sang thư mục độc lập, không ghi đè, kiểm tra disk và SHA-256; chưa kiểm thử vòng theo dõi trên job training unattended thật. | Test copy không ghi đè và sao chép một checkpoint thật; BLOCKED cho bảo vệ trong suốt training. |
| CONFIRMED DESIGN GAP | Chuỗi inference ablation, SFT độc lập và submission candidate chưa được tích hợp/kiểm thử tự động. | BLOCKED; hard gate ngăn chạy runner. |
| ALREADY FIXED | Trainer MiniCPM LoRA, exact resume và tải adapter đã có trước; không rewrite trainer. | Tiny resume EQUIVALENT; real inference 1 sample với adapter smoke. |
| ENVIRONMENT BLOCKER | Docker CLI không có; không có container run, official private scorer hoặc A30. | `bash -n docker/entrypoint.sh` chỉ kiểm tra cú pháp. |
| UNVERIFIED | Truy cập đầy đủ quy định/điểm chính thức, toàn split, chất lượng final state và recovery unattended thật. | Không có điểm leaderboard hoặc tuyên bố submission ready. |

## Dataset, phân loại và cấu hình dự kiến

Baseline dataset `outputs/datasets/minicpm_baseline/train` đã có; manifest khớp held-out, gồm 4.298 train/477 val record sau fit. Prototype baseline dự kiến dùng `configs/training_lora.yaml`, model `models/MiniCPM5-2B`, `--save-steps 100`, output `outputs/overnight/baseline/checkpoints/lora/`, giữ BF16, r16/alpha32, LR 2e-4, 8192 token, microbatch 1, accumulation 8, 3 epoch. Không dùng overlay như config training hoàn chỉnh.

| Loại | Thí nghiệm | Dataset / trạng thái |
|---|---|---|
| Training | baseline | dataset có sẵn; chưa train trong phiên |
| Training | slot masking, tool dropout | dataset độc lập chưa chuẩn bị/kiểm định; BLOCKED |
| Training | noisy-ASR | transcript thật và hiệu quả augmentation chưa được kiểm tra trong runner; BLOCKED / thiếu asset phải SKIPPED_MISSING_ASSET |
| Inference | ITN, guardrail, constrained decoding, VAD, RMS | dùng adapter đã train; khung ablation có checkpoint/held-out, nhưng orchestration và quan sát thực thi toàn bộ feature chưa hoàn tất |
| Tùy chọn | Gipformer 1.5, genuine N-best, Multi-ASR | chỉ chạy khi artifact/runtime thật được xác minh; không giả lập thành công |

## Prototype runner và các hạn chế

`scripts/run_overnight.py` cùng `configs/overnight_experiments.yaml` có dry-run, status, prototype state JSON atomic, log tách biệt, kế hoạch baseline/đánh giá held-out, signal dừng hợp tác, theo dõi checkpoint và copy candidate. Kế hoạch chọn trong prototype là local tool-call exact match trên cùng 20 ID held-out, chỉ so sánh các checkpoint còn lại; đó không phải official score hay bằng chứng chọn được checkpoint tốt nhất toàn bộ run. Bản copy một checkpoint MiniCPM thật có SHA-256 `c63789a4f4556b1f6ea7d357dd76c4e5af84221783b852be26f5da6349ac8334`; vòng theo dõi trong khi train chưa chạy. Prototype có ngưỡng disk 15 GiB, GPU free 12.000 MiB và thời gian khởi động tối thiểu 120 phút; chưa xác nhận đầy đủ RAM, race process, crash recovery, retention trong training hay thời lượng. **Các phần này chưa đủ điều kiện release.**

Có thay đổi tracked tồn tại trước phiên ở `src/lifemate/evaluation/metrics.py`, cùng các file/thư mục untracked của người dùng. Chúng không bị sửa hoặc commit; prototype kiểm tra tracked tree sạch trước khi chạy để không gắn sai Git SHA. Do đó, ngoài các blocker kỹ thuật, cần xử lý trạng thái làm việc này bằng một checkout sạch hoặc theo quyết định của chủ sở hữu trước khi chạy unattended.

Dự kiến thứ tự khi hệ thống hoàn chỉnh: baseline → bảo vệ checkpoint hợp lệ → held-out validation → chọn candidate bằng bằng chứng task-level → inference-only ablation → SFT độc lập nếu đủ thời gian/tài nguyên → submission candidate riêng → báo cáo. Hiện runner **không thực hiện được đầy đủ thứ tự này**; không có best candidate hay file submission tự động được tạo.

## Xác minh và Docker

- Targeted `.venv/bin/python -m pytest -q tests/test_overnight_safety.py tests/test_inference.py tests/test_submission_readiness_repairs.py`: 55 passed (bao gồm kiểm tra đường dẫn cấm và symlink).
- Tiny exact resume `PYTHONPATH=src /tmp/cudaenv/bin/python scripts/verify_training_resume.py --tiny --steps 12 --interrupt-at 5 --accum 3`: EQUIVALENT.
- Real smoke: `scripts/infer.py` với audio public thật, Gipformer, MiniCPM, adapter LoRA **smoke** và official ToolEnv, một hàng; SDK `zaic_tools --validate` báo 1 dòng, 0 lỗi schema. Agent `repeated_failure`, 6 calls. Một held-out closed-loop thật cũng lỗi lặp. Đây không phải checkpoint SFT cạnh tranh, không chứng minh action/final-state đúng.
- Full regression sau sửa cuối: `WANDB_BASE_URL=http://127.0.0.1:9 PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q`: 983 passed, 1 skipped, 1 warning, 33,24 giây; bao gồm thay đổi người dùng tồn tại sẵn trong workspace. `git diff --check`, `bash -n docker/entrypoint.sh` chạy tại phiên này.
- Không có Docker: build BLOCKED, container/offline runtime BLOCKED, submission runtime NOT VERIFIED. Khi có Docker và adapter thật, lệnh tham khảo sau cần thay `ADAPTER_DIR`; chưa chạy và chưa xác minh interface private-test:

```bash
docker build -f docker/Dockerfile -t lifemate-infer .
ADAPTER_DIR=/duong/dan/adapter-thuc
docker run --rm --gpus all --network none \
  -v "$PWD/data":/data:ro -v "$PWD/models/MiniCPM5-2B":/models/base:ro \
  -v "$ADAPTER_DIR":/models/ckpt:ro -v "$PWD/models/gipformer-68M-rnnt":/models/asr:ro \
  -v "$PWD/outputs":/outputs -e SPLIT=public_test -e MODEL_MODE=lora \
  -e MODEL_NAME=/models/base -e CHECKPOINT=/models/ckpt -e ASR=gipformer -e DEVICE=cuda lifemate-infer
```

## Gate G01–G30

| Gate | Status | Bằng chứng hoặc giới hạn |
|---|---|---|
| G01 | NOT TESTED | Chưa audit toàn bộ dependency. |
| G02 | PASS | Regression/tiny resume, không thay trainer. |
| G03 | PASS | Held-out source/hash kiểm tra; dataset baseline đã tồn tại. |
| G04 | PASS | Regression parser/target hiện hữu. |
| G05 | PASS | Một lượt nạp adapter LoRA thật với SDK thật. |
| G06 | PASS | ID held-out và một mẫu closed-loop thật. |
| G07 | PASS | Guard ID/group overlap và kiểm tra 156 mẫu. |
| G08 | PASS | Hai arm thật truyền cùng adapter và cùng held-out ID; mẫu thất bại nên không có kết luận chất lượng. |
| G09 | BLOCKED | Chưa có pipeline config SFT độc lập hoàn chỉnh. |
| G10 | NOT TESTED | Các dataset augmentation chưa được tạo/xác minh. |
| G11 | BLOCKED | Một số feature thiếu quan sát thực thi; hiện fail closed. |
| G12 | BLOCKED | Copy bảo vệ đã có và test độc lập, vòng theo dõi khi training chưa xác minh. |
| G13 | PASS | Tiny exact resume EQUIVALENT. |
| G14 | PASS | Journal checksum/order và ngắt/resume 3 mẫu mock. |
| G15 | PASS | Một mẫu dùng SDK thật; không chứng minh task đúng. |
| G16 | PASS | Guard đường dẫn, source compliance regression; không có chủ ý truy cập hidden state. |
| G17 | BLOCKED | Chưa triển khai pipeline nhiều experiment. |
| G18 | NOT TESTED | Có lock/kiểm tra GPU trong prototype nhưng chưa kiểm thử unattended thực. |
| G19 | BLOCKED | Chưa kiểm thử toàn bộ chuỗi lỗi/khôi phục runner. |
| G20 | BLOCKED | Worker resume thực chưa kiểm thử; hiện khóa. |
| G21 | BLOCKED | Chưa triển khai skip asset cho pipeline SFT. |
| G22 | BLOCKED | Chưa chạy lựa chọn checkpoint huấn luyện cạnh tranh. |
| G23 | BLOCKED | Chưa triển khai tạo nhiều candidate submission. |
| G24 | BLOCKED | Mới kiểm schema một hàng, chưa toàn split. |
| G25 | PASS | Docker vắng được báo rõ, không tuyên bố đã chạy. |
| G26 | BLOCKED | Dry-run baseline đúng, các stage sau còn placeholder/khóa. |
| G27 | PASS | Các test tập trung đã chạy. |
| G28 | PASS | Full suite 983 passed, 1 skipped (bao gồm workspace người dùng). |
| G29 | BLOCKED | `--status`/dry-run hoạt động; start/resume/worker khóa có chủ ý, stop/recovery thật chưa kiểm thử. |
| G30 | PASS | Không full training; `--start` từ chối trước khi tạo state. |

## Lệnh trạng thái — không khởi chạy training

```bash
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --dry-run
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/run_overnight.py --status
```

CLI có `--stop`, `--start --stop-at <ISO8601 có timezone>` và `--resume`, nhưng **không có lệnh launch/recovery đã được release**: start/resume bị khóa cho đến khi các gate BLOCKED được xử lý. `--stop` chỉ có ý nghĩa nếu có state của một run. Không trình bày các lệnh này như workflow unattended sẵn sàng.

Commit chứa báo cáo xác định bằng `git log -1 --format=%H -- docs/FINAL_OVERNIGHT_READINESS.vi.md`; đối chiếu HEAD/remote bằng `git rev-parse HEAD` và `git ls-remote origin refs/heads/main`. Không commit dataset, weights, checkpoint, log hay các thay đổi người dùng có sẵn.
