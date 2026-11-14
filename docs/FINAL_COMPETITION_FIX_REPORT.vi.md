# Báo cáo sửa lỗi LifeMate

## A. Tổng quan

Đã sửa đường truyền hypotheses ASR sang evaluation và cache, kết hợp N-best/Multi-ASR, chọn Gipformer 1.5 theo cờ, ITN cho hypotheses, phiên âm audio lịch sử lúc chạy, đặt deadline inference trước history/ASR, sửa chọn chế độ evaluation và ngăn ablation ASR dùng transcript cố định. Công việc chưa đủ bằng chứng để tuyên bố sẵn sàng nộp bài: còn các giới hạn và kiểm thử thực tế bên dưới.

Workspace khi bắt đầu có các thay đổi chưa commit. `src/lifemate/evaluation/transcripts.py` đã được sửa sẵn và được mở rộng để tích hợp cache; các thay đổi có sẵn khác được giữ trong workspace, không đưa vào commit này. Không sửa trainer hoặc notebook.

## B. Yêu cầu chính thức

Tài liệu do ban tổ chức cấp `data/README.md` và `data/*/zaic_tools/README.md` xác nhận audio yêu cầu cuối và history, context, installed apps, ToolEnv theo mẫu, history đã áp dụng vào state, `sys.ask_missing_info`, trường submission `id`, `tool_calls`, `state_hash`, và đánh giá bằng replay/state/call F1. Trang [LifeMate](https://challenge.zalo.ai/portal/lifemate) xác nhận mục tiêu xử lý yêu cầu, hỏi thông tin thiếu và không hành động ngoài phạm vi. Chi tiết tab quy định không trích xuất được trong phiên này: chưa xác minh độc lập giới hạn 30 giây, A30, kích thước model, context, offline và mount Docker. Không coi nhận định cũ là quy tắc chính thức. [README upstream](https://github.com/ggroup-ai-lab/gipformer/blob/main/README.md) liệt kê model `g-group-ai-lab/gipformer1.5-68M-rnnt`.

## C. Lỗi và thay đổi

| Mã | Mức | Nguyên nhân và thay đổi | Bằng chứng |
|---|---|---|---|
| ASR-SECONDARY | P0 | Truyền `model_name` cho Gipformer; nay chọn `model_id` cho Gipformer. | `test_multi_asr.py`; backend giả. |
| ASR-FLOW | P0 | Metadata bị bỏ khi cache/evaluate; `transcription.py`, `evaluator.py`, `evaluate.py` nay giữ và chuyển hypotheses cùng transcript. | `test_competition_pipeline_fixes.py` kiểm tra cache và evaluator. Chưa có kiểm thử parity prompt đầu-cuối đầy đủ. |
| ASR-COMBINE | P0 | Nhánh `if/elif` loại trừ Multi-ASR; nay cho primary top-K và secondary, giới hạn và khử trùng lặp. Khi N-best tắt dùng primary top-1. | Kiểm thử backend giả; runtime Sherpa 2.0.1 không cung cấp decoder N-best thật. |
| ASR-V15 | P0 | Cờ không chọn model mới; nay chuyển model ID, không mượn thư mục/revision baseline. | Kiểm thử factory giả. Artifact spec chưa xác minh và backend từ chối; chạy model thật BLOCKED. |
| ASR-ITN | P1 | Hypotheses hiển thị không được ITN; nay xử lý từng hypothesis và lưu bản gốc trong metadata. | Kiểm thử `_transcribe` tới prompt xác nhận 07:30 và 20:00; chưa đo chất lượng model. |
| CACHE | P1 | Key cũ thiếu audio và tham số; tích hợp thay đổi có sẵn, hash SHA-256 audio và thông số ASR/audio, lưu hypotheses, bỏ cache lỗi hoặc fallback. | Kiểm thử ghi/đọc và thay đổi audio. Chưa đảm bảo nhận biết checkpoint bị thay nội dung tại cùng đường dẫn/model ID. |
| HISTORY | P0 | History không có text bị từ chối; thêm phiên âm audio được cung cấp theo thứ tự, giữ guard đường dẫn và từ chối lượt không xử lý được. | Kiểm thử giả và chạy Gipformer thật trên một mẫu public, 2 lượt được điền, 0 lỗi. Evaluation dùng cùng helper. |
| TIME | P0 | Deadline bắt đầu sau ASR; nay bắt đầu trước history/ASR. HF generator nhận `max_time`, không thực thi action sau deadline; profile `configs/competition_inference.yaml` đặt mục tiêu bảo thủ 30 giây. | Kiểm thử hiện có; `max_time` là cơ chế hợp tác, không bảo đảm ngắt cứng kernel/ASR. Evaluation tiền xử lý ASR trước vòng agent nên chưa đo toàn bộ latency theo mẫu. |
| EVAL | P1 | Audio-to-Action chạy thêm teacher forcing; nay chỉ chạy khi yêu cầu phù hợp. ASR/audio ablation tự chọn audio mode khi không nêu mode, từ chối text mode tường minh. | Bộ regression; không báo điểm chính thức từ mock. |
| STATUS | P0 | Fallback có thể bị tính hiệu quả; evaluation đếm yêu cầu/thực thi/suy giảm/bị chặn cho N-best và Multi-ASR; ablation vô hiệu hóa delta với trạng thái suy giảm hoặc ASR/audio chưa quan sát. | Kiểm thử hiện có; chưa có thống kê đầy đủ cho mọi feature. |

## D. Những phần giữ nguyên

Baseline tiếp tục tắt các cờ tùy chọn; MiniCPM5-2B và LoRA vẫn là mặc định. Không kết luận những hạng mục chưa kiểm tra là false positive. Metric LCS-F1 hiện là chỉ số chẩn đoán; công thức điểm chính thức đầy đủ chưa được công bố trong tài liệu có thể kiểm tra.

## E–G. Pipeline, lịch sử và thời gian

Inference: đọc audio chính thức → phiên âm các lượt user trước còn thiếu → phiên âm lượt hiện tại → chuẩn bị prompt → agent → SDK ToolEnv → ghi call/state hash. Không dùng nhãn test hoặc tự tạo transcript. Các lượt assistant và kết quả tool giữ nguyên. History không có audio hoặc ASR thất bại bị báo lỗi trừ opt-out chẩn đoán. Không có cache history runtime; nhiều lượt có thể làm quá ngân sách. Profile 30 giây là lựa chọn thận trọng, **không phải xác nhận quy định**. Không có phân phối mean/P50/P95/max toàn pipeline và không đo A30.

## H–I. ASR, evaluation và ablation

Gipformer baseline thật chạy offline thành công trên một file public (0,148 giây cho một lần đo riêng ASR); history thật hai lượt mất 0,182 giây trên cùng máy. Đây không phải benchmark. N-best thật và Gipformer 1.5 BLOCKED do runtime/spec; Multi-ASR và ITN kiểm tra chủ yếu bằng giả lập. VAD/RMS không bật mặc định; chưa xác minh feature execution đầy đủ. Audio-to-Action và Text-to-Action được phân biệt, nhưng đánh giá audio hiện tính ASR trước deadline agent. Điểm chính thức **UNAVAILABLE**; LCS-F1 không phải điểm leaderboard.

## J. Training và resume

Tiny resume: 12 bước, ngắt tại 5, accumulation 3: weights, optimizer, scheduler, thứ tự mẫu, loss và RNG tương đương. Smoke MiniCPM thật 2 bước trên môi trường CUDA kết thúc mã 137 sau khi tải trọng số: **BLOCKED**, chưa xác định chính xác nguyên nhân. Qwen LoRA/QLoRA thực tế **NOT TESTED**; checkpoint compatibility và noisy-ASR được bao phủ bởi regression hiện có, chưa đánh giá thực tế bổ sung.

## K. Docker và ToolEnv

`bash -n docker/entrypoint.sh`: PASS. Docker không có trong PATH: build và container **BLOCKED**, offline inference trong container và submission runtime **NOT TESTED**. SDK chính thức public_test được khởi tạo trên một mẫu, liệt kê 48 tool, từ chối tool không hợp lệ, trả ba trường submission và state hash; đây là smoke test, chưa chứng minh final state đúng.

## L. Kết quả kiểm thử

- `WANDB_BASE_URL=http://127.0.0.1:9 .venv/bin/python -m pytest -q`: **945 passed, 16 skipped**, 0 failed, 13,85 giây. Chạy trên workspace có các thay đổi sẵn có.
- `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q`: **957 passed, 1 skipped**, 1 cảnh báo, 52,30 giây; trước những sửa nhỏ cuối cùng, đã chạy lại targeted sau đó.
- `python3 tests/run_tests.py`: 926 passed, 0 failed, 26 skipped (cần fixture); một số module cần pytest.
- `scripts/verify_all.py --no-write --gates A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,R,P`: các gate của script PASS; phần lớn dùng fixture/mô phỏng, không phải xác minh cuộc thi.
- `scripts/verify_training_resume.py --tiny --steps 12 --interrupt-at 5 --accum 3`: EQUIVALENT.
- Kiểm tra cú pháp Python, `bash -n`, `git diff --check`: PASS tại thời điểm chạy.

## M–N. Blocker và quyết định

Docker/A30 không có; checkpoint 1.5 chưa được hỗ trợ bởi spec xác minh; chưa có scorer chính thức chi tiết; smoke MiniCPM thật bị dừng mã 137. Cần xác nhận quy định chính thức khi truy cập được cổng, nhất là thời gian/phần cứng/mount/điểm. Chưa xác nhận submission baseline có thể chạy hợp lệ trong container. Những việc đó không cần quyết định về thiết kế của người dùng nhưng phụ thuộc môi trường hoặc tài liệu. Không có quyết định người dùng nào đang yêu cầu để tiếp tục các phần đã thực hiện.

## O. Git

Commit khởi đầu `e07ccbae992413586faaadb541312fa44b49f0c4`, remote ban đầu `7037f1dfe6bdf4b3322bd4e58151caf759331203`. SHA commit cuối và kết quả push được nêu trong bàn giao hội thoại; một commit không thể tự ghi SHA của chính nó vào nội dung.

## Các cổng nghiệm thu G01–G29

| Cổng | Trạng thái | Giới hạn bằng chứng |
|---|---|---|
| G01 | BLOCKED | Tab quy định chính thức không đọc đủ. |
| G02 | PASS | Default flags qua regression. |
| G03 | PASS | Cache và evaluator chuyển hypotheses qua kiểm thử giả. |
| G04 | NOT TESTED | Ma trận thật với hai model/N-best chưa có runtime phù hợp. |
| G05 | PASS | Kiểm thử tham số factory bằng giả lập. |
| G06 | PASS | Kiểm thử chọn ID bằng giả lập. |
| G07 | PASS | Spec 1.5 chưa xác minh bị từ chối; chưa có weights để thử thực tế. |
| G08 | FAIL | Mới đếm N-best/Multi-ASR, chưa bao phủ mọi feature. |
| G09 | PASS | Kiểm thử đổi nội dung audio làm cache miss. |
| G10 | NOT TESTED | Chưa chạy đủ ma trận tham số cache end-to-end. |
| G11 | PASS | Kiểm thử `_transcribe` tới prompt với hai hypotheses và ITN. |
| G12 | PASS | Đã đổi nhánh và chạy regression. |
| G13 | PASS | Code yêu cầu audio mode khi thay ASR/audio; regression qua. |
| G14 | NOT TESTED | Dùng helper chung; chưa chứng minh prompt parity đầy đủ. |
| G15 | NOT TESTED | History audio thật chạy cho một mẫu public, private chưa có. |
| G16 | FAIL | Chưa có timeout cứng; evaluation chưa tính trọn ASR trong deadline. |
| G17 | BLOCKED | Thiếu công thức chính thức đầy đủ. |
| G18 | NOT TESTED | Chưa kiểm tra mọi output có nhãn diagnostic. |
| G19 | NOT TESTED | Smoke model thật bị dừng mã 137. |
| G20 | PASS | Tiny resume bit-identical; model thật chưa xác minh. |
| G21 | PASS | Regression checkpoint hiện có qua. |
| G22 | PASS | Regression noisy-ASR hiện có qua. |
| G23 | NOT TESTED | SDK thật smoke khởi tạo/validate; chưa kiểm tra chuỗi hành động end-to-end. |
| G24 | NOT TESTED | Chưa xác nhận submission thật end-to-end. |
| G25 | BLOCKED | Docker không có trong PATH. |
| G26 | PASS | Không thêm truy cập môi trường ẩn; guard và kiểm tra compliance qua. |
| G27 | PASS | Các kiểm thử mới qua. |
| G28 | PASS | Bộ pytest hiện có qua với các test skipped nêu trên. |
| G29 | PASS | Source thay đổi được xem diff và kiểm tra cú pháp. |
