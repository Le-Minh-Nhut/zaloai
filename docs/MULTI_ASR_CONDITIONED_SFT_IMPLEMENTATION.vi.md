# Multi-ASR conditioned SFT: triển khai và giới hạn

Trạng thái: **đã tích hợp và kiểm thử tổng hợp trên CPU; chưa xác minh mô hình thật, huấn luyện thật hoặc kết quả tác vụ**. Không suy ra khả năng nộp bài từ các kiểm thử này. Nhánh xuất phát `codex/multi-asr-top1`, commit `47ae258c5934ff850aef1c2ed9d0f29a55f55bac`; công việc thực hiện trong worktree riêng.

## Luồng dữ liệu và thay đổi

`transcribe_split.py --multi-top1-records` tạo JSONL có phiên bản và manifest, chứa kết quả riêng cho từng recognizer và từng lượt audio, bao gồm audio lịch sử. Các trường nhận diện model/artifact, cấu hình, hash audio, raw/normalized, lỗi, provenance, thời gian và cache cho phép kiểm tra và tái sử dụng. Đầu ra tạm được ghi bền vững và hỗ trợ tiếp tục; không ghi đè thành phẩm. `conditioned_records.py` và `top1_cache.py` thực hiện lưu trữ và kiểm tra cache; cache chặn đường dẫn trỏ vào thư mục environment bị cấm.

`prepare_dataset.py` chia nhóm trước khi bổ sung biến thể train, giữ nguyên mọi gold action và clean record. `conditioned_sft.py` kiểm tra hash artifact/audio, yêu cầu bốn backend thành công, thay transcript lịch sử user bằng kết quả ASR của chính lượt đó và loại biến thể nếu thiếu dữ liệu. Ba chế độ `single`, `multi`, `mixed` và tỷ lệ được chọn xác định bằng seed. Không chỉnh validation hay nhãn vàng. Manifest ghi hash dataset, đặc tính ASR, cấu hình và quyết định audit. Dùng tokenizer MiniCPM thực khi chuẩn bị dữ liệu thật để thống kê độ dài và truncation.

Prompt train và inference dùng chung renderer trong `prompt_builder.py`: phân biệt các cách nhận dạng của cùng một phát ngôn, không xem đồng thuận là sự thật; ITN và khử trùng lặp xảy ra trước chọn K, provenance được gộp. K=1–4 chỉ dành cho chế độ conditioned; các chế độ cũ giữ giới hạn cũ. Khi tính năng tắt, hợp đồng prompt baseline được giữ. Inference từ chối `--asr none` kết hợp multi-ASR; số lượng ứng viên không nguyên, boolean và ngoài phạm vi bị từ chối. Direct API, latency thực tế, latency lịch sử của cache, thứ hạng, nguồn và các lỗi từng backend được giữ riêng. Lỗi một backend được ghi degraded, không được coi là thí nghiệm bốn-model hoàn chỉnh.

Audit coverage là heuristic thận trọng, **không phải chứng minh ngữ nghĩa**. Mâu thuẫn thời gian rõ ràng được phân loại `UNSUPPORTED`; phần lớn trường hợp khó là `UNKNOWN`; `SUPPORTED` chỉ là kiểm tra khớp chữ hạn chế. Mặc định loại `UNKNOWN` và `UNSUPPORTED` khỏi biến thể, vẫn giữ dữ liệu sạch. Do đó tập thật có thể không tạo được biến thể nào; không xem tập rỗng là thành công. Chỉ cân nhắc `--conditioned-unknown include` sau khi kiểm tra độc lập và chấp nhận rủi ro nhãn không đủ chứng cứ; tùy chọn này không biến heuristic thành bộ kiểm chứng semantic. Không tạo nhãn hỏi lại giả.

Các phần chính: `scripts/{infer,evaluate,prepare_dataset,transcribe_split,report_multi_asr,compare_conditioned_experiments}.py`; `src/lifemate/asr/{conditioned_records,top1_cache,top1_models,multi_backend,interface}.py`; `src/lifemate/data/{conditioned_sft,history}.py`; prompt/agent, config/registry và evaluator. Cấu hình nghiên cứu nằm trong `configs/training_multi_asr_conditioned.yaml`, `configs/research/multi_asr_experiments.json`; kiểm thử trong `tests/test_conditioned_sft.py` và `tests/test_four_model_top1.py`.

## Thiết kế thí nghiệm

- E0: adapter baseline hiện có, single ASR; không huấn luyện lại.
- E1: cùng adapter, prompt multi-ASR.
- E2: LoRA độc lập từ base, clean + single-ASR; E3: clean + multi-ASR; E4: clean + single + multi. Giữ nguyên hyperparameter baseline, tách output và theo dõi group/seed/manifest. Đây là cấu hình chuẩn bị, chưa phải kết quả.
- K=1,2,3,4 được hỗ trợ tại renderer trong chế độ mới. Producer lưu cả bốn kết quả; thay K không yêu cầu nhận dạng lại khi dựng dataset. Đường đánh giá end-to-end hiện vẫn có thể phải decode lại: chưa có lệnh ablation K tái sử dụng trực tiếp artifact bốn-ASR để chạy agent và ToolEnv. Đây là phần còn thiếu, không tuyên bố đã hoàn thành.
- `report_multi_asr.py` chỉ báo cáo WER/CER khi có reference train, thất bại và cache; `compare_conditioned_experiments.py` so sánh chẩn đoán local trên cùng ID held-out và protocol audio/history, từ chối mock/degraded. Điểm chính thức và correctness final-state chỉ có khi nguồn quan sát tương ứng thực sự có sẵn. Các phép đo intent/slot, VRAM và phân tích task-level đầy đủ chưa được thực chứng.
- `evaluate.py --force-asr-history` loại transcript gold của user trong lịch sử trước ASR. Lượt không có audio hợp lệ phải được ghi nhận lỗi; không thay bằng transcript gold hoặc bịa nội dung.

## Kết quả xác minh

Một lượt regression CPU mới: **435 PASS, 0 FAIL, 1 SKIPPED, 3 DESELECTED**; hai cảnh báo PyTorch. Lệnh:

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q \
  tests/test_conditioned_sft.py tests/test_four_model_top1.py tests/test_multi_asr.py \
  tests/test_asr.py tests/test_asr_variants.py tests/test_asr_model_specs.py tests/test_gipformer.py \
  tests/test_agent.py tests/test_features.py tests/test_feature_integration.py \
  tests/test_prepare_dataset_features.py tests/test_evaluation_modes.py tests/test_training.py tests/test_train_target_logits.py \
  --deselect=tests/test_prepare_dataset_features.py::test_real_split_is_not_required_but_available \
  --deselect=tests/test_asr_model_specs.py::test_backend_reports_missing_fp32_files_before_importing_sherpa \
  --deselect=tests/test_gipformer.py::test_factory_default_decoding_matches_backend_and_config
```

Ba bài bị loại phụ thuộc dữ liệu/weights không có trong worktree kiểm thử; không tính là PASS. Đây không phải toàn bộ test suite. Các bài test mô phỏng kiểm tra prompt, cache, timeout mềm, lỗi backend, provenance, tách nhóm, target, lịch sử và mặt nạ loss; chúng không chứng minh tương thích weights thật. Chưa chạy inference ASR, benchmark, SFT hay đánh giá task-level. **Real ASR: BLOCKED/NOT TESTED; real training: NOT TESTED; task-level: NOT TESTED; submission: NOT VERIFIED.** Chưa tải model hoặc khởi chạy GPU. ChunkFormer cần đúng package/runtime và checkpoint; Gipformer 1.5 cần artifact xác minh; không giả định các dependency hoặc checkpoint đã tồn tại.

## Lệnh thực thi sau khi tự kiểm tra tài nguyên

Chạy từ worktree có `data`, `models` và Python/dependency phù hợp. Các lệnh dưới đây tạo artifact mới và có thể tốn nhiều tài nguyên; **chưa chạy trong lượt triển khai này**. Trước tiên kiểm tra đủ bốn model thật và môi trường offline, tránh chạy đồng thời job GPU khác.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/transcribe_split.py --data-root data --split train \
  --config configs/training_multi_asr_conditioned.yaml \
  --multi-top1-records outputs/research/multi_asr/train_top1.jsonl
# Để tiếp tục artifact dở dang, chạy lại lệnh trên kèm --resume-multi-top1.
python scripts/report_multi_asr.py --data-root data \
  --records outputs/research/multi_asr/train_top1.jsonl \
  --output outputs/research/multi_asr/asr_metrics.json
python -c "from lifemate.data.conditioned_sft import load_records; i,m=load_records('outputs/research/multi_asr/train_top1.jsonl'); print(len(i),m['sha256'])"
```

Ví dụ E3 (E2 dùng `--conditioned-mode single`; E4 dùng `mixed`, với output riêng). Lệnh audit coverage nằm trong quá trình chuẩn bị và ghi vào manifest/stats; đọc phân bố trước khi quyết định huấn luyện:

```bash
python scripts/prepare_dataset.py --data-root data --split train \
  --config configs/training_multi_asr_conditioned.yaml \
  --conditioned-multi-records outputs/research/multi_asr/train_top1.jsonl \
  --conditioned-mode multi --output outputs/research/e3_dataset \
  --tokenizer models/MiniCPM5-2B --max-seq-length 8192
python -c "import json; print(json.dumps(json.load(open('outputs/research/e3_dataset/train/manifest.json'))['conditioned_multi_asr'],ensure_ascii=False,indent=2))"
[ ! -e outputs/checkpoints/lifemate_e3 ] && python scripts/train.py \
  --config configs/training_multi_asr_conditioned.yaml \
  --dataset-dir outputs/research/e3_dataset/train --model models/MiniCPM5-2B \
  --output-dir outputs/checkpoints/lifemate_e3 --save-steps 100
```

Chỉ chạy lệnh train nếu manifest, kiểm tra prompt/target, tokenizer thực, tài nguyên và checkpoint output đã được xác nhận. Prompt parity tổng hợp được kiểm tra trong `tests/test_conditioned_sft.py`; chưa có kiểm tra parity bằng dữ liệu thật. Sau khi có adapter, dùng `scripts/evaluate.py --model-mode lora --checkpoint <adapter> --heldout-dataset <dataset> --evaluation-mode audio_to_action --force-asr-history --closed-loop --config <config> --data-root data --split train --output <output>`; dùng cùng tập ID và protocol cho mọi arm. Với artifact metrics đã có, `scripts/compare_conditioned_experiments.py --experiment E0=<metrics.json> --experiment E1=<metrics.json> --data-root data --output <new-report.json>`. Các placeholder này yêu cầu chọn đường dẫn artifact thật; chúng không khẳng định checkpoint hay kết quả đã tồn tại.

## Hạn chế cần xử lý trước khi tuyên bố nghiên cứu hoàn tất

Không có xác minh runtime bốn ASR, transcript thật, khả năng fit GPU, real-LoRA resume, ToolEnv task-level hoặc submission. Đã bổ sung replay artifact cho ablation K; chưa đo tất cả chỉ số intent/slot/final-state yêu cầu. Policy coverage hiện có thể quá bảo thủ; không tự động xóa rào chắn bằng cách đưa `UNKNOWN` vào train. Cần kiểm tra dữ liệu và review semantic trước khi quyết định cách xử lý. Không có bằng chứng cải thiện chất lượng hay điểm leaderboard.


## Bổ sung sau commit f52f24a — sửa E2, provenance và replay

- E2 chọn model bằng `--conditioned-single-model` (mặc định Gipformer 68M). Chỉ model được chọn phải thành công cho từng lượt hiện tại và lịch sử. E3 yêu cầu đủ bốn model; E4 đánh giá độc lập nhánh single và multi, không đánh đồng kết quả hai nhánh.
- Audit coverage nhận biết một tập từ khóa ý định báo thức tiếng Việt, dùng ITN hiện có để nhận giờ nói rõ và giữ `UNKNOWN` khi mơ hồ hoặc bằng chứng ngữ cảnh xung đột. Đây là heuristic hẹp, không chứng minh ý định, ngày, số lượng hay liên hệ slot đầy đủ; không đổi chính sách loại `UNKNOWN` mặc định. Không suy ra độ bao phủ trên dữ liệu thật từ fixture. Phân bố thực: **NOT TESTED**, chưa có artifact bốn ASR thật để audit.
- Mỗi record tăng cường ghi hash artifact, run identity và phiên bản prompt. Manifest conditioned ghi hash train/val; train từ chối file bị đổi và gắn hash toàn bộ manifest vào checkpoint. Hash thứ tự dataset và kiểm tra exact-resume sẵn có được giữ. Không thay identity của baseline cũ.
- `--replay-multi-asr` của `scripts/evaluate.py` đọc artifact train đã xác minh checksum, model, revision, audio và toàn bộ lượt lịch sử; không khởi tạo ASR. `--replay-k 1..4` chọn lại candidate từ cùng artifact. Kết quả ghi `replay_identity`; so sánh từ chối artifact hoặc tập ID/protocol khác. Replay là quality evaluation: thời gian decode ASR trong quá khứ không được tính thành latency hiện tại và không chứng minh giới hạn thời gian thi. Không sử dụng gold history.
- Replay cố ý chặn output thiếu model, model identity/revision không khớp, audio hoặc history thiếu/sai hash. Quy trình hiện chỉ nhận artifact split train và giả định đầy đủ bốn model; chưa chạy SDK thật. Các lựa chọn số candidate, model và lịch sử phải được giữ nhất quán khi so sánh.

Ví dụ sau khi có artifact thật và adapter đã chọn (chưa chạy; không dùng khi GPU đang bận):

```bash
PYTHONPATH=src python scripts/evaluate.py --config configs/training_multi_asr_conditioned.yaml \
  --data-root data --split train --heldout-dataset outputs/research/e3_dataset/train \
  --model-mode lora --checkpoint outputs/checkpoints/lifemate_e3 \
  --evaluation-mode audio_to_action --closed-loop \
  --replay-multi-asr outputs/research/multi_asr/train_top1.jsonl --replay-k 3 \
  --output outputs/research/e3_replay_k3
```

### Kết quả kiểm thử lượt này

- Nhóm CPU độc lập: **330 PASS, 0 FAIL, 1 SKIPPED, 2 DESELECTED**, lệnh `WANDB_MODE=disabled PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q tests/test_conditioned_sft.py tests/test_asr.py tests/test_multi_asr.py tests/test_asr_model_specs.py tests/test_features.py tests/test_feature_integration.py tests/test_asr_variants.py tests/test_prepare_dataset_features.py tests/test_training.py tests/test_training_resume.py -k 'not test_backend_reports_missing_fp32_files_before_importing_sherpa and not test_real_split_is_not_required_but_available'`. Kết quả trước hai kiểm thử bổ sung cuối; các kiểm thử bổ sung nằm trong lượt tập trung bên dưới.
- Sau thay đổi cuối, `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q tests/test_conditioned_sft.py tests/test_evaluation_modes.py tests/test_train_target_logits.py`: **36 PASS, 0 FAIL, 0 SKIPPED**. Fixture/mock, không phải ASR hoặc ToolEnv thật.
- Lượt đầu bao gồm cả hai bài phụ thuộc môi trường: **329 PASS, 2 FAIL, 1 SKIPPED**. Hai FAIL: worktree không có `models/gipformer-68M-rnnt` và `data/train/train.jsonl`; không tải weights hoặc tạo dữ liệu giả để che lỗi. Hai bài này **BLOCKED: 2** trong worktree hiện tại.
- Các mục **NOT TESTED: 4**: phân bố coverage trên dữ liệu thật, ASR bốn model thật, huấn luyện/checkpoint thật, đánh giá task-level với ToolEnv thật. Không có kết quả hay điểm thi chính thức. Không đủ bằng chứng tuyên bố nghiên cứu hoàn tất hoặc sẵn sàng nộp bài.
