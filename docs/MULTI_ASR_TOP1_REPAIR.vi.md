# Rà soát Multi-ASR Top-1

Đã sửa metadata `model_id` theo model thực sự sinh transcript được chọn (gồm trường hợp primary lỗi); giữ thông tin từng nguồn. Đo wall time cho toàn bộ bước ASR/tiền xử lý và latency từng backend, kèm cache-hit. Đường bốn model giữ toàn bộ candidate thô, áp dụng ITN trước khi gộp text cuối, nối provenance, rồi chọn tối đa ba; nếu hai candidate trùng sau ITN thì candidate thứ tư có thể bù vào. Aggregate cache kiểu cũ của chế độ này được vô hiệu hóa theo version chọn candidate. Legacy primary/secondary tiếp tục hoạt động.

Cấu hình và constructor từ chối Top-K ngoài 1..3; `build_top1_multi` từ chối trước khi tải model. Kiểm thử xác nhận prompt cuối, fallback primary lỗi, cache-hit, đổi artifact/decoder/audio, thất bại từng model và budget. Budget vẫn không ngắt cứng một decode đang thực thi.

## Chính sách chọn

Mặc định chạy đủ bốn Top-1 theo thứ tự cấu hình, rồi lấy ba text khác nhau đầu tiên, không so sánh acoustic score. Khi cả bốn khác nhau, bản thứ tư không vào prompt: một trong bốn decode, tức 25% **số lượt decode**, không được dùng trực tiếp trong prompt. Không thể suy ra 25% latency vì tốc độ và cache của từng model khác nhau. Để giảm chi phí, cấu hình chỉ ba model ưu tiên sẽ tránh decode model thứ tư; đổi lại không có candidate thứ tư để bù nếu ITN/khử trùng lặp làm giảm số text. Không có đo task-level chứng minh chính sách nào tốt hơn; cần đánh giá có nhãn trước khi chọn.

## ChunkFormer

Metadata [PyPI chunkformer 1.2.2](https://pypi.org/project/chunkformer/) yêu cầu Python >=3.11, torch/torchaudio >=2.5.1 và nhiều dependency gồm deepspeed, pydub. `/tmp/cudaenv/bin/python` là 3.12.11, nhưng chưa cài chunkformer, torchaudio, deepspeed, pydub, sentencepiece hoặc librosa. [Mã upstream](https://github.com/khanld/chunkformer/blob/main/chunkformer/chunkformer_model.py) dùng `from_pretrained` và `endless_decode`, xử lý âm thanh về mono, 16 kHz theo config. Mã local có adapter tương ứng, song chưa load weights thật; tính tương thích package/torch/transformers và output thực tế vẫn BLOCKED.

## Giới hạn xác minh

Đã dùng test CPU synthetic/mocks; không download weights hoặc chạy model GPU/CPU thật, không benchmark. Smoke model thật và so sánh task-level với baseline đều BLOCKED vì public-test inference đang chiếm tài nguyên; ngoài ra thiếu weights/runtime của ba model optional. Test mock và schema không chứng minh chất lượng hành động hay submission readiness.
