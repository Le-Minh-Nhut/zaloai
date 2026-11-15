# Kết quả thực nghiệm E2/E3 — 15/11/2026

## E0 so với E2 step 62

Đã nạp adapter MiniCPM5-2B LoRA E2 tại `outputs/checkpoints/lifemate_e2/lora/interrupt-62` để inference; không resume training. Baseline dùng `outputs/overnight/candidates/step-1614`. Cả hai dùng cùng 12 ID held-out (SHA256 danh sách `68665740385d74c4ebb03db9cfa19c292883b9fc48ea45de65e2debcbd356a15`), cùng artifact Gipformer 68M, lịch sử ASR, prompt K=1 và ToolEnv thật. Đây là artifact replay; thời gian không bao gồm decode ASR lịch sử.

| Chỉ số cục bộ theo episode | E0 | E2 step 62 |
|---|---:|---:|
| Exact match toàn bộ chuỗi tool-call | 4/12 | 0/12 |
| Khớp toàn bộ tên tool theo thứ tự | 7/12 | 0/12 |
| Khớp tên và arguments toàn bộ chuỗi | 4/12 | 0/12 |
| Episode kết thúc | 12/12 | 12/12 |
| Timeout | 0 | 0 |
| Tổng thời gian replay trung bình | 6,69 s | 4,15 s |

Tỷ lệ action invalid chưa được đo riêng: status `finished` không chứng minh không có action invalid được xử lý lại. Final-state correctness và điểm chính thức: UNAVAILABLE. Mẫu 12 nhỏ, không đại diện toàn bộ dữ liệu và không đủ kết luận tổng quát. Không có bằng chứng E2 tốt hơn baseline trong phép thử này.

Kết quả từng mẫu, gold và tool-call được lưu trong `outputs/research/e2_e3_pilot_20261115/e0_e2_comparison.json`; output gốc nằm trong các thư mục `e0/` và `e2/` cùng cha. Các lệnh đã chạy nằm tại `outputs/research/e2_e3_pilot_20261115/evaluate.sh`. E2 checkpoint được tạo từ source `888d83970eafc34f1c4c1427a828ecbf7f41f780`; replay dùng source riêng nhánh `codex/e2-e3-pilot` với sửa single-ASR.

## ASR thật

- Gipformer 68M: đã có 5.036 lượt thật, artifact `outputs/research/e2_gipformer_20261115/train_top1.jsonl`, SHA256 `d061c99f1c914ea215e3d48b7289d44c2387b38e2babd27738f26dc5d4b6e98d`; không chạy lại toàn bộ.
- Gipformer 1.5: weights revision `dd9227dcd8705c13f33bdbe59728d546ab94480f`; smoke CPU 3/3 audio thành công, load 0,507 s, decode 0,068–0,107 s/audio; sau đó giải mã 82 lượt pilot. Không phải benchmark GPU.
- PhoWhisper-small: weights revision `a86b604c346caf7148c37512eafe783a16420adb`; smoke CPU 1/1 thành công, load 2,49 s, decode 2,36 s. Đang giải mã tập pilot, xem `phowhisper.log`.
- ChunkFormer RNNT: weights revision `9863592e00d6c02634c4ce2db8a4f3414f1ba7d6`; môi trường Python 3.10 riêng đang cài dependency. Giải mã thật: NOT TESTED tại thời điểm ghi. Xem `chunk_install.log`.

Pilot hiện chọn 16 group train phân bố đều và 12 group held-out cùng E0/E2, gồm 82 lượt audio cả lịch sử; danh sách lưu tại `outputs/research/e2_e3_pilot_20261115/pilot_ids.json`. Không dùng gold transcript làm output ASR. Trạng thái này là ảnh chụp lúc ghi; tiến trình CPU có thể vẫn đang chạy.

## E3

Artifact đủ bốn model, phân bố coverage, dataset và pilot train E3: NOT TESTED/BLOCKED cho đến khi ChunkFormer chạy thật và các record đồng nhất audio/model được ghép, audit. Không có checkpoint E3; không có điểm E3 hoặc Top-K ablation. Không tự khởi chạy huấn luyện E3. Trước khi train phải xác nhận đủ số record Multi-ASR thực được giữ; pilot tối đa 100 optimizer step, checkpoint 50, không tự nối dài.

## Cập nhật E3 pilot bốn ASR

Các model đều đã chạy thật trên cùng 82 lượt audio: 28 lượt hiện tại, 54 lượt lịch sử. Gipformer 68M được lấy từ artifact đã có với kiểm tra hash; Gipformer 1.5 82/82 thành công, decode trung bình 66,17 ms/lượt; PhoWhisper-small 82/82, trung bình 2.039 ms/lượt; ChunkFormer 82/82, trung bình 102,37 ms/lượt. Đây là đo CPU cục bộ, không phải hiệu năng GPU. ChunkFormer dùng môi trường Python 3.10 độc lập, torch/torchaudio 2.8.0; smoke load 2,27 s, decode 0,15 s.

Artifact đã ghép `outputs/research/e2_e3_pilot_20261115/four_asr_pilot.jsonl`, SHA256 `b89ca11d04a155323b8837dfe759c011d409e09b9df884bfe2e59d0635bc5577`; manifest cùng tên `.manifest.json`. Lệnh ghép:

```bash
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/merge_multi_asr_pilot.py \
 --base /home/heheboiz/data/zaloai/outputs/research/e2_gipformer_20261115/train_top1.jsonl \
 --input-dir /home/heheboiz/data/zaloai/outputs/research/e2_e3_pilot_20261115 \
 --models-dir /home/heheboiz/data/zaloai/models/research_e3 \
 --output /home/heheboiz/data/zaloai/outputs/research/e2_e3_pilot_20261115/four_asr_pilot.jsonl
```

Dataset pilot thực `outputs/research/e2_e3_pilot_20261115/e3_dataset/train`: 16 nhóm train, 12 nhóm validation tách biệt; 56 record sạch, 9 record conditioned; 48 record validation. Coverage: SUPPORTED 9, UNKNOWN 47, UNSUPPORTED 0. Manifest SHA256 `90bae76903275ec00bb60958e9f66b344c3f421afd02b04972104cc496d6c823`. Tokenizer MiniCPM thật: 11/65 prompt train và 7/48 prompt validation cắt giữa; 0 target bị loại. Các record `UNKNOWN` bị loại. Chỉ 9 record conditioned là quá ít cho thí nghiệm 50–100 optimizer step đáng tin cậy, nên **không khởi chạy E3 training**; E3 checkpoint/reload/metrics đều NOT TESTED. Cần thêm audio train thực và mở rộng coverage bằng bằng chứng an toàn trước pilot train.

Đã khởi chạy đánh giá E0/E2 cùng artifact bốn ASR K=3 trong `tmux` `lifemate_e2_e3_multieval_20261115`, log `outputs/research/e2_e3_pilot_20261115/evaluate_multi.log`; xem output `e0_multi_k3/` và `e2_multi_k3/` khi tiến trình hoàn tất. Không tự chạy training tiếp. Thử replay K=1..4 thành công về cấu trúc trên 12 mẫu; đây không phải kết quả accuracy Top-K.

Sửa code trong nhánh `codex/e2-e3-pilot`: replay một model xác minh audio, ghép các model độc lập với hash audio, và coverage chỉ dùng khớp transcript gốc làm bằng chứng, không đưa transcript gốc vào prompt. Kiểm thử tập trung mới nhất: `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q tests/test_conditioned_sft.py` — **30 passed**.

## Kết quả E0/E2 với cùng đầu vào bốn ASR K=3

Đã hoàn tất `evaluate_multi.sh` trên cùng 12 ID held-out và ToolEnv thật, dùng artifact bốn model đã lưu. E0 đạt exact match tool-call cục bộ **4/12**, tên tool toàn chuỗi **5/12**, có 9 episode `finished`, 2 `max_steps`, 1 `repeated_failure`; E2 đạt **0/12** exact match và **0/12** tên tool, 12 episode `finished`. Timeout 0 cho cả hai. Thời gian replay trung bình E0 **9,44 s**, E2 **1,77 s**, không bao gồm thời gian nhận dạng ASR. Tỷ lệ action invalid và final-state correctness: UNAVAILABLE. Kết quả 12 mẫu là thăm dò, không phải điểm chính thức.

Dự đoán từng mẫu: `outputs/research/e2_e3_pilot_20261115/e0_e2_multi_k3_comparison.json`; metrics gốc trong `e0_multi_k3/metrics.json` và `e2_multi_k3/metrics.json`. Đầu vào K=3 khác K=1 nên phải so sánh riêng theo từng chế độ. E3 không có checkpoint do chỉ có 9 record conditioned đủ bằng chứng; không có kết quả E3 hay accuracy ablation K=2/4.
