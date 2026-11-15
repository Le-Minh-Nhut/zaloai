# Tạo dữ liệu bốn ASR toàn bộ — trạng thái 15/11/2026

**HOÀN TẤT:** artifact bốn ASR đã được tạo và kiểm tra độc lập. Không chạy LLM training hoặc tiếp tục E2. Các dòng tiến độ bên dưới là ảnh chụp lịch sử; xem phần kết quả cuối ở cuối tài liệu.

- Source worker `ff53fdf9fe148c5baf5cc914c739afa533b07564`; merger `fa59274827acb82c844cd1bfd9397424c3d7fc75`. Worktree thực thi `/tmp/lifemate-e2-e3-pilot`.
- 5.036 audio train được kiểm tra hash so với artifact Gipformer 68M gốc, gồm 1.545 lượt hiện tại và 3.491 lượt lịch sử. Artifact gốc SHA256 `d061c99f1c914ea215e3d48b7289d44c2387b38e2babd27738f26dc5d4b6e98d` được tái sử dụng; không chạy lại model này.
- Pilot 82 lượt đối chiếu theo `(sample_id, turn_id, audio_sha256)`, model ID, revision, identity artifact và decoder. Các worker tái sử dụng kết quả pilot khi gặp lượt tương ứng trong thứ tự nguồn; số hit tăng dần, không phải mất cache.
- Tại một thời điểm kiểm tra sau khoảng 2 phút 48 giây: Gipformer 1.5 **2.217 thành công, 0 lỗi, 49 hit**; PhoWhisper **62 thành công, 0 lỗi, 9 hit**; ChunkFormer **1.079 thành công, 0 lỗi, 42 hit**. Số này thay đổi liên tục, xem `progress.json` để biết hiện tại. Chưa có số 5.036/5.036 cho các model mới.
- Phiên `tmux`: `lifemate_full_gipformer_15_20261115`, `lifemate_full_phowhisper_small_20261115`, `lifemate_full_chunkformer_rnnt_20261115`. Cả ba dùng CPU, cùng decoder/precision như pilot; không khởi chạy GPU ASR hay training. PhoWhisper chậm hơn nhiều trong quan sát ban đầu; chưa đo ETA đủ chắc chắn.
- `lifemate_full_asr_merge_20261115` chờ ba PID kết thúc, sau đó chạy `scripts/merge_full_multi.py`. Merger yêu cầu journal có completion hash hợp lệ, đủ 5.036 key, kiểm tra identity và hash audio, giữ lỗi nếu có, rồi ghi artifact theo thứ tự nguồn và thực hiện kiểm tra reader/replay. Nếu worker chết hoặc journal thiếu, merger báo lỗi và không tuyên bố hoàn thành; không tự retry vô hạn.
- Worker lưu từng kết quả có checksum, `fsync`, khóa model và metadata; resume xác minh identity và bỏ phần dòng cuối chưa hoàn chỉnh. Các lượt lỗi không bị thử lại tự động; có thể dùng `--retry-failures` sau khi xác định nguyên nhân. Không ghi đè transcript thành công.
- Regression CPU đã chạy: `PYTHONPATH=src /tmp/cudaenv/bin/python -m pytest -q tests/test_full_multi_worker.py tests/test_conditioned_sft.py` → **34 passed**. Đây không phải kết quả hoàn tất ASR.

Kiểm tra tiến độ thực tế:

```bash
ROOT=/home/heheboiz/data/zaloai/outputs/research/e3_four_asr_full_20261115
for m in gipformer_15 phowhisper_small chunkformer_rnnt; do
  echo "$m"; cat "$ROOT/$m/progress.json" 2>/dev/null || true
done
tmux list-sessions
```

Tiếp tục worker đã dừng (không chạy khi worker cùng model còn hoạt động; file lock sẽ từ chối duplicate):

```bash
cd /tmp/lifemate-e2-e3-pilot
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
ROOT=/home/heheboiz/data/zaloai
OUT="$ROOT/outputs/research/e3_four_asr_full_20261115"
/tmp/cudaenv/bin/python scripts/transcribe_full_multi.py --root "$ROOT" --output-dir "$OUT" --model gipformer_15 --source-commit ff53fdf
/tmp/cudaenv/bin/python scripts/transcribe_full_multi.py --root "$ROOT" --output-dir "$OUT" --model phowhisper_small --source-commit ff53fdf
"$ROOT/outputs/research/e2_e3_pilot_20261115/chunkformer_env/bin/python" scripts/transcribe_full_multi.py --root "$ROOT" --output-dir "$OUT" --model chunkformer_rnnt --source-commit ff53fdf
```

Sau khi tất cả hoàn thành, nếu merger tự động đã dừng vì một worker lỗi, chạy lại:

```bash
PYTHONPATH=src /tmp/cudaenv/bin/python scripts/merge_full_multi.py --root "$ROOT" --output-dir "$OUT"
cat "$OUT/execution_report.json"
PYTHONPATH=src /tmp/cudaenv/bin/python -c 'import sys; from lifemate.data.conditioned_sft import load_records; i,m=load_records(sys.argv[1]); print(len(i),m["sha256"])' "$OUT/four_asr_train.jsonl"
```

`execution_report.json` chỉ được tạo sau merge và kiểm tra; chứa counts từng model, coverage 0–4, SHA256 và trạng thái replay K=1–4. Nếu có ASR thất bại, replay nghiêm ngặt có thể báo BLOCKED cho mẫu đó; không thay failure bằng gold. Artifact cuối cùng mong đợi tại `outputs/research/e3_four_asr_full_20261115/four_asr_train.jsonl`, **đã tồn tại và được kiểm tra như phần kết quả cuối**.

## Thử nhanh GPU PhoWhisper

Đã đo riêng ba audio pilot bằng GPU float16: thời gian decode 0,855/0,227/0,297 giây, ba transcript trùng chính xác CPU float32; peak CUDA allocated 568 MiB. Kết quả đo nằm tại `outputs/research/e3_four_asr_full_20261115/phowhisper_gpu_probe.json`, không nhập vào artifact. Ba mẫu chưa chứng minh parity toàn tập; worker chính tiếp tục cấu hình CPU float32 của pilot để giữ một decoder identity nhất quán, không dừng hay lặp lại công việc đã hoàn tất. Chưa đo batching.


## Kết quả cuối đã kiểm chứng

Artifact: `outputs/research/e3_four_asr_full_20261115/four_asr_train.jsonl`; manifest cùng tên thêm `.manifest.json`; SHA256 **`b219aaef1feab1bc2313994ca602d0ca3cf94c94caea2aaa57257bd092354b00`**. Có đúng **5.036 record**, gồm **1.545 lượt hiện tại** và **3.491 lượt lịch sử**, theo đúng thứ tự nguồn; không có key trùng.

| Model | Thành công | Thất bại | Tái sử dụng | Mới giải mã | Thời gian worker |
|---|---:|---:|---:|---:|---:|
| Gipformer 68M | 5.036 | 0 | 5.036 | 0 | artifact đã có |
| Gipformer 1.5 | 5.036 | 0 | 82 | 4.954 | 362,35 s |
| PhoWhisper-small | 5.036 | 0 | 82 | 4.954 | 9.693,03 s |
| ChunkFormer RNNT | 5.036 | 0 | 82 | 4.954 | 686,46 s |

Tổng **20.144/20.144 kết quả**, coverage **4/4: 5.036**, **3/4, 2/4, 1/4, 0/4: 0**. Thời gian worker là thời gian quan sát từng process; các process chạy đồng thời. Revision ba model mới: `dd9227dcd8705c13f33bdbe59728d546ab94480f`, `a86b604c346caf7148c37512eafe783a16420adb`, `9863592e00d6c02634c4ce2db8a4f3414f1ba7d6` theo thứ tự bảng.

Đã đọc `load_records`, xác nhận SHA manifest, checksum từng record, thứ tự/key/audio hash và số lượng; đối chiếu riêng toàn bộ journal với các record đã ghép và 82 kết quả pilot mỗi model. Replay `ArtifactReplay` K=1–4 PASS trên 12 mẫu held-out, không giải mã lại audio. Số candidate có thể thấp hơn K khi transcript trùng nhau; bốn output thô vẫn được giữ. Tất cả kết quả là ASR thật, không phải đánh giá chất lượng tool-call, training hoặc điểm thi.

Báo cáo máy đọc: `outputs/research/e3_four_asr_full_20261115/execution_report.json`. Không tạo dataset SFT E3 mới và không khởi chạy training trong nhiệm vụ này.
