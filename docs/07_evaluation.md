# Đánh giá, attention và so sánh

**Người giữ:** TV5 (đánh giá, so sánh), TV2 (attention). **Đầu việc:** TASK-007, TASK-008, TASK-009, TASK-012. **Code:** `src/evaluate.py`, `scripts/error_analysis.py`, `src/attention.py`, `src/compare.py` (chưa có).

## Thành phần làm gì
Đo kết quả cuối trên test, phân tích lỗi, trực quan hóa attention của ViT và sinh bảng so sánh ViT với CNN. Mọi số đều lấy từ file.

## Quy tắc đánh giá
- Top-1, Top-5 trên test; per-class accuracy; top 20 cặp lớp nhầm.
- Đánh giá test sau khi chốt cấu hình, cùng tập test cho hai mô hình.
- Ablation chỉ báo kết quả trên validation. Test chỉ dùng cho các mô hình chính cuối cùng (ViT và CNN).
- Chênh lệch dưới khoảng 1–1,5 điểm giữa hai mô hình không đủ để kết luận (xem yêu cầu chất lượng trong [01_define.md](01_define.md)).

## Hợp đồng

**Kết quả JSON** (`outputs/results/<run_name>.json`): `run_name, model, seed, params, macs, top1, top5, epoch_time_s, inference_img_per_s, peak_vram_mb`. Chưa có số thì để `null`. `<run_name>` theo quy ước ở [06_training.md](06_training.md).

Đầu vào là checkpoint theo hợp đồng ở [06_training.md](06_training.md) và mô hình theo [05_model.md](05_model.md).

## FR-005: Đánh giá và phân tích lỗi
- **Người sử dụng:** TV5.
- **Mục đích:** Có số cuối và hiểu vì sao mô hình sai.
- **Đầu vào:** Checkpoint đã chốt, tập test.
- **Luồng bình thường:** Tính chỉ số, xuất per-class accuracy, cặp lớp nhầm và lưới ảnh sai.
- **Lỗi và biên:** Checkpoint không khớp mô hình thì báo lỗi.
- **Kết quả:** JSON kết quả, CSV per-class, hình lỗi.

## FR-006: Trực quan hóa attention
- **Người sử dụng:** TV2, người đọc báo cáo.
- **Mục đích:** Thấy ViT nhìn vào vùng nào.
- **Đầu vào:** Checkpoint ViT, ảnh test.
- **Luồng bình thường:** Lấy attention, tính CLS map và rollout, phủ heatmap lên ảnh.
- **Quy tắc:** Chọn cả ảnh dự đoán đúng và sai, nhiều lớp khác nhau.
- **Lỗi và biên:** Lưới phải đúng với mọi `patch_size` hợp lệ.
- **Kết quả:** Hình trong `outputs/figures/attention/`.

## FR-007: So sánh
- **Người sử dụng:** TV5, người đọc báo cáo.
- **Mục đích:** Có bảng so sánh ViT và CNN.
- **Đầu vào:** JSON kết quả của các mô hình.
- **Luồng bình thường:** Đọc JSON, sinh bảng và biểu đồ.
- **Quy tắc:** Mọi số lấy từ file; bảng bắt buộc có cột FLOPs/MACs (đo nếu làm được, ghi "chưa có" nếu chưa).
- **Lỗi và biên:** Số chưa có thì hiển thị "chưa có", không tự điền.
- **Kết quả:** `comparison.md`, `comparison.csv`, biểu đồ.

## Tiêu chí nghiệm thu

| Mã | Điều kiện ban đầu | Thao tác | Kết quả mong đợi |
|---|---|---|---|
| AC-009 | Có checkpoint | Chạy `src.evaluate` | Có Top-1, Top-5, per-class CSV, top 20 cặp nhầm, hình ảnh sai cho cả hai mô hình |
| AC-010 | Có checkpoint ViT | Chạy `src.attention` | Có hình cho ít nhất 12 ảnh, gồm cả đúng và sai |
| AC-011 | Có JSON kết quả | Chạy `src.compare` | Bảng khớp với JSON, có đủ khóa kể cả `macs`, `peak_vram_mb`; số thiếu hiện "chưa có" |

## Cách chạy

| Mục đích | Lệnh | Điều kiện | Trạng thái |
|---|---|---|---|
| Đánh giá | `python -m src.evaluate` | Có checkpoint | Chưa có code |
| Attention | `python -m src.attention` | Có checkpoint ViT | Chưa có code |
| So sánh | `python -m src.compare` | Có JSON kết quả | Chưa có code |

## Rủi ro
- Rò rỉ dữ liệu test: chỉ `evaluate.py` đọc test, chạy sau khi chốt cấu hình.
