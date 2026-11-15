# Huấn luyện

**Người giữ:** TV4. **Đầu việc:** TASK-006, TASK-011. **Code:** `src/train.py` (chưa có).

## Thành phần làm gì
Train ViT và CNN từ một lệnh, chọn checkpoint trên validation, chịu được ngắt phiên, và ghi log cùng checkpoint theo hợp đồng chung.

## Cấu hình huấn luyện
- AdamW cho cả hai mô hình trong so sánh chính (DEC-002).
- Batch 64, lr 2,5e-4 cho cả ViT và CNN, cùng batch (DEC-003). CNN chỉ đổi lr nếu chỉnh trên validation.
- Warmup 5 epoch rồi cosine về 1e-5, cập nhật theo từng iteration, cosine tính trên tổng 50 epoch.
- 50 epoch, đây là mức sàn (DEC-008).
- Weight decay 0,05, label smoothing 0,1, gradient clipping 1,0.
- Mixup/CutMix với xác suất 0,5 mỗi batch, bật/tắt và xác suất trong config.
- `nn.CrossEntropyLoss` nhận cả nhãn mềm.
- AMP khi có GPU.
- Chọn checkpoint tốt nhất theo validation Top-1, không đọc test.
- Train accuracy không dùng để nhận xét khi bật Mixup/CutMix.

Augmentation trên ảnh nằm ở [04_data.md](04_data.md).

## Hợp đồng

**Tên lần chạy:** `<run_name>` trong các đường dẫn là `<run_name trong YAML>_seed<seed>`, do `train.py` tự sinh để các seed không ghi đè nhau.

**Checkpoint** (`.pt`): `model`, `optimizer`, `scheduler`, `scaler` (trạng thái `GradScaler`, `None` khi không dùng AMP), `epoch`, `best_val_top1`, `config`, `seed`, `class_names`. Demo lấy tên lớp từ `class_names` và mean/std từ `config`. Checkpoint chỉ chứa tensor, dict, list, số và chuỗi; không lưu đối tượng Python hay trạng thái RNG của numpy, để `weights_only=True` nạp được; trạng thái RNG nếu cần lưu thành tensor.

**Log CSV** (`outputs/logs/<run_name>.csv`): `epoch, train_loss, train_acc, val_loss, val_top1, val_top5, lr, epoch_time_s`.

Bên dùng: [07_evaluation.md](07_evaluation.md), [08_demo.md](08_demo.md). Muốn đổi thì sửa ở đây và báo cả nhóm.

## FR-004: Huấn luyện
- **Người sử dụng:** TV4.
- **Mục đích:** Train được từ một lệnh và chịu được ngắt phiên.
- **Đầu vào:** Config, dữ liệu đã chuẩn bị.
- **Luồng bình thường:** Mỗi epoch train, đánh giá validation, ghi log, lưu checkpoint tốt nhất và cuối.
- **Quy tắc:** Chọn checkpoint theo validation Top-1, không đọc test; AdamW cho kết quả so sánh chính; lr của CNN mặc định 2,5e-4, nếu chỉnh thì chỉ dựa trên validation; tên lần chạy gồm cả seed.
- **Lỗi và biên:** `--resume` tiếp tục không trùng dòng log; phiên bị ngắt lúc lưu không hỏng checkpoint cũ; không có GPU thì dùng CPU.
- **Kết quả:** Checkpoint, log CSV, hình đường cong học.

## Tiêu chí nghiệm thu

| Mã | Điều kiện ban đầu | Thao tác | Kết quả mong đợi |
|---|---|---|---|
| AC-007 | Không có dữ liệu thật, CPU | Chạy train với `--quick` | Dùng dữ liệu giả và in cảnh báo; kết thúc không lỗi, có checkpoint và log CSV |
| AC-008 | Train bị ngắt sau epoch k | Chạy `--resume` | Tiếp tục từ epoch k+1, log không trùng dòng, lr schedule khớp với chạy liền mạch (chênh không quá 1e-8), `val_loss` ở epoch k+1 chênh không quá 10% so với epoch k (không đòi trọng số giống hệt, vì shuffle, augmentation và GPU không tất định) |

Ở các epoch đầu `val_loss` có thể giảm hơn 10% một cách bình thường, nên khi kiểm tra AC-008 nên ngắt sau giai đoạn warmup (sau epoch 5).

## Xử lý lỗi

| Tình huống | Cách xử lý |
|---|---|
| Chưa có mảng cache | `--quick` dùng dữ liệu giả và in cảnh báo; train thật báo lỗi |
| Train bị ngắt (máy GPU tắt, mất điện, phiên Kaggle/Colab hết giờ) | Ghi file tạm rồi đổi tên (DEC-005), chạy lại bằng `--resume` |
| Resume từ checkpoint cũ hơn log | Các dòng log từ epoch đó trở đi được ghi đè |
| Checkpoint từ người khác | Nạp với `weights_only=True` |

## Kiểm tra cần làm khi có code

| Kiểm tra | Mong đợi |
|---|---|
| Train bị ngắt rồi resume | Như AC-008 |
| Phiên bị ngắt lúc đang lưu checkpoint | Checkpoint cũ không hỏng |
| Nạp checkpoint bằng `weights_only=True` | Nạp được, có đủ `scaler` và `class_names` |
| Chạy hai seed khác nhau | Log, checkpoint và kết quả nằm ở hai tên khác nhau, không ghi đè |
| `--quick` khi chưa có cache; train thật khi chưa có cache | `--quick` dùng dữ liệu giả và cảnh báo; train thật báo lỗi |
| `--quick` trên máy GPU của thành viên | Chạy được, xác nhận đang dùng CUDA |
| VRAM đỉnh | ViT và CNN ở batch 64 đều không lỗi hết bộ nhớ trên 6GB; nếu lỗi thì dùng gradient accumulation hoặc giảm batch còn 32 với cả hai mô hình |

## Cách chạy

| Mục đích | Lệnh | Điều kiện | Trạng thái |
|---|---|---|---|
| Smoke test | `python -m src.train --config configs/vit_small.yaml --quick` | Không cần dữ liệu thật | Chưa có code |
| Train | `python -m src.train --config configs/vit_small.yaml` | Dữ liệu đã chuẩn bị, nên có GPU | Chưa có code |

**Lưu ý, chưa kiểm chứng:**
- Chạy `--quick` trên CPU trước khi tốn giờ GPU.
- Nếu máy GPU chạy Windows thì dùng `if __name__ == "__main__":`, kiểm tra `num_workers` trên Windows, không dùng `torch.compile`. Khi `num_workers=0` thì không bật `persistent_workers` (PyTorch báo lỗi).
- Lưu checkpoint thường xuyên và sao lưu ra ngoài máy GPU (Google Drive hoặc Kaggle Dataset), vì cả nhóm phụ thuộc vào một máy.

## Rủi ro

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Chỉ một máy GPU, các lần train phải chạy lần lượt | Không đủ thí nghiệm | Thời gian mỗi epoch là dự đoán (xem bảng giả định trong [01_define.md](01_define.md)); đo thật trước khi chốt số thí nghiệm; cắt việc theo [03_plan.md](03_plan.md); Kaggle hoặc Colab dự phòng |
| 6GB VRAM không đủ | Lỗi hết bộ nhớ | Đo VRAM đỉnh trước; gradient accumulation hoặc batch 32 cho cả hai mô hình |
| Máy GPU hỏng hoặc chủ máy bận | Trễ toàn bộ train | Sao lưu checkpoint ra ngoài máy; chuyển Kaggle hoặc Colab và `--resume` |
| Mixup/CutMix có thể lợi cho ViT hơn CNN | So sánh thiên lệch | Nêu trong báo cáo; ablation nếu còn thời gian |
