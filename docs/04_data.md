# Dữ liệu

**Người giữ:** TV1. **Đầu việc:** TASK-002, TASK-003. **Code:** `scripts/prepare_data.py` (TASK-002), `src/data.py` (TASK-003, chưa có).

## Thành phần làm gì
Chuẩn bị Tiny ImageNet thành ba tập train, validation, test không rò rỉ, lưu thành mảng cache để DataLoader đọc nhanh, và áp dụng augmentation khi train.

## Các tập dữ liệu

| Tập | Nguồn | Số ảnh | Dùng để |
|---|---|---|---|
| Train | 90% train gốc, phân tầng | 90.000 | Huấn luyện, có augmentation |
| Validation | 10% train gốc, phân tầng | 10.000 | Chọn siêu tham số, checkpoint |
| Test | Val gốc | 10.000 | Số cuối, chỉ sau khi chốt cấu hình |

**Ràng buộc:** train và validation không trùng ảnh; validation và test chỉ ToTensor + Normalize; test gốc không nhãn bị bỏ qua.

**Cache:** `prepare_data.py` xuất train/val/test thành mảng `uint8` (.npy) đã resize về `img_size`, DataLoader đọc từ mảng này vì giải mã JPEG và augmentation trên CPU thường là nút thắt. Cache nằm ở `data.cache_dir` (mặc định `data/cache/tiny_imagenet_64/`), gồm `{train,val,test}_x.npy` (uint8, N×64×64×3) và `{train,val,test}_y.npy` (int64, nhãn theo thứ tự `class_names.json`). Đo thật 2026-11-16: 1,35 GB; ghi file tạm rồi đổi tên nên chạy lại không làm hỏng.

**Chuẩn hóa và augmentation:**
- Mean/std của ImageNet, khai báo trong config `data.mean`, `data.std` (DEC-011).
- Train: RandomCrop(`img_size`, padding=4), lật ngang, RandAugment; RandomErasing mặc định tắt (DEC-012).
- Mixup/CutMix áp ở vòng lặp train, xem [06_training.md](06_training.md).

## FR-001: Chuẩn bị dữ liệu
- **Người sử dụng:** Thành viên nhóm, giảng viên chạy lại.
- **Mục đích:** Có dữ liệu đúng cấu trúc và không rò rỉ giữa các tập.
- **Đầu vào:** File Tiny ImageNet đã tải, đường dẫn trong `data.root`.
- **Luồng bình thường:** Tải (với `--download`) và giải nén, đọc nhãn `val/` từ `val_annotations.txt` (không di chuyển file), kiểm tra số lượng, xuất train/val/test thành mảng `uint8` (.npy) đã resize về `img_size`, in và lưu thống kê.
- **Quy tắc:** Train gốc chia 90/10 có phân tầng, seed cố định; val gốc là test; test gốc không nhãn bị bỏ qua.
- **Lỗi và biên:** Thiếu file hoặc sai số lượng thì dừng và báo rõ; chạy lại lần hai không làm hỏng dữ liệu.
- **Kết quả:** Mảng cache, `class_names.json`, `dataset_stats.json` (gồm thông tin cache), `samples.png`.

## Tiêu chí nghiệm thu

| Mã | Điều kiện ban đầu | Thao tác | Kết quả mong đợi |
|---|---|---|---|
| AC-001 | Có file Tiny ImageNet | Chạy `scripts/prepare_data.py` | Báo 200 lớp, 100.000 ảnh train gốc, 10.000 ảnh val gốc |
| AC-002 | Dữ liệu đã chuẩn bị | Tạo các tập | 90.000 train, 10.000 validation, 10.000 test; train và validation không trùng ảnh; mỗi lớp 450/50; có mảng cache khoảng 1,35 GB (tính tay); chạy lại không làm hỏng |
| AC-003 | Thiếu thư mục dữ liệu | Chạy `scripts/prepare_data.py` | Dừng, báo lỗi kèm hướng dẫn tải |

## Xử lý lỗi

| Tình huống | Cách xử lý |
|---|---|
| Thiếu dữ liệu, sai số lượng | `prepare_data` dừng và báo rõ |
| Chưa có mảng cache | `--quick` dùng dữ liệu giả và in cảnh báo; train thật báo lỗi (xem [06_training.md](06_training.md)) |

## Cách chạy

| Lệnh | Điều kiện | Trạng thái |
|---|---|---|
| `python scripts/prepare_data.py --download` | Máy mới, chưa có dữ liệu (tải 248 MB từ Stanford) | Đã chạy 2026-11-16 trên máy TV1 |
| `python scripts/prepare_data.py` | Đã có `data/tiny-imagenet-200/` | Đã chạy 2026-11-16 trên máy TV1: 64 phút (CPU, Windows) |

## Rủi ro
- Nguồn dữ liệu: trang Kaggle cần lời mời, nên tải từ Stanford (DEC-015). Nếu link Stanford chết trước khi máy GPU tải xong thì chuyển CIFAR-100 (DEC-001).
- Rò rỉ dữ liệu test: chỉ `evaluate.py` đọc test, chạy sau khi chốt cấu hình.
