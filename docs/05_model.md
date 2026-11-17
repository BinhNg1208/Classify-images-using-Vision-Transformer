# Mô hình

**Người giữ:** TV2 (ViT), TV3 (CNN, đo FLOPs/MACs). **Đầu việc:** TASK-004, TASK-005. **Code:** `src/models/__init__.py` (`build_model`), `src/models/vit.py`, `src/models/cnn.py` (chưa có).

## Thành phần làm gì
Cung cấp ViT tự cài (có lấy được attention) và một ResNet nhỏ có số tham số chênh không quá 5% so với ViT, để so sánh công bằng.

## Cấu hình

| Mô hình | Cấu hình | Số tham số |
|---|---|---|
| ViT-small | Ảnh 64, patch 8, dim 256, depth 8, heads 8, MLP ratio 4, drop path 0,1, dropout 0 | 6.436.296 (tính tay, đã kiểm tra lại bằng phép tính, chưa xác minh bằng code) |
| ResNet nhỏ | BasicBlock, BatchNorm, stem 3x3 stride 1 nhưng tầng đầu tiên downsample sớm, hoặc stem stride 2 (tránh feature map 64x64 ở nhiều tầng, vì tốn VRAM và thời gian; DEC-007); cấu hình cuối do `match_param_count` chọn | Trong khoảng 5% so với ViT (chưa xác minh) |

Cùng cỡ tham số nhưng chi phí tính toán có thể chênh nhiều. **Dự đoán, chưa đo bằng code:** ViT khoảng 0,43 GMACs (khoảng 0,86 GFLOPs), CNN khoảng 1–1,5 GMACs. Số của CNN tính cho stem cũ chưa downsample; thiết kế downsample sớm có thể thấp hơn, phải đo. Khi đo, ghi rõ đơn vị (MACs hay FLOPs) và dùng cùng một đơn vị cho cả hai mô hình.

## Hợp đồng

```python
from src.models import build_model
model = build_model(cfg)                  # cfg từ load_config; chọn theo cfg["model"]["name"]
logits = model(x)                         # x: (B,3,H,W), logits: (B,num_classes)
logits, attns = vit(x, return_attn=True)  # chỉ ViT; attns: list[(B,heads,N+1,N+1)], mỗi layer một phần tử
```

- `build_model(cfg)` là cách duy nhất để train, đánh giá, attention và demo tạo mô hình; không ai import thẳng lớp ViT hay CNN.
- `model.name: vit` → ViT (TV2, TASK-004 tạo hàm này và nhánh `vit`); `model.name: resnet_small` → CNN (TV3, TASK-005 thêm nhánh). Tên khác thì báo `ValueError` liệt kê tên hợp lệ.
- Đọc `img_size`, `num_classes` từ `cfg["data"]`, phần còn lại từ `cfg["model"]`; không hardcode.
- Thứ tự gộp vào `main`: TASK-004 → TASK-005 → TASK-006 (TV1 chốt 2026-11-17).

Bên dùng: [06_training.md](06_training.md), [07_evaluation.md](07_evaluation.md), [08_demo.md](08_demo.md). Muốn đổi thì sửa ở đây và báo cả nhóm.

## FR-002: ViT tự cài
- **Người sử dụng:** TV4 (train), TV2 (attention), TV5 (demo).
- **Mục đích:** Có mô hình chính đúng đề và lấy được attention.
- **Đầu vào:** Tensor ảnh (B,3,H,W), cấu hình trong YAML.
- **Luồng bình thường:** Patch embedding, ghép CLS và positional embedding, qua các encoder block, lấy CLS rồi phân loại.
- **Quy tắc:** Không dùng mô hình dựng sẵn từ thư viện (DEC-006); kích thước lưới tính từ `img_size // patch_size`; khi train dùng `scaled_dot_product_attention`, khi `return_attn=True` dùng nhánh tính tường minh.
- **Lỗi và biên:** `img_size` không chia hết cho `patch_size` thì báo lỗi; attention lấy trước dropout.
- **Kết quả:** Logits, và danh sách attention khi `return_attn=True`.

## FR-003: CNN cùng số tham số
- **Người sử dụng:** TV4, TV5.
- **Mục đích:** Có baseline so sánh công bằng.
- **Đầu vào:** Số tham số mục tiêu của ViT.
- **Luồng bình thường:** `match_param_count` tìm cấu hình ResNet nhỏ và in kết quả.
- **Quy tắc:** Stem theo DEC-007; cấu hình cuối do `match_param_count` chọn; chênh lệch không quá 5%.
- **Lỗi và biên:** Không tìm được cấu hình đạt thì báo lỗi.
- **Kết quả:** Mô hình và số tham số thực tế.

## Tiêu chí nghiệm thu

| Mã | Điều kiện ban đầu | Thao tác | Kết quả mong đợi |
|---|---|---|---|
| AC-004 | ViT cấu hình mặc định | Forward input (2,3,64,64) | Output (2,200) |
| AC-005 | ViT ở `eval()` | Gọi `return_attn=True` | Mỗi layer có attention (B,heads,N+1,N+1), mỗi hàng tổng bằng 1 (sai số 1e-4); logits hai nhánh chênh tuyệt đối tối đa 1e-4 |
| AC-006 | ViT và CNN mặc định | Đếm tham số | Chênh không quá 5% |

## Rủi ro
- 6GB VRAM không đủ, nhất là CNN và ablation patch 4: đo VRAM đỉnh bằng `--quick` trước khi train thật (xem [06_training.md](06_training.md)).
- ViT thua xa CNN: kết quả kém nhưng vẫn hợp lệ, báo cáo trung thực và phân tích nguyên nhân.
