# Mô hình

**Người giữ:** TV2 (ViT), TV3 (CNN, đo FLOPs/MACs). **Đầu việc:** TASK-004, TASK-005. **Code:** `src/models/__init__.py` (`build_model`), `src/models/vit.py`, `src/models/cnn.py`.

**TASK-004:** hoàn thành, TV1 review và gộp vào `main` ngày 2026-11-17 qua [PR #2](https://github.com/QuocAn118/Classify-images-using-Vision-Transformer/pull/2). Nhánh nguồn của PR là `BinhNg1208:thanhbinh` trên fork cá nhân do TV2 chưa có quyền push repo gốc. Describe tại [Issue #1](https://github.com/QuocAn118/Classify-images-using-Vision-Transformer/issues/1).

## Thành phần làm gì
Cung cấp ViT tự cài (có lấy được attention) và một ResNet nhỏ có số tham số chênh không quá 5% so với ViT, để so sánh công bằng.


## Cấu hình

| Mô hình | Cấu hình | Số tham số |
|---|---|---:|
| ViT-small | Ảnh 64, patch 8, dim 256, depth 8, heads 8, MLP ratio 4, drop path 0,1, dropout 0 | 6.436.296 |
| ResNet-small | BasicBlock, BatchNorm, widths [64, 128, 256, 512], blocks [1, 2, 2, 1], stem stride 2 | 6.476.808 |

CNN có số tham số chênh khoảng 0,63% so với ViT, đạt yêu cầu AC-006 (không quá 5%).

### Chi phí tính toán

Đo bằng `torch.utils.flop_counter.FlopCounterMode`, model ở chế độ `eval()`, batch size 1, input `(1, 3, 64, 64)`.

| Mô hình | FLOPs / ảnh 64×64 |
|---|---:|
| ResNet-small | 809.050.112 |
| ViT-small | 824.283.136 |

ResNet-small thấp hơn ViT-small 15.233.024 FLOPs/ảnh, tương đương khoảng 1,85%.

**Lưu ý:** Đây là số FLOPs được đo bằng `FlopCounterMode`, không phải MACs. Chưa có số MACs được xác minh bằng công cụ đo riêng.

Bằng chứng: `../outputs/results/task005_cnn_checks.json`.

## Hợp đồng

```python
from src.models import build_model
model = build_model(cfg)                  # cfg từ load_config; chọn theo cfg["model"]["name"]
logits = model(x)                         # x: (B,3,H,W), logits: (B,num_classes)
logits, attns = model(x, return_attn=True)  # chỉ ViT; attns: list[(B,heads,N+1,N+1)], mỗi layer một phần tử
```

- `build_model(cfg)` là cách duy nhất để train, đánh giá, attention và demo tạo mô hình; không ai import thẳng lớp ViT hay CNN.
- `model.name: vit` → ViT (TV2, TASK-004 tạo hàm này và nhánh `vit`); `model.name: resnet_small` → CNN (TV3, TASK-005 thêm nhánh). Tên khác thì báo `ValueError` liệt kê tên hợp lệ.
- ViT đọc `img_size`, `num_classes` từ `cfg["data"]`, các tham số còn lại từ `cfg["model"]`.
- ResNet-small đọc `num_classes` từ `cfg["data"]`, `widths`, `blocks`, `stem_stride` từ `cfg["model"]`.
- `build_model(cfg)` không gọi `match_param_count()` khi khởi tạo ResNet-small.
- `match_param_count()` được giữ lại để tìm kiếm và kiểm tra cấu hình phù hợp độc lập.
- Thứ tự gộp vào `main`: TASK-004 → TASK-005 → TASK-006 (TV1 chốt 2026-11-17).

Bên dùng: [06_training.md](06_training.md), [07_evaluation.md](07_evaluation.md), [08_demo.md](08_demo.md). Muốn đổi thì sửa ở đây và báo cả nhóm.


## Cài đặt ViT hiện có (TASK-004)

`build_model(cfg)` hỗ trợ cả `model.name: vit` và `model.name: resnet_small`. ViT được triển khai trong TASK-004; ResNet-small được bổ sung trong TASK-005. Hai mô hình đều được khởi tạo thông qua `build_model(cfg)`, không cần import trực tiếp lớp mô hình từ bên gọi.

Luồng một batch mặc định:

1. `PatchEmbedding`: Conv2d kernel/stride 8 biến `(B,3,64,64)` thành `(B,64,256)`.
2. Ghép CLS token học được thành 65 token, cộng positional embedding `(1,65,256)` rồi áp dropout.
3. Qua 8 `TransformerEncoderBlock` Pre-LN: LayerNorm → multi-head attention → residual, rồi LayerNorm → MLP GELU → residual. DropPath bỏ nhánh theo từng ảnh khi train, xác suất tăng từ 0 đến 0,1 theo độ sâu; ở eval thì tắt.
4. LayerNorm cuối → lấy CLS → Linear cho logits `(B,200)`, chưa softmax để dùng CrossEntropyLoss.

Nhánh thường gọi `scaled_dot_product_attention` và không giữ danh sách attention. Nhánh `return_attn=True` tính softmax của `QKᵀ / sqrt(head_dim)` tường minh, trả weights **trước dropout** cho từng layer. Khi `eval()`, cả dropout và DropPath đều tắt; hai nhánh phải khớp logits trong ngưỡng AC-005. Lưới patch lấy từ config; ảnh input phải đúng kích thước RGB trong config. Cấu hình kích thước, số head, tỷ lệ MLP và xác suất không hợp lệ được báo bằng `ValueError`.

Ví dụ dùng từ bên gọi:

```python
import torch
from src.models import build_model
from src.utils import load_config

cfg = load_config("configs/vit_small.yaml")
model = build_model(cfg).eval()
with torch.no_grad():
    logits, attns = model(torch.randn(2, 3, 64, 64), return_attn=True)
```

## Cài đặt ResNet-small (TASK-005)

`build_model(cfg)` hỗ trợ `model.name: resnet_small` để tạo mô hình CNN baseline.

ResNet-small sử dụng BasicBlock và BatchNorm. Cấu hình được khai báo trong `configs/cnn_baseline.yaml`:

- `widths`: `[64, 128, 256, 512]`
- `blocks`: `[1, 2, 2, 1]`
- `stem_stride`: `2`
- Số lớp đầu ra: `200`

`build_model()` đọc `num_classes` từ `cfg["data"]` và các tham số kiến trúc từ `cfg["model"]`. Các danh sách `widths` và `blocks` trong YAML được chuyển thành tuple trước khi khởi tạo ResNet-small.

Việc khởi tạo mô hình không gọi `match_param_count()`, tránh phải tìm kiếm cấu hình mỗi lần tạo model. Hàm `match_param_count()` vẫn được giữ lại để tìm kiếm và kiểm tra cấu hình ResNet đáp ứng giới hạn sai lệch số tham số.

Với cấu hình hiện tại, ResNet-small có 6.476.808 tham số, ViT-small có 6.436.296 tham số. Sai lệch tương đối khoảng 0,63%, đáp ứng AC-006.

## Bằng chứng kiểm tra TASK-004

Ngày 2026-11-17, Windows, Python 3.12.10, torch 2.15.0+cpu; không có CUDA. Các input mô hình đều là dữ liệu giả, không đọc tập test thật.

| Kiểm tra | Lệnh / bằng chứng | Kết quả |
|---|---|---|
| AC-004, AC-005, số tham số, cấu hình lưới khác, lỗi cấu hình, dropout/DropPath, backward hai nhánh, nạp state dict với `weights_only=True` | `python -m pytest tests/test_models.py -q -W error` | 23 passed |
|  Hồi quy toàn repo | `python -m pytest -q -W error` | 48 passed |
| Cấu hình ViT mặc định, forward và một bước AdamW trên CPU | [task004_vit_checks.json](../outputs/results/task004_vit_checks.json) | 6.436.296 tham số; logits `(2,200)`; 8 attention `(2,8,65,65)`; gradient mọi tham số hữu hạn; trọng số head thay đổi sau optimizer step |
| Sai khác logits / sai số tổng hàng attention ở eval | Cùng file JSON | Tối đa khoảng 7,15e-7 / 2,38e-7, đều dưới 1e-4 |

Chưa kiểm tra CUDA, AMP hay VRAM đỉnh. Chưa có kết quả huấn luyện hoặc đánh giá accuracy. `src.train --quick` thuộc TASK-006; các kiểm tra trong TASK-004 chỉ smoke test trực tiếp mô hình ViT.

## Bằng chứng kiểm tra TASK-005

Các kiểm tra được thực hiện trên Windows, Python 3.12.10, torch 2.15.0+cpu. Input sử dụng dữ liệu giả, kích thước `(1,3,64,64)` khi đo FLOPs.

| Kiểm tra | Lệnh / bằng chứng | Kết quả |
|---|---|---|
| Kiểm tra model, số tham số, output shape và tìm cấu hình | `python -m pytest tests/test_models.py -q -W error` | 27 passed |
| ResNet-small so với ViT-small | `outputs/results/task005_cnn_checks.json` | CNN: 6.476.808 tham số; ViT: 6.436.296 tham số; sai lệch khoảng 0,63% |
| Output ResNet-small | Test input `(2,3,64,64)` | Logits `(2,200)` |
| Tìm cấu hình có số tham số gần mục tiêu | Test `match_param_count(6_436_296)` | Sai lệch không quá 5% |
| Từ chối mục tiêu tham số quá nhỏ | Test `match_param_count(1_000)` | Báo `ValueError` |
| Đo FLOPs ResNet-small | `outputs/results/task005_cnn_checks.json` | 809.050.112 FLOPs/ảnh |
| Đo FLOPs ViT-small | `outputs/results/task005_cnn_checks.json` | 824.283.136 FLOPs/ảnh |

Phương pháp đo FLOPs sử dụng `torch.utils.flop_counter.FlopCounterMode`, model ở chế độ `eval()`, batch size 1 và input `(1,3,64,64)`. ResNet-small thấp hơn ViT-small 15.233.024 FLOPs/ảnh, tương đương khoảng 1,85%.

Các số liệu trên là FLOPs được đo bằng công cụ nêu trên; MACs chưa được đo riêng nên không báo cáo số MACs.

## FR-002: ViT tự cài

- **Người sử dụng:** TV4 (train), TV2 (attention), TV5 (demo).
- **Mục đích:** Có mô hình chính đúng đề và lấy được attention.
- **Đầu vào:** Tensor ảnh (B,3,H,W), cấu hình trong YAML.
- **Luồng bình thường:** Patch embedding, ghép CLS và positional embedding, qua các encoder block, lấy CLS rồi phân loại.
- **Quy tắc:** Không dùng mô hình dựng sẵn từ thư viện (DEC-006); kích thước lưới tính từ `img_size // patch_size`; khi train dùng `scaled_dot_product_attention`, khi `return_attn=True` dùng nhánh tính tường minh.
- **Lỗi và biên:** `img_size` không chia hết cho `patch_size` thì báo lỗi; attention lấy trước dropout.
- **Kết quả:** Logits, và danh sách attention khi `return_attn=True`.

## FR-003: CNN cùng số tham số

- **Người sử dụng:** TV4 (train), TV5 (demo).
- **Mục đích:** Cung cấp mô hình CNN baseline để so sánh với ViT.
- **Đầu vào:** Tensor ảnh `(B,3,H,W)` và cấu hình trong YAML.
- **Luồng bình thường:** `build_model(cfg)` đọc cấu hình `resnet_small` từ `configs/cnn_baseline.yaml` và tạo ResNet-small.
- **Quy tắc:** Sử dụng BasicBlock, BatchNorm; cấu hình widths `[64,128,256,512]`, blocks `[1,2,2,1]`, stem stride 2. Số tham số chênh không quá 5% so với ViT.
- **Lỗi và biên:** Cấu hình ResNet không hợp lệ thì báo lỗi; `match_param_count()` báo `ValueError` nếu không tìm được cấu hình đáp ứng sai số cho phép.
- **Kết quả:** Logits `(B,num_classes)` và mô hình có số tham số đáp ứng AC-006.

## Tiêu chí nghiệm thu

| Mã | Điều kiện ban đầu | Thao tác | Kết quả mong đợi |
|---|---|---|---|
| AC-004 | ViT cấu hình mặc định | Forward input (2,3,64,64) | Output (2,200) |
| AC-005 | ViT ở `eval()` | Gọi `return_attn=True` | Mỗi layer có attention (B,heads,N+1,N+1), mỗi hàng tổng bằng 1 (sai số 1e-4); logits hai nhánh chênh tuyệt đối tối đa 1e-4 |
| AC-006 | ViT và CNN mặc định | Đếm tham số bằng `count_params` | ViT: 6.436.296; CNN: 6.476.808; sai lệch khoảng 0,63%, đạt yêu cầu |

## Rủi ro

- 6GB VRAM có thể không đủ, nhất là khi chạy CNN và ablation patch 4: cần đo VRAM đỉnh trước khi train thật (xem [06_training.md](06_training.md)).
- ViT có thể cho kết quả thấp hơn CNN; kết quả cần được báo cáo trung thực và phân tích nguyên nhân.
- Số FLOPs phụ thuộc vào phương pháp đếm và các phép toán được công cụ hỗ trợ; khi so sánh cần giữ nguyên input shape, batch size và công cụ đo.
