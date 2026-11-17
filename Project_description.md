# Project description: ViT phân loại ảnh trên Tiny ImageNet

Đồ án môn Học sâu, đề số 9, nhóm 5 người, hạn nộp 2026-11-25 (đã gia hạn từ 2026-11-20). File này tóm tắt Define, Design, Plan; chi tiết nằm trong `docs/`, mỗi thứ chỉ viết một lần ở đó.

## Thứ tự đọc

Người mới đọc file này trước, rồi đọc `docs/` theo số:

1. `01_define.md`: bài toán, phạm vi, yêu cầu.
2. `02_architecture.md`: các thành phần ghép với nhau thế nào.
3. `03_plan.md`: ai làm gì, mốc nào.
4. `04_data.md` đến `08_demo.md`: từng thành phần theo luồng dữ liệu. Ai giữ phần nào thì đọc kỹ file đó.
5. `09_decisions.md`, `10_work_log.md`: tra cứu khi cần biết vì sao chọn một phương án, hoặc đã làm gì gần đây.

## Define
- **Bài toán:** Phân loại ảnh 64x64 vào 200 lớp Tiny ImageNet bằng Vision Transformer tự cài, so sánh với một CNN có số tham số chênh không quá 5%, và trực quan hóa attention.
- **Người dùng:** Sinh viên, người nghiên cứu thị giác máy tính, giảng viên chấm bài.
- **Phạm vi:** ViT, CNN cùng số tham số, attention, demo Gradio, phân tích lỗi. Ngoài phạm vi: ViT pretrained, deepfake, triển khai production.
- **Thành công khi:** Có Top-1, Top-5 trên test cho cả hai mô hình; tham số chênh không quá 5%; có hình attention cho ít nhất 12 ảnh.
- **Ràng buộc:** khoảng 10 ngày tính từ 15/11, một máy GPU 6GB VRAM, không dùng test để chọn cấu hình, không bịa số liệu.

Chi tiết, yêu cầu chức năng và giả định còn mở: [docs/01_define.md](docs/01_define.md).

## Design
- **Luồng:** Tiny ImageNet → chuẩn bị dữ liệu và cache → train ViT, CNN → checkpoint → đánh giá, attention, demo → bảng so sánh.
- **Mô hình:** ViT-small (patch 8, dim 256, depth 8, khoảng 6,4 triệu tham số, tính tay) và ResNet nhỏ khớp số tham số.
- **Huấn luyện:** AdamW, batch 64, lr 2,5e-4, 50 epoch (mức sàn), chọn checkpoint trên validation.

| Thành phần | Tài liệu | Người giữ |
|---|---|---|
| Kiến trúc tổng, config, môi trường | [docs/02_architecture.md](docs/02_architecture.md) | TV1 |
| Dữ liệu | [docs/04_data.md](docs/04_data.md) | TV1 |
| Mô hình | [docs/05_model.md](docs/05_model.md) | TV2, TV3 |
| Huấn luyện | [docs/06_training.md](docs/06_training.md) | TV4 |
| Đánh giá, attention, so sánh | [docs/07_evaluation.md](docs/07_evaluation.md) | TV5, TV2 |
| Demo | [docs/08_demo.md](docs/08_demo.md) | TV5 |
| Quyết định thiết kế | [docs/09_decisions.md](docs/09_decisions.md) | Cả nhóm |

## Plan
- **Mốc:** duyệt tài liệu và nộp mẫu đăng ký → khung repo → `--quick` chạy thông → train thật → số cuối trên test → nộp bài 24/11 (25/11 dự phòng).
- **Trạng thái:** theo dõi trên GitHub Issues và Projects. Đã xong khung repo, dữ liệu, DataLoader (TASK-001 đến 003) và ViT (TASK-004); đang làm CNN, vòng lặp train và attention.
- **Đầu việc tiếp theo:** TASK-005 (TV3, CNN) → TASK-006 (TV4, train); song song TASK-008 (TV2, attention). Chi tiết ở [docs/03_plan.md](docs/03_plan.md).

Phân công, danh sách đầu việc, thứ tự cắt việc và checklist nộp bài: [docs/03_plan.md](docs/03_plan.md). Nhật ký: [docs/10_work_log.md](docs/10_work_log.md).

Yêu cầu kỹ thuật đầy đủ và nháp mẫu đăng ký: [SPEC_ViT.md](SPEC_ViT.md).
