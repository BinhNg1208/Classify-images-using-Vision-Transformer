# Kế hoạch và phân công

File này giữ phân công, danh sách đầu việc, mốc và quy tắc cắt việc; ít khi thay đổi. **Trạng thái từng đầu việc theo dõi trên GitHub**, không ghi ở đây.

## Phân công

| Thành viên | Vai trò | Tài liệu giữ | Phần báo cáo (mục 15 spec) |
|---|---|---|---|
| TV1 | Trưởng nhóm, dữ liệu, tài liệu | [04_data.md](04_data.md), [01_define.md](01_define.md), [02_architecture.md](02_architecture.md), file này | 1, 2, 8, 9 |
| TV2 | ViT và attention | [05_model.md](05_model.md) phần ViT, phần attention trong [07_evaluation.md](07_evaluation.md) | 3 (ViT), 6 |
| TV3 | CNN, đo FLOPs/MACs | [05_model.md](05_model.md) phần CNN | 3 (CNN), 6 (Grad-CAM nếu có) |
| TV4 | Huấn luyện | [06_training.md](06_training.md) | 4 |
| TV5 | Đánh giá, demo, ghép báo cáo | [07_evaluation.md](07_evaluation.md), [08_demo.md](08_demo.md) | 5, 7, 10 |

Tên thật sẽ thay cho TV1-TV5 khi nhóm chốt. TV4 nên là thành viên có máy GPU, vì phụ trách train và quản lý checkpoint.

## Theo dõi trạng thái trên GitHub
- Mỗi đầu việc là một **Issue**, tạo từ mẫu `.github/ISSUE_TEMPLATE/task.md`. Phần mô tả đầu việc (bước Describe) viết trong Issue.
- Trạng thái theo dõi trên bảng **GitHub Projects** với các cột: Chưa bắt đầu → Đang làm → Đã triển khai, chưa xác minh → Hoàn thành. Dùng nhãn `bi-chan` kèm comment nêu nguyên nhân nếu thiếu điều kiện.
- Mỗi PR ghi `Closes #<số issue>`. PR theo mẫu `.github/pull_request_template.md`, có phần Review, Test và bằng chứng.
- Chỉ chuyển sang Hoàn thành khi PR đã có bằng chứng kiểm tra và người duyệt đã xem. AI báo xong chưa đủ.
- Lỗi phát sinh tạo Issue từ mẫu `.github/ISSUE_TEMPLATE/bug.md`.

**Hiện trạng:** Issue và bảng Projects chưa tạo. Cho đến khi tạo: TASK-000 hoàn thành (nhóm duyệt `docs/` ngày 2026-11-15); các đầu việc khác chưa bắt đầu.

## Danh sách đầu việc

| Mã | Đầu việc | Phụ trách | Yêu cầu | Phụ thuộc |
|---|---|---|---|---|
| TASK-000 | Hoàn thiện `docs/` | TV1 | Quy trình `workflow/` | Không |
| TASK-014 | Nộp mẫu đăng ký (mục 21 spec) | TV1 | Đề | Nhóm chốt TV |
| TASK-001 | Khung repo và tiện ích dùng chung | TV1 | AC-013, một phần AC-008 | TASK-000 |
| TASK-002 | Chuẩn bị dữ liệu, chia tập, xuất mảng cache | TV1 | FR-001 | TASK-001, có file dữ liệu |
| TASK-003 | DataLoader đọc mảng cache, augmentation | TV1 | FR-001 | TASK-002 |
| TASK-004 | ViT và test | TV2 | FR-002 | TASK-001 |
| TASK-005 | CNN và `match_param_count` | TV3 | FR-003 | TASK-004 |
| TASK-006 | Vòng lặp train, `--quick`, `--resume` | TV4 | FR-004 | TASK-001, 003, 004, 005 |
| TASK-007 | Đánh giá và phân tích lỗi | TV5 | FR-005 | TASK-006 |
| TASK-008 | Trực quan hóa attention | TV2 | FR-006 | TASK-006 |
| TASK-009 | Bảng so sánh | TV5 | FR-007 | TASK-007 |
| TASK-010 | Demo | TV5 | FR-008 | TASK-006 |
| TASK-011 | Train thật ViT và CNN | TV4 | FR-004 | TASK-006, máy GPU của thành viên sẵn sàng |
| TASK-012 | Đánh giá test cuối và báo cáo | TV5, cả nhóm | FR-005, 007, 009 | TASK-011 |
| TASK-013 | Phần làm thêm theo thứ tự cắt việc dưới đây | Theo phần | Không bắt buộc | TASK-011 |

**Điều kiện bắt đầu code:** Chỉ bắt đầu TASK-001 và các đầu việc code khác sau khi nhóm duyệt `docs/`. Bản thử đầu của TASK-001 viết sai thứ tự nên đã bị xóa.

**Đầu việc tiếp theo:** TASK-014. Mục tiêu: giảng viên nhận được mẫu đăng ký đủ 10 mục. Việc cần làm: điền quyền sử dụng dữ liệu, sửa nháp ở mục 21 spec nếu cần, rồi nộp đúng nơi giảng viên quy định. Phụ thuộc: nhóm chốt ai là TV nào.

## Mốc và lịch

| Mốc | Kết quả trình diễn được | Đầu việc | Hạn |
|---|---|---|---|
| M0 | `docs/` được nhóm duyệt, đã nộp mẫu đăng ký | 000, 014 | 8/11 |
| M1 | Khung repo và hợp đồng sẵn sàng | 001 | 8/11 |
| M2 | `--quick` chạy thông từ dữ liệu đến checkpoint | 002-006 | 10/11 |
| M3 | Có checkpoint thật, đủ công cụ đánh giá và demo | 007-011 | 14/11 |
| M4 | Có số cuối trên test, dừng mọi thí nghiệm | 012, 013 | 16/11 |
| M5 | Báo cáo khớp số liệu, nộp bài | 012 | 19/11 |

20/11 để dự phòng. Quy tắc cắt việc khi trễ ở mục 20 của spec.

## Thứ tự cắt việc khi trễ
CNN thứ hai có FLOPs xấp xỉ ViT, rồi ablation, rồi 3 seed, rồi Grad-CAM, cuối cùng mới đến đo FLOPs/MACs (TV3 phụ trách). Thí nghiệm SGD được cắt cùng lúc với CNN thứ hai; attention theo head và layer được cắt cùng lúc với Grad-CAM. Giữ nguyên bộ lõi: ViT, CNN, attention, demo, báo cáo. Không giảm dưới 50 epoch (DEC-008).

## Nộp bài (Definition of Done)

Với đồ án này, "phát hành" là bản nộp cho giảng viên. Chưa có bản nộp.

**Trước khi nộp:**
- [ ] Tiêu chí trong phạm vi đã có bằng chứng kiểm tra.
- [ ] Hạn chế và phần chưa xác minh được nêu rõ.
- [ ] README khớp cách chạy thật.
- [ ] Môi trường và dữ liệu cần thiết đã được kiểm tra trên máy sạch.
- [ ] Mọi số trong báo cáo khớp với file kết quả.
- [ ] Link checkpoint mở được khi không đăng nhập tài khoản của nhóm.
- [ ] Repo không chứa `kaggle.json`, dữ liệu hoặc checkpoint.
- [ ] Có commit hoặc tag đánh dấu bản nộp để quay lại khi cần.
- [ ] Đã xác nhận nơi và hình thức nộp bài.

**Bản nộp:**
- **Phiên bản / commit:** Chưa có
- **Nơi truy cập:** GitHub `QuocAn118/Classify-images-using-Vision-Transformer`; nơi nộp chính thức chưa xác nhận
- **Sản phẩm bàn giao:** Mã nguồn, checkpoint, bảng so sánh, hình attention, phân tích lỗi, demo, báo cáo
- **Cách quay lại bản trước:** Checkout commit hoặc tag đã đánh dấu

**Sau khi nộp, thử lại:**

| Luồng cần thử | Kết quả thực tế | Bằng chứng |
|---|---|---|
| Clone mới, cài đặt, chạy `pytest` | Chưa chạy | Chưa có |
| Tải checkpoint, chạy `app/demo.py`, thử một ảnh | Chưa chạy | Chưa có |
| Chạy `src.compare`, đối chiếu với báo cáo | Chưa chạy | Chưa có |

Phản hồi của giảng viên ghi vào [10_work_log.md](10_work_log.md); đầu việc phát sinh tạo Issue mới.
