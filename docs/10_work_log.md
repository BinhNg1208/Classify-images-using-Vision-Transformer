# Nhật ký làm việc

Ghi ngắn sau mỗi buổi hoặc mốc đáng kể.

## 2026-11-07

- **Đầu việc:** TASK-000 (tài liệu).
- **Kết quả / thay đổi chính:**
  - Phân tích đề và sửa `SPEC_ViT.md` theo đề gốc, thêm phân công, lịch, nháp mẫu đăng ký.
  - Điền `docs/`, viết `README.md`, `AGENTS.md`, `AI_Usage.md`.
  - Chốt train trên máy GPU của một thành viên, Kaggle hoặc Colab là dự phòng.
  - AI đã viết thử TASK-001 trước khi `docs/` hoàn thành, sai thứ tự quy trình. Nhóm quyết định xóa và làm lại sau khi `docs/` được duyệt.
- **Bằng chứng kiểm tra:** Tài liệu chưa được nhóm đọc. Repo chưa có code nên chưa có kết quả chạy.
- **Điều tôi giải thích được về code:** Chưa có code.
- **Điều chưa hiểu / chưa xác minh:** Các giả định trong `../01-define/DEFINE.md`.
- **Bước tiếp theo:** Nhóm đọc và duyệt `docs/`, chốt TV, nộp mẫu đăng ký, rồi mới bắt đầu TASK-001.

## 2026-11-07 (chỉnh sửa theo quyết định nhóm)

- **Đầu việc:** TASK-000 (tài liệu).
- **Kết quả / thay đổi chính:**
  - Quyết định: số chưa đo ghi là "dự đoán"; train 50 epoch và đây là mức sàn; batch 64, lr ViT 2,5e-4 cho cả ViT và CNN; bỏ ViT pretrained; áp dụng nhận xét của senior.
  - File đã sửa: `SPEC_ViT.md`, `docs/01_DEFINE.md`, `docs/02_REQUIREMENTS.md`, `docs/03_DESIGN.md`, `docs/04_BREAKDOWN.md`, `docs/05_TOOLS.md`, `docs/06_VERIFICATION.md`, `docs/07_WORK_LOG.md`, `AGENTS.md`, `AI_Usage.md`.
  - Quyết định thêm: giai đoạn tài liệu làm thẳng trên `main`, từ TASK-001 mỗi người làm nhánh riêng rồi mở PR, TV1 duyệt; thứ tự cắt việc là CNN thứ hai, ablation, 3 seed, Grad-CAM, cuối cùng mới đến đo FLOPs/MACs.
  - Áp dụng 11 điểm rà soát tài liệu. Hợp đồng dùng chung đổi: JSON thêm `macs`, `peak_vram_mb`; checkpoint thêm `scaler`, `class_names`; config thêm `mean`, `std`; tên lần chạy gồm seed; `--quick` dùng dữ liệu giả khi chưa có cache. Cần báo cả nhóm.
- **Bằng chứng kiểm tra:** Chỉ rà soát tài liệu bằng `grep`; chưa có code nên chưa có kết quả chạy.
- **Điều tôi giải thích được về code:** Chưa có code.
- **Điều chưa hiểu / chưa xác minh:** FLOPs/MACs, thời gian train, VRAM, mức Top-1 dự đoán; nội dung tài liệu chưa được nhóm duyệt.
- **Bước tiếp theo:** Nhóm đọc và duyệt `docs/`, chốt TV, nộp mẫu đăng ký, rồi mới bắt đầu TASK-001.

## 2026-11-14

- **Đầu việc:** TASK-000 (tài liệu).
- **Kết quả / thay đổi chính:**
  - Nhóm xóa `.agents/docs/` (bộ mẫu cũ) và đổi tên `.agents/` thành `workflow/`.
  - Thử chia `docs/` theo 14 bước của `workflow/` nhưng nhiều file rỗng; nhóm chọn chia theo thành phần (DEC-013).
  - `docs/` mới: `01_define.md`, `02_architecture.md`, `09_decisions.md`, `03_plan.md`, `04_data.md`, `05_model.md`, `06_training.md`, `07_evaluation.md`, `08_demo.md`, `10_work_log.md`. Nội dung cũ được chuyển sang, không thêm yêu cầu mới. FR và AC chuyển về file thành phần; hợp đồng ghi ở file của bên tạo ra nó.
  - `Project_description.md` thành bản tóm tắt Define, Design, Plan.
  - Trạng thái đầu việc chuyển lên GitHub Issues và Projects (chưa tạo). Thêm mẫu PR, mẫu Issue "Đầu việc" và "Lỗi" trong `.github/`.
  - Sửa `AGENTS.md`, `README.md` theo cấu trúc mới.
- **Bằng chứng kiểm tra:** Script kiểm tra link tương đối và đối chiếu mã FR, AC, DEC, TASK giữa bản cũ và bản mới.
- **Điều tôi giải thích được về code:** Chưa có code.
- **Điều chưa hiểu / chưa xác minh:** Nội dung tài liệu chưa được nhóm duyệt; giảng viên chưa xác nhận cách nộp bằng chứng trên Issue/PR.
- **Bước tiếp theo:** Tạo Issue và bảng Projects; hỏi giảng viên về cách nộp bằng chứng; nhóm duyệt `docs/`, chốt TV, nộp mẫu đăng ký.

## 2026-11-15

- **Đầu việc:** TASK-000 (tài liệu).
- **Kết quả / thay đổi chính:** Đánh số file `docs/` theo thứ tự đọc. Nhóm duyệt `docs/`; TASK-000 hoàn thành. Từ đây được bắt đầu code.
- **Bằng chứng kiểm tra:** Nhóm xác nhận duyệt (người dùng báo lại). Script kiểm tra link: 0 link hỏng.
- **Điều tôi giải thích được về code:** Chưa có code.
- **Điều chưa hiểu / chưa xác minh:** Giảng viên chưa xác nhận cách nộp bằng chứng trên Issue/PR. Các mốc M0 đến M3 trong `03_plan.md` đã quá hạn, cần lập lại lịch.
- **Cập nhật:** Giảng viên gia hạn nộp đến 25/11. Lập lại lịch trong `03_plan.md` và mục 20 spec: mục tiêu nộp 24/11, dừng thí nghiệm 21/11, 25/11 dự phòng.
- **Cập nhật:** Người dùng xác nhận mình là TV1 và đã nộp mẫu đăng ký; TASK-014 hoàn thành.
- **Cập nhật:** Đổi tên `AGENT_SPEC_ViT_TinyImageNet.md` thành `SPEC_ViT.md`, sửa tiêu đề cho tự nhiên hơn. TV1 làm mọi đầu việc của mình thẳng trên `main` (DEC-014); không tạo Issue cho đầu việc của TV1.
- **Bước tiếp theo:** Chốt TV2-TV5, TV4 tải thử dữ liệu, tạo Issue, bắt đầu TASK-001.

## Mẫu buổi làm
- Ngày: [YYYY-MM-DD]
- Đầu việc: [TASK-xxx — tên]
- Kết quả / thay đổi chính: [cần điền]
- Bằng chứng kiểm tra: [lệnh → kết quả thật]
- Điều tôi giải thích được về code: [cần điền]
- Điều chưa hiểu / chưa xác minh: [cần điền]
- Bước tiếp theo: [cần điền]

## Câu hỏi cuối đầu việc
Tôi dựa vào bằng chứng nào để biết phần này đúng?

## Khi tiếp tục với AI
Đọc `AGENTS.md`, đầu việc đang làm và bản ghi mới nhất.
Xác minh trạng thái code hiện tại trước khi tiếp tục.
Không thực hiện lại việc đã hoàn thành chỉ vì phiên trò chuyện mới.
