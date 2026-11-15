# Quy tắc làm việc trong repo

Áp dụng cho cả thành viên nhóm và AI coding agent.

## Đọc trước khi làm

1. Issue của đầu việc trên GitHub, và [docs/03_plan.md](docs/03_plan.md) để biết phân công, phụ thuộc.
2. [docs/10_work_log.md](docs/10_work_log.md): bản ghi mới nhất.
3. File `docs/` của thành phần đang làm (danh sách trong [Project_description.md](Project_description.md)) và [docs/02_architecture.md](docs/02_architecture.md).
4. Hướng dẫn của bước đang làm trong [workflow/](workflow/).

## Quy trình

Đi theo thứ tự trong `workflow/`: Define, Design, Plan, rồi với từng đầu việc là Describe, Generate, Review, Run, Test, Debug và Refactor khi cần, Commit, Explain, Documentation, Definition of Done.

| Bước | Ghi ở đâu |
|---|---|
| Define, Design, Plan | `Project_description.md` (tóm tắt) và `docs/` |
| Describe | Issue tạo từ mẫu "Đầu việc" |
| Generate, Review, Run, Test, Explain | PR theo `.github/pull_request_template.md` |
| Debug | Issue tạo từ mẫu "Lỗi" |
| Commit | Commit và PR được duyệt |
| Documentation | File `docs/` của thành phần, `README.md`, `AI_Usage.md`, `docs/10_work_log.md` |
| Definition of Done | Mục "Nộp bài" trong `docs/03_plan.md` |

Nhóm đã duyệt `docs/` ngày 2026-11-15, được bắt đầu viết code theo `docs/03_plan.md`.

Với đầu việc nhỏ, một dòng bằng chứng trong PR là đủ. Ưu tiên thời gian cho train và báo cáo.

## Quy tắc bắt buộc

- Mỗi lần chỉ làm một đầu việc trong `docs/03_plan.md`, không tự mở rộng phạm vi.
- Trước khi làm, mô tả đầu việc trong Issue.
- Muốn đổi hợp đồng dùng chung (ghi trong file `docs/` của bên tạo ra nó) thì sửa tài liệu và báo cả nhóm trước.
- Không dùng tập test để chọn siêu tham số hay checkpoint.
- Không bịa số liệu. Số trong báo cáo phải lấy từ file kết quả; chưa có thì ghi "chưa có".
- Không commit dữ liệu, checkpoint, `kaggle.json` hay bất kỳ bí mật nào.
- AI báo xong chưa đủ để đánh dấu Hoàn thành: phải có bằng chứng trong PR và người duyệt đã xem.

## Git

- Giai đoạn tài liệu: làm thẳng trên `main`.
- Từ TASK-001 trở đi, khi 5 người code song song: mỗi người làm trên nhánh riêng rồi mở PR gộp vào `main`, TV1 là người duyệt; PR của chính TV1 do TV4 duyệt.
- Không dùng `git push --force` lên nhánh chung.
- Commit message mô tả kết quả cụ thể, không thêm dòng `Co-Authored-By`.
- AI chỉ commit hoặc push khi được yêu cầu.

## Sau mỗi buổi

- Cập nhật trạng thái Issue trên GitHub Projects.
- Thêm bản ghi vào `docs/10_work_log.md`.
- Ghi công cụ AI đã dùng vào `AI_Usage.md`.
