# Ghi nhận việc dùng AI

Chỉ ghi công cụ và đóng góp đã biết. Phần chưa rõ ghi "chưa xác nhận". Cập nhật khi có thêm công việc.

| Tools used | Main purposes | AI hỗ trợ | Đóng góp sinh viên/người thực hiện |
|---|---|---|---|
| Claude Code (Claude Opus 5.5), plugin ponytail và superpowers | Phân tích đề, soạn tài liệu | Đối chiếu spec với đề gốc; sửa `SPEC_ViT.md`; soạn `docs/`, `README.md`, `AGENTS.md`, `Project_description.md`, nháp mẫu đăng ký | Cung cấp đề, mẫu đăng ký, hạn nộp, quy trình `.agents/`; yêu cầu và quyết định hướng làm. Việc duyệt nội dung: chưa xác nhận |
| Claude Code (Claude Opus 5.5) | Viết thử code | Viết thử khung repo và tiện ích dùng chung, đã xóa theo quyết định của nhóm vì sai thứ tự quy trình; không còn trong repo | Phát hiện sai thứ tự và quyết định xóa |
| Claude Code (Claude Opus 5.5), plugin ponytail và superpowers | Chỉnh sửa tài liệu theo quyết định nhóm | Sửa spec, `docs/`, `AGENTS.md`, `AI_Usage.md` theo các quyết định được giao | Người dùng cung cấp quyết định. Việc duyệt nội dung: chưa xác nhận |
| Claude Code (Claude Opus 5.5), plugin ponytail và superpowers | Sắp xếp lại `docs/` | Chia `docs/` theo thành phần, tóm tắt `Project_description.md`, viết mẫu Issue và PR, sửa `AGENTS.md` | Người dùng chọn hướng chia theo thành phần và theo dõi trạng thái trên GitHub. Việc duyệt nội dung: chưa xác nhận |
| Claude Code (Claude Opus 5.5), plugin ponytail và superpowers (skill TDD) | TASK-001: khung repo | Viết test trước, rồi `src/utils.py`, hai file config, `.gitignore`, `requirements.txt`, `pytest.ini`; thử làm hỏng code để kiểm tra test | TV1 giao việc, quyết định phạm vi và làm thẳng trên `main`. TV1 đã đọc `src/utils.py` ngày 2026-11-16 |
| Claude Code (Claude Opus 5.5), plugin ponytail và superpowers (skill TDD) | TASK-002: chuẩn bị dữ liệu | Viết test trước, rồi `scripts/prepare_data.py`; tìm nguồn tải Stanford thay Kaggle; chạy trên dữ liệu thật và kiểm tra số lượng | TV1 giao việc, quyết định không chia sẻ dữ liệu (DEC-015). TV1 đã đọc `scripts/prepare_data.py` ngày 2026-11-17 |
| Claude Code (Claude Opus 5.5), plugin ponytail và superpowers (skill TDD) | TASK-003: DataLoader | Viết test trước, rồi `src/data.py`; chốt hợp đồng `build_loader`; đo tốc độ trên cache thật | TV1 giao việc; TV1 đã đọc `src/data.py` ngày 2026-11-17 |
| Claude (trong cuộc trò chuyện giữa người dùng và Claude), model cụ thể chưa xác nhận | Soạn bản đầu của `SPEC_ViT.md` | Soạn bản đầu của spec | Người dùng trao đổi yêu cầu và đưa file vào repo; đóng góp chi tiết chưa xác nhận |
| Codex (GPT-6) | Đọc và nắm hiện trạng dự án (2026-11-17) | Đọc tài liệu, config, mã nguồn và test; đối chiếu tiến độ; chạy `python -m pytest -q -W error` (25 passed); ghi nhật ký buổi đọc. Không triển khai đầu việc code mới | Người dùng yêu cầu đọc dự án và cung cấp quy tắc repo; chưa có xác nhận review mới |
| Codex (GPT-6) | TASK-004: ViT và kiểm thử (2026-11-17) | Pull main, tạo nhánh `tv2/task-004-vit`, tạo Issue #1 theo mẫu; viết ViT PyTorch thuần, `build_model(cfg)` và 23 test; chạy toàn bộ 48 test, forward/backward mặc định và lưu bằng chứng JSON; cập nhật tài liệu, chuẩn bị nội dung PR. Sau đó tạo nhánh `thanhbinh` để commit theo yêu cầu mới | Người dùng xác nhận vai trò TV2 và giao TASK-004, yêu cầu đọc gốc trước docs và làm nhánh riêng; sau đó yêu cầu mọi thay đổi được commit trên `thanhbinh`. TV1 chưa review; chưa yêu cầu push |

## Ghi chú

- Phân công cho từng thành viên chưa chốt, nên chưa ghi đóng góp theo tên.
- Đã có tiện ích, chuẩn bị dữ liệu và DataLoader (TASK-001 đến TASK-003); ViT (TASK-004) đã triển khai trên nhánh riêng, chờ review. Chưa có số liệu huấn luyện/đánh giá; số tham số và kiểm tra mô hình có bằng chứng trong `outputs/results/task004_vit_checks.json`.
