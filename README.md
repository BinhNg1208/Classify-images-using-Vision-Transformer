# Phân loại ảnh Tiny ImageNet bằng Vision Transformer

Đồ án môn Học sâu, đề số 9. Nhóm tự cài ViT, so sánh với một CNN có cùng số tham số và trực quan hóa attention.

> **Trạng thái:** nhóm đã duyệt `docs/` (2026-11-15), chuẩn bị viết code. Repo chưa có code, dữ liệu hay kết quả.

## Bài toán

Phân loại ảnh 64x64 vào 200 lớp. Dự án trả lời hai câu hỏi: với cùng số tham số, ViT so với CNN thế nào trên tập dữ liệu nhỏ, và ViT nhìn vào vùng nào của ảnh khi dự đoán.

## Tính năng dự kiến

- ViT tự cài bằng PyTorch (patch embedding + Transformer encoder).
- ResNet nhỏ có số tham số chênh không quá 5% so với ViT.
- Bảng so sánh Top-1, Top-5, số tham số, FLOPs/MACs, thời gian train và tốc độ inference.
- Hình CLS attention và attention rollout.
- Demo Gradio: tải ảnh lên, xem Top-5 và ảnh attention.

## Kiến trúc

Xem sơ đồ và thành phần tại [docs/02_architecture.md](docs/02_architecture.md); mỗi thành phần có một file riêng trong `docs/`.

## Cài đặt và chạy

Chưa có. Lệnh dự kiến nằm trong mục "Cách chạy" của từng file `docs/` và sẽ được đưa vào đây sau khi đã chạy thử thật.

## Tài liệu

| File | Nội dung |
|---|---|
| [Project_description.md](Project_description.md) | Tóm tắt Define, Design, Plan và mục lục `docs/` |
| [docs/](docs/) | Tài liệu theo thành phần: dữ liệu, mô hình, huấn luyện, đánh giá, demo; kèm define, kiến trúc, quyết định, kế hoạch, nhật ký |
| [SPEC_ViT.md](SPEC_ViT.md) | Yêu cầu kỹ thuật chi tiết, nháp mẫu đăng ký |
| [AGENTS.md](AGENTS.md) | Quy tắc cho người và AI khi làm việc trong repo |
| [AI_Usage.md](AI_Usage.md) | Ghi nhận việc dùng AI |
| [workflow/](workflow/) | Quy trình 14 bước giảng viên yêu cầu |
| [.github/](.github/) | Mẫu Issue và PR, nơi ghi các bước Describe, Review, Test |
