# Quyết định thiết kế

Mỗi quyết định một dòng. Muốn đổi quyết định đã chốt thì thêm dòng mới ghi rõ thay cho mã nào, không xóa dòng cũ.

| Mã | Quyết định | Vì sao | Lựa chọn khác và đánh đổi | Trạng thái |
|---|---|---|---|---|
| DEC-001 | Tiny ImageNet là dữ liệu chính, CIFAR-100 dự phòng | Có sẵn link trong đề, ảnh lớn hơn | CIFAR-100 nhẹ hơn nhưng ảnh 32x32 | Đề xuất |
| DEC-002 | AdamW cho cả hai mô hình trong so sánh chính | So sánh công bằng | SGD có thể tốt hơn cho CNN, chỉ làm tham khảo | Đề xuất |
| DEC-003 | Batch 64, lr 2,5e-4 mặc định cho cả ViT và CNN, cùng batch | Dự đoán vừa 6GB VRAM, chưa đo; scale tuyến tính từ 1e-3 ở batch 256 | Batch 128 qua gradient accumulation 2 bước (giữ lr 5e-4, không tốn thêm VRAM, nhưng epoch không nhanh hơn và BatchNorm của CNN chỉ gần tương đương) | Đề xuất |
| DEC-004 | Thư mục gốc repo là gốc project | Tránh lồng thêm `vit-tiny-imagenet/` | Thư mục con như spec cũ, đường dẫn dài hơn | Đề xuất |
| DEC-005 | Ghi checkpoint và log qua file tạm | Phiên bị ngắt lúc ghi không hỏng file cũ | Ghi trực tiếp đơn giản hơn nhưng có thể mất checkpoint | Đề xuất |
| DEC-006 | Tự cài ViT thay vì dùng `timm` | Dễ chỉnh số tham số và lấy attention | `timm` nhanh hơn; đề không bắt buộc tự cài | Đề xuất |
| DEC-007 | CNN dùng stem 3x3 stride 1 nhưng tầng đầu tiên downsample sớm, hoặc stem stride 2 (tránh feature map 64x64 ở nhiều tầng, vì tốn VRAM và thời gian) | Giảm VRAM và thời gian, vẫn khớp tham số trong 5% qua `match_param_count` | Stem stride 1 không downsample sớm giữ chi tiết hơn nhưng nặng; chọn cách nào sẽ ghi lại ở đây | Đề xuất |
| DEC-008 | Train 50 epoch, đây là mức sàn | Vừa ngân sách một máy GPU | 100 epoch có thể tốt hơn cho ViT nhưng tốn gấp đôi thời gian; train chậm thì cắt phần làm thêm và chuyển Kaggle/Colab, không giảm epoch | Đã chốt |
| DEC-009 | Bỏ ViT pretrained | Nhóm quyết định tập trung vào so sánh chính | Giữ làm mốc tham chiếu nhưng tốn VRAM và thời gian | Đã chốt |
| DEC-010 | Giai đoạn tài liệu làm thẳng trên `main`; từ TASK-001 mỗi người làm nhánh riêng và mở PR, TV1 duyệt, PR của TV1 do TV4 duyệt | Tránh đụng độ khi code song song và vẫn có người review | Làm thẳng trên `main` nhanh hơn nhưng dễ ghi đè nhau | Thay bằng DEC-014 |
| DEC-011 | Chuẩn hóa bằng mean/std của ImageNet, khai báo trong config | Có sẵn, không phải tính lại, dùng được cả khi chuyển CIFAR-100 | Mean/std tính từ tập train sát dữ liệu hơn nhưng thêm một bước và phải lưu kèm | Đề xuất |
| DEC-012 | Augmentation: RandomCrop padding 4, lật ngang, RandAugment; dropout 0 | Nhẹ hơn RandomResizedCrop, hợp với 50 epoch; đã có drop path 0,1 | RandomResizedCrop hoặc TrivialAugment, dropout 0,1 có thể thử trong ablation | Đề xuất |
| DEC-013 | `docs/` chia theo thành phần; Describe, Review, Test, Commit của từng đầu việc ghi trên GitHub Issue và PR; trạng thái đầu việc theo dõi trên GitHub | Mỗi người giữ một file, ít xung đột khi PR song song; bằng chứng gắn với code; không có file rỗng | Chia `docs/` theo 14 bước của `workflow/`: khớp tên bước nhưng nhiều file rỗng và một thành phần bị rải nhiều nơi | Đã chốt, chờ giảng viên xác nhận cách nộp bằng chứng |
| DEC-014 | Thay DEC-010: TV1 làm mọi đầu việc của mình thẳng trên `main`; TV2-TV5 làm nhánh riêng và mở PR, TV1 duyệt | TV1 là trưởng nhóm, làm phần nền trước; bỏ thủ tục PR cho phần của TV1 để nhanh hơn | Phần của TV1 không có người review; bù bằng test tự động và TV1 báo nhóm khi đổi hợp đồng | Đã chốt |
