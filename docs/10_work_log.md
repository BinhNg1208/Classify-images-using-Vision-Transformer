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

## 2026-11-15 (TASK-001)

- **Đầu việc:** TASK-001, khung repo và tiện ích dùng chung (TV1, thẳng trên `main`).
- **Kết quả / thay đổi chính:**
  - `src/utils.py`: `load_config` (báo tất cả khóa thiếu trong một lần, AC-013), `make_run_name`, `set_seed`, `count_params`, `atomic_save` (ghi file tạm rồi `os.replace`, lỗi thì xóa file tạm và giữ file cũ, DEC-005).
  - `configs/vit_small.yaml`, `configs/cnn_baseline.yaml` theo spec; phần `data`, `train`, `seed` giống hệt nhau và có test giữ điều này.
  - `.gitignore`, `requirements.txt`, `pytest.ini`, `tests/test_utils.py`.
- **Bằng chứng kiểm tra:**
  - Viết test trước: `python -m pytest -q` → 12 failed (hàm chưa viết). Sau khi viết code: 12 passed.
  - Cố ý làm hỏng code: bỏ bước xóa file tạm → test `keeps_old_file` fail; bỏ khóa `data.mean` khỏi danh sách bắt buộc → test `missing_key` fail. Khôi phục: 12 passed.
  - `git check-ignore` từng đường dẫn: chặn `data/`, checkpoint, `*.pt`, `kaggle.json`, `__pycache__`; không chặn `outputs/logs`, `outputs/results`, `src/`, `configs/`, `workflow/`.
  - Môi trường: Windows 10, Python 3.13.9, torch 2.9.1+cpu, pyyaml 6.0.3, pytest 8.4.2.
- **Điều tôi giải thích được về code:** TV1 đã đọc `src/utils.py` ngày 2026-11-16 và đồng ý đưa lên `main`.
- **Điều chưa hiểu / chưa xác minh:** `atomic_save` chưa thử khi mất điện thật; `os.replace` chỉ nguyên tử khi file tạm và file đích cùng ổ đĩa (đang đảm bảo vì tạo cùng thư mục). Chưa cài thử `requirements.txt` trên máy sạch và máy GPU.
- **Bước tiếp theo:** TV1 làm TASK-002; TV2 bắt đầu TASK-004 trên nhánh riêng.

## 2026-11-16 (TASK-002)

- **Đầu việc:** TASK-002, chuẩn bị dữ liệu, chia tập, xuất mảng cache (TV1, thẳng trên `main`).
- **Kết quả / thay đổi chính:**
  - `scripts/prepare_data.py`: `--download` tải zip từ Stanford (trang Kaggle cần lời mời, DEC-015); đọc nhãn val từ `val_annotations.txt`; kiểm tra số lượng; chia train gốc 90/10 phân tầng seed 42; xuất cache `.npy` (ghi file tạm rồi đổi tên); `class_names.json`, `dataset_stats.json`, `samples.png`.
  - Thêm khóa bắt buộc `data.cache_dir` vào cả hai config và `REQUIRED_KEYS`; thêm `pillow`, `matplotlib` vào `requirements.txt`.
- **Bằng chứng kiểm tra:**
  - Viết test trước: 6 test mới fail (hàm chưa viết), sau khi viết code toàn bộ 18 passed.
  - Cố ý làm hỏng code: bỏ sắp xếp mã lớp → 2 test fail; chia không theo lớp → 2 test fail. Khôi phục: 18 passed.
  - Chạy thật trên máy TV1: "200 lớp, 100000 ảnh train gốc, 10000 ảnh val gốc"; `{"train": 90000, "val": 10000, "test": 10000}`, cache 1.35 GB; đếm nhãn trong cache: mỗi lớp 450 train / 50 val / 50 test. Thời gian 64 phút.
  - Ảnh mẫu lần đầu chỉ có một lớp (lấy 25 ảnh đầu) và đường dẫn trong stats dùng `\`; đã sửa (lấy ngẫu nhiên, đường dẫn dạng `/`, có test), tạo lại hai file này từ cache.
- **Điều tôi giải thích được về code:** TV1 đã đọc `scripts/prepare_data.py` ngày 2026-11-17 và đồng ý đưa lên `main`.
- **Điều chưa hiểu / chưa xác minh:** Chưa chạy lại toàn bộ trên dữ liệu thật sau lần sửa cuối (chỉ tạo lại ảnh mẫu và stats); chưa thử `--download` trên Linux/máy GPU; chưa có test cho trường hợp ghi `.npy` bị lỗi giữa chừng. Vì sao chạy mất 64 phút chưa đo (nghi do đọc 110.000 file nhỏ trên Windows).
- **Bước tiếp theo:** TV1 làm TASK-003; TV2 làm TASK-004 trên nhánh riêng.

## 2026-11-17 (TASK-003)

- **Đầu việc:** TASK-003, DataLoader đọc mảng cache và augmentation (TV1, thẳng trên `main`).
- **Kết quả / thay đổi chính:**
  - `src/data.py`: `build_loader(cfg, split)` và `CachedDataset`; mở cache bằng mmap lười, không pickle cả mảng sang worker (Windows tạo worker bằng spawn); train: RandomCrop padding 4, lật ngang, RandAugment, RandomErasing theo config; val/test: chỉ chuẩn hóa.
  - Hợp đồng ghi trong `docs/04_data.md`. Sửa các chỗ còn ghi Kaggle trong `SPEC_ViT.md`; thêm hướng dẫn chạy vào `README.md`.
- **Bằng chứng kiểm tra:**
  - Viết test trước: 7 test mới fail (chưa có code); sau khi viết code: 25 passed (coi cảnh báo là lỗi).
  - Cố ý làm hỏng code: bỏ `__getstate__` → test pickle fail; val bỏ Normalize → test val fail; train không bỏ batch lẻ → test option fail. Khôi phục: 25 passed.
  - Cache thật, `configs/vit_small.yaml`, 4 worker, CPU: batch `(64, 3, 64, 64)` float32; 1406 batch train/epoch, 157 batch val; train khoảng 825 ảnh/giây, val khoảng 3.223 ảnh/giây (100 batch, một lần đo).
- **Điều tôi giải thích được về code:** TV1 đã đọc `src/data.py` ngày 2026-11-17 và đồng ý đưa lên `main`.
- **Điều chưa hiểu / chưa xác minh:** Chưa đo trên máy GPU; chưa biết GPU có phải chờ DataLoader không.
- **Bước tiếp theo:** TV2 làm TASK-004, TV3 làm TASK-005, TV4 làm TASK-006 trên nhánh riêng.

## 2026-11-17 (hợp đồng tạo mô hình)

- **Đầu việc:** chốt hợp đồng dùng chung `build_model(cfg)` trong `docs/05_model.md` trước khi TV2, TV3, TV4 bắt đầu.
- **Kết quả / thay đổi chính:** thêm `build_model(cfg)` (chọn theo `model.name`) vào hợp đồng mô hình; TV1 chốt làm lần lượt TASK-004 → 005 → 006, không cắt việc, giữ các mốc hiện tại.
- **Bằng chứng kiểm tra:** chỉ sửa tài liệu, chưa có code.
- **Điều chưa hiểu / chưa xác minh:** TV2, TV3, TV4 chưa xác nhận đã đọc hợp đồng.
- **Bước tiếp theo:** TV1 nhắn TV2 bắt đầu TASK-004 trên nhánh riêng.

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
