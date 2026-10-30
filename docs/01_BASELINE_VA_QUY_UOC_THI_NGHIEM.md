# Baseline và quy ước thí nghiệm

Ngày: 30/10/2026 • Xem [README](00_README_BAN_GIAO.md). Các giá trị mục 1–2 được xác nhận từ snapshot đã đọc; quy tắc mục 3 trở đi là thiết kế bàn giao cần thực hiện.

## 1. Baseline B0 đang có

```text
RGB → resize giữ tỉ lệ → ToTensor → Normalize
→ VGG16 features bỏ max-pool cuối → scalar GeM → L2 → 512-D
```

Mean RGB = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]. `thumbnail` giới hạn cạnh dài, không ép vuông và không phóng ảnh nhỏ lên đủ 1024. Trọng số feature khởi tạo: `imagenet-caffe-vgg16-features-d369c8e.pth`.

| Tham số | Snapshot baseline |
|---|---|
| Source commit | 747166e627e514842f5c5be531f4e0f73612cc90 |
| Train | 15.953 ảnh, 100 cluster, 34.337 positive pairs |
| Validation | 739 ảnh, 20 cluster, 181 positive pairs |
| Subset seed | 42 |
| Tuple | query + positive + 5 negatives từ 5 cluster khác nhau |
| Tuples train/epoch | 2.000 mục cặp, không bảo đảm 2.000 query duy nhất |
| Mining pool | yêu cầu 20.000, thực tế 15.953 |
| Validation processed | 180 tuple do batch 5, drop_last=True |
| Resolution train/eval | cạnh dài tối đa 1024 |
| Optimizer | Adam; feature LR 1e-6, GeM LR 1e-5 |
| Weight decay | feature 1e-6; GeM 0 |
| Loss | contrastive, margin 0,7 |
| GeM | một p dùng chung, khởi tạo 3 |
| Batch / update_every | 5 tuple / 1; 400 optimizer steps/epoch |
| Scheduler | ExponentialLR, gamma=exp(-0.01), gọi trước train mỗi epoch |
| Target | tổng 100 epoch; không phải số epoch đã hoàn tất |
| Workers / precision | 8 / FP32 |
| GPU | model parallel hai T4, feature split_at=17 |
| Selection | validation loss thấp nhất |
| Evaluation | legacy Oxford5k/Paris6k, một scale, không whitening/QE |

Không nhầm `momentum=0.9` trong config chung với tham số đang dùng của Adam. Mining thực hiện theo mô hình mỗi epoch. Loss được cộng theo tuple trong backward, không tự đổi sang lấy trung bình gradient vì sẽ thay đổi độ lớn cập nhật.

## 2. Artifact đã đọc và điểm khác paper

Snapshot `iir.zip`: best epoch 12; latest epoch 14; file latest_resume epoch 7. Run trước chưa chứng minh hoàn tất 100 epoch. Không dùng latest_resume epoch 7 thay cho latest epoch 14 nếu mục tiêu là tiếp tục trạng thái mới nhất đã bàn giao; kiểm tra file thực tế trước khi chọn.

Paper v2 mục 5.1 mô tả cấu hình khác: train tối đa 362 px, nhiều nhất 30 epoch, margin VGG 0,75, weight decay 5e-4, lịch exp(-0.1i), chọn mạng theo validation mAP. Paper còn có learned whitening, multi-scale và alpha-QE. Thí nghiệm subset 100 cluster của paper chọn các cluster lớn nhất, không phải phép chọn ngẫu nhiên seed42 của notebook.

Vì vậy báo cáo gọi đây là **baseline triển khai dựa trên paper, với cấu hình subset-100 đã ghi rõ**, không tuyên bố tái lập nguyên xi toàn bộ thiết lập hoặc bảng kết quả paper.

## 3. Các yếu tố phải khóa giữa M0–M3

- Cùng subset, train/validation cluster, positive pair list và pretrained feature weights.
- Cùng precision, kích thước ảnh, preprocessing nền, batch, loss, optimizer nhóm chung, scheduler và tổng số optimizer steps.
- Cùng seed danh sách và quy tắc lấy mẫu query/pool. Negative IDs có thể khác do descriptor khác; không ép chúng giống nhau rồi thay đổi ý nghĩa mining.
- Cùng lịch evaluation, cách chọn checkpoint và ngân sách tuning.
- AUG là khác biệt có chủ đích M1/M3; Attention là khác biệt có chủ đích M2/M3.
- Attention parameters dùng cùng quy tắc khởi tạo/LR ở M2 và M3; ghi rõ vì baseline không có nhóm này.
- Dùng cùng evaluator, image IDs, ground truth và ảnh corruption cho mọi nhánh.

Đóng băng các quyết định trong `protocol_v1.json` trước test; lưu hash. Nếu cần sửa, tăng protocol version, không trộn kết quả cũ/mới trong cùng bảng.

## 4. Training budget và seed

Mục tiêu tham chiếu là 100 epoch = 40.000 steps. Nếu tài nguyên buộc dùng E epoch thấp hơn, chọn E trước khi chạy nhánh mới và dùng cùng ngân sách cho cả bốn nhánh trong bảng chính. Số steps, không chỉ thời gian GPU, là tiêu chuẩn khớp ngân sách; thời gian GPU được báo riêng.

Kế hoạch đầy đủ: seed 42, 43, 44 cho mỗi nhánh. Với seed 42, snapshot cũ chỉ được tái sử dụng nếu đủ thông tin kiểm chứng. Subset seed luôn 42; training seed thay đổi, không tạo lại subset theo seed train.

Nếu chỉ đủ một seed, vẫn phải làm đủ các hướng nhưng gắn nhãn kết quả một seed; không báo độ ổn định training như đã được xác nhận. Chạy nhiều seed là phần cần hoàn thành để có báo cáo đầy đủ theo kế hoạch này.

## 5. Resume và khởi tạo nhánh mới

**Resume cùng thí nghiệm:** nạp model, optimizer, scheduler, epoch, RNG và history; kiến trúc/cấu hình/dataset identity phải khớp. Giữ cơ chế strict của notebook cho resume hợp lệ.

**Khởi tạo thí nghiệm mới:** nạp có chủ đích trọng số tương thích, tạo optimizer/scheduler/history mới và ghi nguồn khởi tạo. Không gọi thao tác này là resume.

Ma trận chính M0–M3 ưu tiên cùng Caffe feature initialization. Nếu nhóm đã bắt đầu cải tiến từ baseline checkpoint, cần một ma trận tiếp tục học riêng: cả bốn nhánh khởi tạo từ cùng checkpoint, M0 là đối chứng được học thêm cùng số steps. Không so M1 được học thêm với M0 dừng trước giai đoạn bổ sung rồi quy toàn bộ lợi ích cho AUG.

Thêm attention làm khác state_dict. Chỉ cho phép thiếu đúng các attention keys khi khởi tạo nhánh mới, assert danh sách thiếu/thừa; không dùng `strict=False` để bỏ qua mọi sai khác. Không nới lỏng assert identity toàn cục của cơ chế resume.

## 6. Chọn checkpoint và validation

Giữ tiêu chí checkpoint của baseline là validation contrastive loss trên ảnh không AUG, cùng cách mining và drop_last. Ghi đây là tín hiệu có mining thay đổi theo model. Không chọn epoch bằng Oxford/Paris test mAP.

Nếu các checkpoint cũ đã được chọn theo test score, không thể làm sạch sai lệch chỉ bằng đổi tên. Ghi rõ đây là kết quả exploratory; chuẩn bị evaluation holdout chưa dùng để chọn nếu cần kết luận xác nhận.

Tuning retrieval dùng development retrieval set tách biệt, theo file 03; không dùng legacy test để chọn k, gate, graph hoặc encoder.

## 7. Manifest bắt buộc

Mỗi run lưu: experiment ID, protocol version/hash, source hash, dataset hashes, seed, initialization path/hash/epoch, model config, AUG config, attention config, optimizer groups, scheduler timing, steps/epochs, checkpoint rule, environment, hardware, timestamps, train/validation metrics và lỗi.

Descriptor cache phải định danh bằng checkpoint SHA-256 + preprocessing + dataset/image manifest. Khi đổi encoder phải trích lại gallery, không dùng gallery vector của M0 cho query M3. Kiểm tra cùng chiều, hữu hạn, norm gần 1, số vector và thứ tự image ID chính xác.

## 8. Nguồn

- Notebook và artifact đã phân tích trong tài liệu đối chiếu `phan-tich-hai-notebook-b0.md` (tài liệu tham khảo chưa được đưa vào repository).
- [Fine-tuning CNN Image Retrieval with No Human Annotation, bản v2](https://arxiv.org/abs/1711.02512v2).
