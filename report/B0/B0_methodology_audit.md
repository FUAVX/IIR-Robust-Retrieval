# Audit phương pháp B0

**Phụ trách:** Nguyên (Methodology & Model Analysis)  
**Trạng thái:** mới audit source; chưa xác minh Kaggle runtime, dataset, training hoặc kết quả mAP.

Tài liệu này hỗ trợ review, không thay thế việc đọc paper của Radenović và cộng sự. Một **quan sát từ source** chỉ mô tả code đã import; tự nó không phải là quyết định của TEAM B0.

| Hạng mục | Quan sát từ source | Trạng thái TEAM B0 | Cần review / quyết định |
| --- | --- | --- | --- |
| Kiến trúc | `train.py` mặc định dùng `resnet101`; VGG16 được hỗ trợ. | Đã thống nhất VGG16. | Xác nhận custom Caffe VGG16 feature weights nạp được trên Kaggle; không thay bằng torchvision ImageNet weights. |
| Pooling và descriptor | `GeM(p=3)` có một scalar trainable dùng chung; retrieval net L2-normalize output. VGG16 cho vector 512-D. | Đã thống nhất GeM một `p` chung và 512-D L2-normalized. | Ghi lại giá trị khởi tạo `p` thực tế và bằng chứng load weights thành công. |
| Loss | CLI mặc định contrastive loss, margin `0.7`. | Đã chọn contrastive; margin chưa chốt. | Chọn và ghi rõ margin trước mọi B0 run. |
| Tuple | Mặc định: một query, một positive, năm negatives. | Đã thống nhất. | Log C3 phải chứng minh mỗi tuple có bảy ảnh. |
| Mining | `create_epoch_tuples()` refresh tuple ở đầu mỗi epoch; pool mặc định 20.000 và loại cluster của query. | Đã chọn hard-negative mining; pool size chưa chốt. | Kiểm tra metadata hỗ trợ pool đã chọn và năm negative clusters khác nhau. |
| Optimizer | CLI mặc định Adam, LR `1e-6`, weight decay `1e-6`; GeM `p` có LR gấp 10 và không weight decay. | Chưa chốt. | Quyết định đầy đủ policy cho từng parameter group; không suy ra quyết định nhóm từ CLI default. |
| Scheduler | `ExponentialLR(gamma=exp(-0.01))`; `scheduler.step()` chạy trước training của mỗi epoch. | Chưa chốt. | Quyết định có giữ timing lịch sử này không; C2 phải xác minh LR mong đợi trước/sau resume. |
| Checkpoint | Lưu epoch, model metadata/state, best loss và optimizer state. Resume dựng lại scheduler từ epoch; không lưu scheduler/RNG/config. | Chưa xác minh. | C2 cần test run bị ngắt và policy tái lập rõ ràng. |
| Chọn model tốt nhất | Dùng validation loss nếu có validation; nếu không dùng training loss. | Chưa chốt. | Ghi rõ có bật validation không và tiêu chí chọn checkpoint. |
| Dataset evaluation | Training CLI mặc định dùng revisited datasets; standalone test mặc định `oxford5k,paris6k`. | Đã chọn legacy Oxford5k và Paris6k. | Không trộn ROxford/RParis vào báo cáo B0. |
| Evaluation | Query vectors dùng bounding box benchmark; ranking dot product descriptor theo thứ tự giảm dần; legacy mAP loại `junk` trước AP. | Đã chọn single scale, không whitening, không QE. | C4 cần bằng chứng dataset/bbox và toy tests tại `tests/test_evaluate.py`. |

## Checklist review C2: checkpoint và resume

- [ ] Log LR của mọi optimizer parameter group trước lần update đầu tiên.
- [ ] Chạy vài updates, lưu checkpoint, khởi động lại process, resume và log lại LR/epoch.
- [ ] Xác nhận không lặp/bỏ epoch và LR không decay thêm ngoài ý muốn.
- [ ] So sánh short run không gián đoạn với run resume tương đương theo seed policy đã chọn.
- [ ] Ghi checkpoint path/checksum, commit, command/configuration, environment và các giới hạn đã biết, bao gồm RNG state.

### Resume và tính liên tục của scheduler

Checkpoint `model_epoch*.pth.tar` của source hiện tại được lưu sau khi một epoch
kết thúc. Nó có `epoch`, model state, `min_loss` và optimizer state; **không**
lưu `scheduler.state_dict()`, configuration hoặc RNG state. Khi resume, source
dựng lại `ExponentialLR(gamma=exp(-0.01))` từ `checkpoint['epoch'] - 1`, sau đó
gọi `scheduler.step()` ở đầu epoch tiếp theo. Ý đồ là tiếp tục learning-rate
(LR) decay thay vì quay lại LR ban đầu.

Điều này chưa đủ để khẳng định resume đúng trên PyTorch hiện đại: cách xử lý
`last_epoch` và thời điểm `scheduler.step()` có thể gây lệch một bước LR. C2
phải so sánh hai run cùng seed/configuration:

```text
Run A: epoch 1 → epoch 2 → epoch 3 liên tục
Run B: epoch 1 → epoch 2 → save → process mới → resume → epoch 3
```

Tại epoch 3, mọi optimizer parameter group phải có cùng LR ở hai run; epoch
number phải liên tục, optimizer state hợp lệ và loss không bất thường. Nếu LR
khác nhau thì không được chấp nhận resume cho B0.

Thiết kế checkpoint robust cho C2 nên lưu `next_epoch`, model state, optimizer
state, `scheduler_state_dict`, configuration và (nếu cần tái lập chặt) RNG/sampling
state. Khi resume, load scheduler state trực tiếp, không dựng lại scheduler bằng
suy đoán từ epoch. Đây là hướng cải tiến cần nhóm duyệt, không phải thay đổi
methodology tự động.

Nếu Kaggle ngắt giữa epoch, checkpoint hiện tại chỉ cho phép quay lại checkpoint
ở cuối epoch trước; phần batch đã chạy trong epoch bị ngắt sẽ mất. Đây vẫn là
resume theo epoch chấp nhận được nếu được ghi rõ, nhưng không phải resume chính
xác đến từng batch.

## Checklist review C3: training stability

- [ ] Kiểm tra data metadata và paths trước khi mining bắt đầu.
- [ ] Kiểm tra mỗi tuple có dạng `q + p + n1 + n2 + n3 + n4 + n5`.
- [ ] Xác nhận negatives không cùng query cluster và mining trả về đủ năm negatives.
- [ ] Log tensor shapes/devices, loss hữu hạn, gradients hữu hạn và bằng chứng parameter được update.
- [ ] Ghi lại GPU memory/OOM, không âm thầm đổi methodology.

## Checklist review C4: evaluation

- [ ] Xác nhận chính xác file legacy Oxford5k/Paris6k, query IDs, bounding boxes, annotation `ok` và `junk`.
- [ ] Xác nhận thứ tự gallery/query descriptors khớp IDs trong ground truth.
- [ ] Xác nhận query extraction nhận bounding boxes.
- [ ] Chạy `python -m unittest discover -s tests -p 'test_*.py'` từ repository root.
- [ ] Ghi riêng Oxford5k và Paris6k: checkpoint, scale, whitening/QE state, commit, command, protocol và mAP.

## Bàn giao cho Phương và Hoàng

**Gửi Hoàng:** manifest của ba datasets, Kaggle paths đã mount, tên metadata files, số lượng/kích thước file, và bằng chứng load thử ảnh/annotation. Nêu rõ protocol có phải legacy Oxford/Paris hay không.

**Gửi Phương:** Kaggle preflight log, custom VGG16-weight load log, B0 command/configuration dự kiến và C1--C3 logs. Không bắt đầu full B0 training khi các TEAM B0 decisions còn mở trong audit này chưa được ghi lại.

**Nguyên thực hiện sau bàn giao:** review logs theo checklist trên, sau đó chấp thuận hoặc từ chối sẵn sàng full B0 run cùng lý do có ghi nhận.
