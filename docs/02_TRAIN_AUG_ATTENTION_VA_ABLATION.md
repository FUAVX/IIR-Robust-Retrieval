# Training AUG, Attention và ablation M0–M3

Ngày: 30/10/2026 • Bắt buộc • Phụ thuộc [quy ước baseline](01_BASELINE_VA_QUY_UOC_THI_NGHIEM.md).

## 1. Ma trận phải hoàn thành

| ID | AUG | Attention | Retrieval khi so encoder |
|---|---|---|---|
| M0 | Không | Không | Cosine exact |
| M1 | Có | Không | Cosine exact |
| M2 | Không | Có | Cosine exact |
| M3 | Có | Có | Cosine exact |

M1/M2 do nhóm báo đã chạy, nhưng chưa kiểm tra mã. M3 phải dùng đúng AUG của M1 và attention của M2 để so kết hợp. Không lặng lẽ đổi loại module hoặc policy AUG ở M3.

## 2. Thông tin chưa có phải điền từ mã nhóm

| Trường | Trạng thái hiện tại |
|---|---|
| Tên/code augmentation policy | PENDING |
| Xác suất, biên độ, thứ tự từng phép | PENDING |
| Áp dụng q, p, negatives; độc lập hay chung tham số | PENDING |
| Policy khi mining | PENDING |
| Policy khi validation | PENDING |
| Attention loại gì, ở lớp nào | PENDING |
| Attention tensor shape, initialization | PENDING |
| LR/weight decay attention; freeze/unfreeze | PENDING |
| Checkpoint, seed, steps và initialization M1/M2 | PENDING |

Các PENDING này là dữ liệu cần nhận, không phải tham số mặc định. Không chạy M3 production trước khi xác định chúng. Có thể triển khai ngay evaluator, manifest, cosine, QE, local feature cache, fusion API và graph builder trên M0 đã có.

## 3. Quy tắc AUG trong ma trận kiểm soát mới

Nếu mã nhóm đã tuân thủ các quy tắc sau, giữ nguyên. Nếu chưa, ghi khác biệt và thống nhất protocol mới cho cả M1 và M3; kết quả cũ được giữ làm tham khảo.

- AUG chỉ ở train forward; validation và benchmark clean không có augmentation ngẫu nhiên.
- Mining dùng ảnh clean đã preprocessing chuẩn để giữ phép chọn negative dễ đối chiếu. Nếu muốn augmented mining, đó là thí nghiệm khác có ID riêng, không trộn vào M3 mặc định.
- Policy AUG và thứ tự pipeline giống nhau ở M1/M3.
- Lấy mẫu biến đổi độc lập cho q/p/n để không tạo dấu hiệu chung nhân tạo giữa hai ảnh positive, trừ khi policy nhóm có lý do khác đã ghi rõ.
- Photometric transform trước Normalize; kiểm soát miền pixel và tránh áp dụng nhiễu trên tensor đã chuẩn hóa mà không điều chỉnh đơn vị.
- Crop/erasing phải giữ đủ đối tượng và vùng chung q–p. Không coi nhãn pair còn đúng chỉ vì file nguồn có cùng cluster.
- Lưu ảnh tuple trước/sau AUG cho kiểm tra bằng mắt; ghi xác suất thực tế của mỗi phép trong một mẫu đủ lớn.
- Seed riêng cho chọn tuple và AUG để số phép random của AUG không vô tình đổi toàn bộ lịch lấy mẫu query giữa các nhánh.

Mục tiêu AUG là descriptor ổn định trước biến đổi bảo toàn danh tính; không tự thêm consistency loss khi mục tiêu hiện tại là kiểm tra đúng policy nhóm đã làm.

## 4. Quy tắc Attention

Giữ output descriptor 512-D nếu có thể; nếu thay chiều phải báo thêm storage/compute và không quy lợi ích chỉ cho attention.

Nếu chèn trước GeM, feature đầu vào GeM cần tuân theo giả định số không âm của pipeline gốc. Kiểm tra việc attention/residual có tạo giá trị âm và có khiến clamp của GeM loại thông tin hay không. Không tự thêm ReLU mà không ghi đó là thay đổi kiến trúc.

Trên hai T4, đặt module đúng GPU theo tensor đầu vào/đầu ra. GeM và loss đang ở GPU 0. Kiểm tra module có nằm trong optimizer, có gradient hữu hạn và có cập nhật sau optimizer step. M2/M3 cùng initialization attention và optimizer groups.

Heatmap chỉ minh họa, không là bằng chứng độc lập rằng robustness đã tăng.

## 5. Các bước thực hiện

### A. Audit M0–M2

Tạo `compatibility_matrix.csv`: mỗi run gồm data hash, init hash, seed, steps, resolution, optimizer, scheduler, mining, AUG, attention và selection. Phân loại `comparable`, `needs_control_run`, hoặc `exploratory` cùng lý do.

### B. Forward và kiểm tra loss

- Một ảnh: shape descriptor (512,1), hữu hạn, norm gần 1.
- Một tuple: descriptor (512,7), labels đúng q/p/5n.
- Loss và gradient hữu hạn; cả backbone, GeM và attention (nếu có) được cập nhật.
- Tắt AUG/attention theo config phải thật sự đi đúng đường M0, không chỉ đổi tên run.

### C. Smoke

Thử riêng từng kiến trúc/policy khác nhau ở resolution mục tiêu: mining pool nhỏ đủ cluster, một batch train, một validation tuple, save/reload. Kiểm tra feature attention và optimizer state sau reload. Smoke checkpoint không được dùng làm initialization full run.

### D. Full run

Seed và budget theo file 01. Lưu latest/best atomic, history, memory hai GPU, elapsed time, số steps thực tế và config. Giữ tiêu chí best theo validation loss để khớp baseline. Không tự hạ resolution/đổi precision khi OOM rồi giữ cùng experiment ID.

### E. Extract và evaluate

Mỗi checkpoint best được chọn bằng validation tạo cache query/gallery riêng. Chạy cosine clean + corruption theo file 03. Đây là bảng model ablation; chưa QE/re-ranking/fusion/diffusion.

## 6. Các phép so bắt buộc

| Hiệu số | Ý nghĩa |
|---|---|
| M1 − M0 | Lợi ích AUG |
| M2 − M0 | Lợi ích attention |
| M3 − M1 | Thêm attention khi đã có AUG |
| M3 − M2 | Thêm AUG khi đã có attention |
| M3 − M0 | Lợi ích kết hợp |

Tính riêng cho clean mAP, mean corrupted mAP và từng corruption. Có thể báo tương tác `I = S(M3) − S(M1) − S(M2) + S(M0)`; I dương chỉ mô tả trên thang đo S, không tự chứng minh ý nghĩa thống kê.

## 7. Chọn encoder cho tuning search

Chọn E* bằng mean corrupted mAP trên development set sau khi checkpoint từng run đã được chọn bằng validation loss. Báo kèm clean mAP; không dùng test để chọn. Nếu hòa trong sai số định trước (ví dụ 0,1 điểm phần trăm), chọn clean mAP cao hơn rồi latency thấp hơn. Khóa lựa chọn trước khi chạy bảng test.

Vẫn giữ đủ M0–M3 trong kết quả; E* chỉ phục vụ sàng lọc tham số search. Fusion bắt buộc M1 và M2, không phụ thuộc E* là mô hình nào.

## 8. Đầu ra và nghiệm thu

- [ ] Bảng tương thích M0–M2 và quyết định có cần đối chứng bổ sung.
- [ ] AUG/attention config hết PENDING; hash code và protocol.
- [ ] M0–M3 có checkpoint, train history, selection và smoke record.
- [ ] Manifest descriptor của từng encoder.
- [ ] Clean/corrupted aggregate và per-query AP của bốn nhánh.
- [ ] Không có test-based tuning, không cache chéo encoder.
- [ ] Báo kết quả âm, chi phí và giới hạn nhiều seed nếu chưa đủ.

Tên run đề nghị: `M3_aug-attn_seed42_protocol-v1`, kèm initialization hash và budget trong config. Đây là quy ước tên, không phải lệnh CLI đã tồn tại.
