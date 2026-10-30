# Bàn giao Hoàng: robustness cho hệ thống tìm kiếm ảnh

Ngày: 30/10/2026 • Phiên bản: 1.0 • Trạng thái: đặc tả triển khai, chưa phải báo cáo kết quả.

## 1. Mục tiêu và phạm vi đã chốt

Mục tiêu: đo và cải thiện khả năng tìm đúng ảnh cùng đối tượng khi query bị suy giảm chất lượng, trên nền VGG16 + GeM + L2, SfM subset-100.

**Bắt buộc thực hiện đầy đủ:**

1. Đối chiếu baseline, AUG, Attention và huấn luyện nhánh AUG + Attention để có ablation M0–M3 hợp lệ.
2. Đánh giá clean và robustness bằng một giao thức chung.
3. Triển khai cả bốn hướng search engine: Query Expansion, Geometric Re-ranking, Rank Fusion và Graph Diffusion.
4. Chạy ablation riêng từng hướng và kiểm tra tương tác với encoder.

Không coi bốn hướng retrieval là tùy chọn. Tuy nhiên, “làm hết” không có nghĩa chạy mọi tổ hợp có thể của mọi tham số. Các file sau xác định ma trận bắt buộc, ngân sách tìm tham số và cấu hình kết hợp cụ thể.

Bộ tài liệu này không thực hiện training, không chứng nhận kết quả cải tiến và không gửi nội dung trực tiếp cho Hoàng. Đây là tài liệu để người dùng bàn giao.

## 2. Thứ tự đọc

| File | Mục đích |
|---|---|
| [01_BASELINE_VA_QUY_UOC_THI_NGHIEM.md](01_BASELINE_VA_QUY_UOC_THI_NGHIEM.md) | Khóa baseline và quy ước so sánh |
| [02_TRAIN_AUG_ATTENTION_VA_ABLATION.md](02_TRAIN_AUG_ATTENTION_VA_ABLATION.md) | Huấn luyện/đối chiếu bốn encoder |
| [03_DANH_GIA_ROBUSTNESS_VA_KET_QUA.md](03_DANH_GIA_ROBUSTNESS_VA_KET_QUA.md) | Tạo benchmark, chọn cấu hình, báo cáo |
| [04_SEARCH_ENGINE_BON_HUONG_VA_ABLATION.md](04_SEARCH_ENGINE_BON_HUONG_VA_ABLATION.md) | Triển khai đủ bốn phương pháp retrieval |

## 3. Quan hệ giữa training và search

```text
TRAIN: M0 baseline / M1 AUG / M2 Attention / M3 AUG+Attention
                           ↓
                  checkpoint từng encoder
                           ↓
             descriptor query và gallery tương ứng
                           ↓
SEARCH: cosine / QE / geometric re-ranking / fusion / diffusion
                           ↓
                 xếp hạng → chấm → báo cáo
```

M3 là một mạng có attention được huấn luyện với AUG. Fusion(M1,M2) là hai mạng tìm riêng rồi gộp thứ hạng; hai cấu hình này không tương đương.

## 4. Thông tin đã biết và giới hạn

| Hạng mục | Trạng thái |
|---|---|
| train-b0-sfm100-progress.ipynb | Đã đọc: cấu hình chạy mới, không có output lưu |
| train-b0.ipynb | Đã đọc: resume từ epoch 7, hoàn thành tới epoch 14; log một phần epoch 15 |
| iir.zip | Đã kiểm tra trong phiên phân tích trước: best.pt epoch 12, latest.pt epoch 14, latest_resume.pt epoch 7 |
| Nhóm đã chạy AUG và Attention | Do người dùng cung cấp; chưa có mã, cấu hình, checkpoint của hai nhánh để kiểm tra |
| AUG + Attention | Công việc phải bổ sung; chưa có kết quả |
| Bốn hướng retrieval | Phạm vi bắt buộc mới; chưa có triển khai/kết quả được kiểm tra |

Các thông tin trên mô tả snapshot đã đọc, không khẳng định file trên máy Hoàng còn ở cùng phiên bản hoặc nhóm chưa chạy thêm từ đó.

Nguồn ban đầu trên máy người dùng: `C:\Users\DELL\Downloads\train-b0-sfm100-progress.ipynb`, `train-b0.ipynb`, `iir.zip`, và `1711.02512v2 (1).pdf`. Đây không phải đường dẫn bắt buộc trên máy Hoàng; cần ghi lại đường dẫn thực khi nhận bàn giao.

## 5. Việc cần tiếp nhận trước training chính thức

- [ ] Mã AUG thực tế: phép biến đổi, xác suất, biên độ, thứ tự; áp dụng q/p/n và mining thế nào.
- [ ] Mã Attention thực tế: loại module, vị trí, forward, khởi tạo, optimizer group.
- [ ] Checkpoint và config của M0/M1/M2; hash, epoch, seed, số optimizer steps, điểm xuất phát.
- [ ] Metadata subset và danh sách cluster/CID; ảnh SfM, Oxford/Paris, ground truth.
- [ ] Trọng số Caffe gốc; môi trường và GPU; lịch sử training của các nhánh.
- [ ] Xác nhận nhánh nào so sánh được dưới cùng ngân sách, nhánh nào chỉ là kết quả tham khảo.

Không tự thay AUG/Attention hiện có bằng một phương pháp khác rồi giữ nguyên tên M1/M2. Nếu thiếu thông tin, ghi `PENDING` và hoàn thành phần độc lập như evaluator, cache và cosine baseline trước. Thời gian chờ không được xem là xác nhận cấu hình.

## 6. Lộ trình và điều kiện hoàn thành

| Mốc | Công việc | Bằng chứng hoàn thành |
|---|---|---|
| P0 | Khóa dữ liệu và cấu hình | manifest + hashes + bảng tương thích M0–M2 |
| P1 | Hoàn thiện encoder | M0–M3 có checkpoint và run record hợp lệ |
| P2 | Benchmark robustness | manifest ảnh query, kiểm tra nhãn, evaluator sanity checks |
| P3 | Bốn hướng search | Mọi hàng ablation Q/G/F/D trong file 04 đã chạy |
| P4 | Tương tác và pipeline kết hợp | Cross-encoder + pipeline cuối; latency/storage |
| P5 | Đóng gói | Bảng tổng hợp, per-query metrics, phân tích failure, README tái chạy |

Ba nhánh cũ chỉ được tái sử dụng trong bảng chính nếu đáp ứng file 01–02. Cần chạy đối chứng bổ sung khi ngân sách/điểm xuất phát khác nhau. Không bỏ một hướng retrieval vì không tăng mAP; kết quả âm vẫn phải được lưu và giải thích.

## 7. Cấu trúc đầu ra triển khai đề nghị

```text
experiment/
  manifests/       # ảnh, split, ground truth, corruption, hash
  configs/         # cấu hình đã khóa trước test
  runs/M0...M3/    # checkpoint, history, environment, logs
  descriptors/     # tách theo checkpoint hash và preprocessing
  local_features/  # keypoints/RootSIFT cache
  graphs/          # gallery graph theo encoder
  rankings/        # đầy đủ thứ tự image ID
  metrics/         # aggregate và per-query
  reports/         # bảng, biểu đồ, failure cases, kết luận
```

## 8. Nguyên tắc diễn giải

Clean mAP cao hơn chưa đủ chứng minh robustness. Luôn báo điểm trên ảnh gốc, ảnh biến đổi, mức suy giảm và chi phí. Các kỹ thuật QE, spatial verification, RRF và diffusion đã có trong nghiên cứu trước; đóng góp của nhóm là thiết kế, tích hợp và đánh giá có kiểm soát trong bài toán này, không tuyên bố phát minh các kỹ thuật đó.
