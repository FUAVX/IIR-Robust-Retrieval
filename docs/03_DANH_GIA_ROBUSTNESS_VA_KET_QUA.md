# Giao thức đánh giá robustness và kết quả

Ngày: 30/10/2026 • Bắt buộc • Áp dụng cho [M0–M3](02_TRAIN_AUG_ATTENTION_VA_ABLATION.md) và [bốn hướng search](04_SEARCH_ENGINE_BON_HUONG_VA_ABLATION.md).

## 1. Tách dữ liệu và vai trò

| Tập | Dùng để |
|---|---|
| SfM train | Cập nhật trọng số |
| SfM validation tuples | Chọn checkpoint bằng validation loss theo baseline |
| Development retrieval set | Chọn encoder/search hyperparameters và kiểm tra corruption |
| Legacy Oxford/Paris test | Báo cáo cuối với config đã khóa |

Development retrieval phải tách khỏi Oxford/Paris test và khỏi train. Không được giả định cùng SfM cluster đồng nghĩa mọi ảnh đều là positive: cần nhãn match/non-match có kiểm chứng hoặc một retrieval development set có ground truth đầy đủ. Positive pair list 181 cặp không tự tạo một benchmark retrieval đầy đủ nếu coi mọi cặp còn lại là negative.

Nếu chỉ có metadata subset hiện tại, cần chuẩn bị development set hợp lệ (dùng thông tin hình học/annotation được kiểm tra) trước tuning. Các thử nghiệm trước đã dùng Oxford/Paris để xem kết quả phải ghi là exploratory; không tuyên bố chúng là test hoàn toàn chưa đụng tới. Không chia ngẫu nhiên các phiên bản cùng query/landmark sang cả development và test.

## 2. Benchmark chính

Giữ legacy Oxford5k (5.063 gallery, 55 queries) và Paris6k (6.392 gallery, 55 queries), đúng imlist/qimlist/gnd của snapshot. Kiểm tra ảnh và hash file thực tế, không chỉ tin tên dataset.

- Query crop theo bbox và resize theo loader baseline.
- Tạo corruption sau crop/resize ở miền RGB [0,1], trước ToTensor/Normalize; áp dụng giống nhau cho global và local-feature evaluation.
- Gallery giữ clean cho bài kiểm tra chính.
- Query clean là đúng kết quả preprocessing nền, không thay đổi thêm.
- Test sử dụng scale đơn, whitening/QE tắt trừ hàng thí nghiệm explicitly bật thành phần tương ứng.

Đây là benchmark robustness tự xây trên query legacy, không đặt tên khiến người đọc tưởng là một benchmark corruption chính thức.

## 3. Corruption protocol v1 đề xuất

Các giá trị dưới đây là thiết kế khởi đầu cụ thể, chưa phải thông số AUG của nhóm. Kiểm tra tính hợp lệ trên development rồi khóa trước test; nếu cần thay đổi, tăng version.

| Loại | Nhẹ | Vừa | Nặng | Định nghĩa |
|---|---:|---:|---:|---|
| Darkening | 0,8 | 0,6 | 0,4 | nhân pixel RGB với hệ số |
| Low contrast | 0,8 | 0,6 | 0,4 | x'=clip(mu+a(x−mu)); mu là mean từng kênh của ảnh |
| Gaussian blur | 0,001L | 0,002L | 0,004L | sigma pixel, L=max(H,W); kernel 2ceil(3sigma)+1 |
| Gaussian noise | 0,02 | 0,05 | 0,08 | std trên miền [0,1], iid từng pixel/kênh, clip |
| JPEG | 80 | 50 | 25 | quality; cố định encoder/version, RGB, subsampling=2 |
| Occlusion | 5% | 10% | 20% | diện tích hình chữ nhật, fill mean RGB của query |

Với blur, sigma tối thiểu 0,1 px; dùng cùng border mode và thư viện/version. JPEG decode lại trước model. Occlusion: lấy aspect ratio log-uniform trong [0,5;2], chọn vị trí hợp lệ bằng seed; kiểm tra vẫn còn đủ chi tiết đối tượng. Không gọi vùng còn lại chắc chắn đúng nhãn nếu mất toàn bộ landmark.

Noise/occlusion dùng 3 realization cố định mỗi query/mức; lấy trung bình AP các realization trước khi tính metric điều kiện. Biến đổi deterministic dùng một realization, không nhân bản giả để tăng sample size.

Quy trình hậu kiểm corruption được thực hiện không nhìn điểm model: nếu một query trở nên vô nghĩa, dùng cùng quy tắc loại/giảm mức cho tất cả model, ghi số lượng và lý do. Ưu tiên khóa quy tắc trên development; không chọn ảnh test dễ cho một model.

Sau khi nhận AUG config, gắn `seen`/`unseen` theo từng họ biến đổi. Nếu tất cả sáu họ đều đã có trong train thì không được tuyên bố kiểm tra unseen; bổ sung một họ giữ lại chỉ cho evaluation trong protocol mới trước test.

## 4. Seed và manifest

Seed corruption phải độc lập training seed, ví dụ 20261030, 20261031, 20261032. Dùng stable hash của (query_id, corruption, severity, realization) để lấy seed con; không dùng Python hash mặc định vốn có thể thay đổi theo tiến trình.

Manifest ghi: dataset, query ID, bbox, clean dimensions, corruption, tham số, realization, seed, checksum ảnh đầu ra, version thư viện. Cùng manifest dùng cho tất cả M/Q/G/F/D và các training seed. Không tạo lại ảnh ngẫu nhiên bên trong từng vòng evaluate model.

## 5. Metric

### Chất lượng

- Legacy mAP: dùng evaluator gốc, đúng AP trapezoidal và xử lý junk; không thay bằng sklearn average_precision_score rồi gọi cùng metric.
- AP từng query, từng điều kiện và seed.
- P@10: precision trong 10 kết quả không-junk đầu; nếu ít hơn 10 kết quả hợp lệ, ghi quy tắc mẫu số thực tế.
- Relevant Recall@K: số relevant trong top-K không-junk / tổng relevant; K=20,50,100. Không nhầm với Hit@K (có ít nhất một ảnh đúng).
- Mean corrupted mAP: trung bình đều 18 điều kiện = 6 họ × 3 mức, sau averaging realization; báo riêng Oxford và Paris, sau đó macro-average hai dataset.
- Worst-condition mAP: min trên 18 điều kiện, không phải min AP một query.
- Drop: clean mAP − corrupted mAP, đơn vị điểm phần trăm; có thể âm.
- QE/fusion/diffusion: thêm tỷ lệ query tăng/giảm AP so với đối chứng paired; tolerance số học 1e-6.

Không dùng Drop một mình: model yếu có thể giảm ít nhưng vẫn có mAP thấp.

### Chi phí hệ thống

Báo median và p95 latency/query, số tham số, descriptor bytes/image, tổng index/local-feature/graph storage và thời gian offline preparation.

Latency end-to-end gồm query decoding/preprocessing, trích descriptor, search, local matching/QE/diffusion/fusion theo pipeline. Đồng bộ GPU trước/sau đo, warm-up cố định, batch query=1, cùng hardware/worker. Báo thêm retrieval-only latency để tái sử dụng cache không làm sai so sánh. Gallery cache/graph build báo offline; query không được coi là đã cache trong số end-to-end.

## 6. Validation tuning và test

Chọn tham số theo mean corrupted mAP development, tie-break clean mAP rồi latency. Không chọn một config khác cho từng test corruption khi hệ thống thực tế không biết corruption label.

Tuning encoder E* và retrieval configs trên seed tham chiếu 42; khóa configs và áp dụng cho các seed 43/44. Khi encoder đổi, cache và graph phải được xây lại, dù hyperparameters giữ nguyên.

## 7. Thống kê

Báo mean ± std qua 3 training seeds cho các metric chính; từng seed không phải từng query. Dùng paired bootstrap theo landmark/query group để xem chênh lệch AP của hai phương pháp; resample nguyên nhóm, giữ mọi corruption của cùng query đi cùng nhau. Đề xuất 2.000 bootstrap replicates, seed cố định; nêu đây là khoảng tin cậy có điều kiện trên các run đã có, không thay thế độ biến thiên training.

Không tính hàng chục bản corruption cùng một query là hàng chục mẫu độc lập. Với ít landmark, diễn giải độ chắc chắn thận trọng; không chỉ báo p-value rồi kết luận tổng quát.

## 8. Sanity checks bắt buộc

- [ ] Cosine trên vector L2 và Euclidean cho thứ hạng tương đương ngoài tie.
- [ ] ID không trùng/mất; query/gallery vector đúng checkpoint và thứ tự.
- [ ] AP tiny fixture kiểm tra junk trước positive, perfect ranking, poor ranking.
- [ ] QE lambda=0 trả lại cosine; gate đóng trả lại query gốc.
- [ ] Re-ranking chỉ đổi top-K; phần đuôi giữ thứ tự, không mất/trùng ảnh.
- [ ] Fusion một danh sách trả cùng thứ hạng; tie-break image ID ổn định.
- [ ] Diffusion beta=0 trả seed score y theo đúng định nghĩa; graph không có self-loop.
- [ ] Không dùng ground truth để lọc ảnh QE, tạo cạnh graph hoặc tính score.

## 9. Schema kết quả

`per_query.csv`: protocol, model_id, checkpoint_hash, train_seed, method_id, config_hash, dataset, query_id, landmark_group, corruption, severity, realization, AP, P10, recall20/50/100, latency_ms, qe_gate_open, geometric_inliers, error_status.

`summary.csv`: model, method, seed, dataset, clean_map, corrupt_map, worst_map, drop_pp, p10, median_ms, p95_ms, storage_bytes. Ghi metric dạng [0,1] trong máy; bảng báo cáo chuyển sang phần trăm và ghi rõ.

Lưu full ranking image IDs để chấm lại và kiểm toán. Không thay lỗi inference bằng AP=0 mà không ghi lỗi; báo tỷ lệ lỗi và cách xử lý công khai.

## 10. Báo cáo cần giao

1. Bảng M0–M3 cosine.
2. Bảng Q/G/F/D ablation.
3. Bảng cross-encoder và pipeline kết hợp.
4. Đồ thị mAP theo severity của từng họ; clean vs corrupted; chất lượng vs latency.
5. Ví dụ query tốt lên và xấu đi, dùng cùng quy tắc chọn: top cải thiện/suy giảm và mẫu ngẫu nhiên, không chỉ chọn đẹp.
6. Manifest config đã khóa và các giới hạn: checkpoint cũ, seed, development labels, phạm vi corruption.

ROxford/RParis hoặc gallery distractors là đánh giá mở rộng ngoài ma trận chính của bộ bàn giao; bốn hướng retrieval vẫn bắt buộc dù không thêm dataset mới. Không trộn điểm giao thức legacy và revisited.
