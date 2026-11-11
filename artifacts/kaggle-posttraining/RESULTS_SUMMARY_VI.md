# Tổng kết post-training hai encoder epoch 100

> Cập nhật: đã nhận baseline B0 best epoch12 (Oxford81,06%, Paris77,12%). Khuyến nghị Run1+QE dưới đây giới hạn trong hai encoder epoch100. Xem [bản tổng hợp có B0](../results-consolidated/TONG_HOP_TAT_CA_KET_QUA.md) để đối chiếu toàn bộ model; B0 chưa có robustness/post-training tương đương.

Đánh giá exploratory, một training seed (42), legacy Oxford5k/Paris6k. Cả hai model dùng final epoch 100, FP32 inference, TF32 tắt. Tham số chính khóa trước chạy; sensitivity chỉ mô tả, không dùng chọn lại cấu hình test.

Đã hoàn thành 68.200 ô chính + 5.500 ô sensitivity = 73.700 dòng per-query; 1.120 mẫu latency trực tiếp. Corrupted mAP trung bình đều 18 điều kiện; noise/occlusion đã average realization trước. Không coi các query-corruption là các training seed độc lập.

## Bảng chính Run 1 — AUG

| Phương pháp | Oxford clean | Oxford corrupted | Paris clean | Paris corrupted |
|---|---:|---:|---:|---:|
| Cosine | 74.00% | 70.80% | 64.29% | 62.84% |
| AQE | 76.41% | 73.81% | 65.90% | 64.84% |
| Alpha-QE | 76.45% | 73.86% | 65.89% | 64.83% |
| Gated alpha-QE | 76.45% | 73.86% | 65.89% | 64.83% |
| Local re-ranking | 78.53% | 75.08% | 65.42% | 64.01% |
| Geometric re-ranking | 76.68% | 73.46% | 64.83% | 63.40% |
| Geometry-verified QE | 78.94% | 76.28% | 66.97% | 65.82% |
| Diffusion mutual | 48.95% | 47.32% | 53.11% | 52.26% |
| Diffusion union | 23.67% | 22.10% | 45.61% | 45.15% |

## Hai encoder ở epoch 100

| Encoder | Oxford clean | Oxford corrupted | Paris clean | Paris corrupted |
|---|---:|---:|---:|---:|
| Run 1: AUG | 74.00% | 70.80% | 64.29% | 62.84% |
| Run 2: AUG + Attention | 71.37% | 68.15% | 62.72% | 61.59% |

Run 1 thắng Run 2 trên clean và corrupted trung bình ở cả hai dataset, tại checkpoint/precision đang đánh giá. Không đủ dữ liệu để kết luận attention nói chung không hiệu quả.

## Diễn giải và lựa chọn

- Mặc định thực dụng: Run 1 + alpha-QE (k=10, alpha=3, lambda=0,5). Trung bình đều hai dataset: clean 69,14 → 71,17%; corrupted 66,82 → 69,34% (+2,52 điểm phần trăm). AQE gần như tương đương: 71,15% / 69,32%; chưa có bằng chứng thống kê alpha-QE hơn AQE.
- Điểm cao nhất trong cấu hình chính: Run 1 + verified QE, macro clean 72,96%, corrupted 71,05%; tăng corrupted +4,23 điểm phần trăm so cosine. So với alpha-QE, thêm khoảng +1,71 điểm corrupted nhưng direct latency median khoảng 8,48 giây Oxford / 5,82 giây Paris.
- Alpha-QE retrieval-only median khoảng 6,55 ms Oxford / 9,88 ms Paris, so cosine 3,13 / 4,79 ms. Direct end-to-end khoảng 75,55 / 87,34 ms trong mẫu thử. Không diễn giải QE nhanh hơn cosine từ các median end-to-end: timing nhiễu, số mẫu nhỏ và thứ tự đo cố định.
- Gated QE tau=0,5 mở 100%, vẫn lấy đủ 10 feedback; output giống alpha-QE. Gate hiện tại chưa tạo hiệu quả từ chối query drift.
- Local-only re-ranking tốt hơn geometric re-ranking ở cả hai dataset. Kiểm chứng hình học hữu ích hơn khi lọc feedback cho verified QE trong thí nghiệm này; không suy ra mọi phương pháp RANSAC/re-ranking đều kém.
- Fusion score/RRF không vượt cosine Run 1, đồng thời cần hai encoder. Không chọn làm mặc định từ kết quả hiện có.
- Diffusion giảm mạnh, cả union lẫn mutual. Hầu hết đã hội tụ, nên không thể quy đơn giản cho không đủ iterations. Cần audit seed/normalization/graph ranking và giả thuyết graph lan truyền sai trước khi thử tuning thêm; chưa chứng minh nguyên nhân. Không dùng cấu hình diffusion này để triển khai.
- Blur mức nặng là điều kiện khó nhất cho Run 1 cosine ở cả hai dataset. Worst-condition Oxford 48,33 → 60,97% với verified QE; Paris 54,71 → 58,60%. Không dùng drop nhỏ một mình để gọi model robust.

## Lưu ý checkpoint và training

Run 1 đạt best validation epoch 27 (0,100764), lên 0,123108 ở epoch 100. Run 2 best epoch 28 (0,100519), lên 0,119673 ở epoch 100. Đây là dấu hiệu overfit theo validation loss, không phải bằng chứng cứ tăng epoch sẽ tốt hơn.

Điểm best checkpoint Colab đã lưu: Run 1 Oxford 79,53%, Paris 71,93%; Run 2 Oxford 80,49%, Paris 72,76%. Những điểm đó cao hơn final-epoch cosine trong đợt này, nhưng khác checkpoint và precision inference, nên chưa là đối chứng chỉ thay epoch dưới cùng điều kiện. Không trộn bảng best cũ với bảng final + post-training mới.

Bước tiếp theo hợp lý: lấy lại best epoch 27 của Run 1 (không có trong hai ZIP hiện tại); best epoch 28 Run 2 có sẵn. Trích lại cả hai best dưới cùng FP32 protocol, rồi so cosine và QE. Không cần training thêm để thực hiện phép so này. Nếu chỉ dùng artifact hiện tại thì giữ lựa chọn Run 1 final + alpha-QE và ghi đúng checkpoint.

## Tệp nguồn

- outputs/iir-post100-report/unpacked/summary.csv: chất lượng và retrieval-only timing.
- outputs/iir-post100-report/unpacked/end_to_end_summary.csv: trực tiếp đo decode → encoder → search/post-processing; 28 mẫu mỗi model/dataset/method, median/p95 chỉ có tính tham khảo.
- outputs/iir-post100-report/unpacked/per_query.csv, conditions.csv, paired_deltas.csv: kiểm tra chi tiết.
- outputs/iir-post100-report/unpacked/diagnostics.csv: gate, fallback, convergence.
- outputs/iir-post100-report/unpacked/failure_examples.png và robustness_*.png: ví dụ và đồ thị.

Tất cả kết luận giới hạn ở hai encoder, một seed, dataset/protocol và các tham số đã chạy. Không tuyên bố tối ưu tổng quát hoặc gain đã được xác nhận qua nhiều training seeds.
