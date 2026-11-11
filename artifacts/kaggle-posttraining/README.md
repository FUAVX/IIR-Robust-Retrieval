# Post-training epoch 100 trên Kaggle

Hai encoder: `run1_aug_epoch100` và `run2_aug_attention_epoch100`. Không train, không dùng descriptor best epoch 27/28. Dùng FP32, tắt TF32, L2-normalize descriptor cho cosine. Kết quả là exploratory, một training seed 42. Không dùng relevance labels để chọn tham số, feedback hoặc cạnh graph.

## Inputs và provenance

`prepare.py` đọc đúng hai ZIP ở project root, lấy latest epoch 100, xác minh SHA sidecar khi có và xuất riêng state_dict. Manifest ghi SHA256 ZIP/checkpoint/weights, config và lịch sử. Xuất/load kiểm tra output FP32 giống hệt trên ảnh xác định. SHA code và protocol được kiểm tra trước khi notebook import runtime.

Run 1 không có sidecar latest trong ZIP: hash checkpoint được tính trực tiếp và lưu vào manifest, không tuyên bố đã đối chiếu với một sidecar không tồn tại. Hai metadata benchmark có frozen hashes khớp kết quả Colab.

Tài khoản thực thi: alias `hoang1`, xác minh owner `ndhoang2310`; chỉ dùng bridge `/home/hoang/.local/bin/kaggle-account`. Không chứa credentials trong mã/metadata/log. Dataset và notebooks private. Benchmark có sẵn `ndhoang2310/iir-b0-benchmarks`.

## Stages

`build_jobs.py` sinh notebook và metadata review được cho audit, extraction GPU, QE/fusion CPU, local-cache CPU, diffusion CPU, latency GPU. Geometry được sinh theo dataset/query shard từ pilot; report gắn outputs của mọi stage. Notebook chỉ đọc `/kaggle/input`, ghi `/kaggle/working/posttraining`.

`orchestrate.py --account hoang1` điều phối: audit → extraction → {QE/fusion, local cache, diffusion} → {geometry shards, direct latency} → report. Tối đa 3 CPU jobs và 1 GPU job ở bước geometry. Mỗi push có intent lưu trước khi gửi; timeout không tự push lại. Mỗi 45 giây kiểm tra status; downloads được xác minh bằng job_status.json. Lỗi dừng pipeline và được ghi vào state/pipeline_status.json. Chạy lại coordinator tiếp tục các stage đã complete, không rerun chúng. Job FAILED cần đọc log và xử lý trước khi reset state; không xóa state để thử lại mù.

Khi số input kernels vượt 15, consolidate.py đóng gói outputs đã tải về thành private dataset phẳng. Không nén folder thành tar/zip yêu cầu notebook tự giải nén.

## Protocol

Query bbox/crop/thumbnail giống runtime Colab; corruption áp dụng sau crop/resize. Gallery clean. 6 họ × 3 mức, noise/occlusion 3 realization, còn lại 1: 31 views/query; 110 queries × 31 × 2 encoders = 6.820 query forwards. Gallery = 22.910 forwards. Không làm tròn hoặc đổi budget giữa các model.

Blur Gaussian dùng OpenCV BORDER_REFLECT_101, kernel 2ceil(3sigma)+1; JPEG Pillow RGB/subsampling=2; noise/occlusion dùng numpy stable-hash seed. Pixel float được clip và round về RGB uint8 trước encoder/local features, được ghi checksum. Cùng pixel manifest được kiểm tra giữa hai encoder và local stage. Không tự loại query theo điểm model. Các ví dụ corruption được lưu, mọi mức suy giảm vẫn báo; không khẳng định occlusion luôn bảo toàn nhãn.

Tham số chính: AQE/alpha-QE/gated-QE k10/lambda.5/alpha3/tau.5/min3; RootSIFT 2000, Lowe .75 + mutual nearest, homography RANSAC 3px/.995/2000 iterations/min8 inliers; re-rank K50 eta.5; score fusion và full-list RRF c60; diffusion union/mutual k20/beta.9/100 iterations/tolerance1e-6. Geometry score, fallbacks, residual QE và normalization theo B1. Verified QE lọc tối đa 10 feedback hợp lệ từ top50, tối thiểu3; không re-rank lần hai.

Clean sensitivity một trục, cả hai encoder: QE k5/10/20, alpha1/3/5, lambda.25/.5/1, gate off/on và tau.4/.5/.6; G K20/50/100 eta.25/.5/.75; RRF c10/60/100; D k10/20/40 beta.5/.9/.99. Không retune main config theo sensitivity. Chỉ so hai model hiện có, không giả định đủ M0–M3.

## Vận chuyển outputs

Mỗi stage đóng gói payload thành tar stream chia part 128 MiB, checksum từng part; chỉ để lại manifest/status/config và các part trên Kaggle. Coordinator tải từng part bằng một invocation bridge, tránh giới hạn 180s khi tải hàng nghìn file riêng lẻ, rồi giải nén an toàn vào `outputs/<job>/unpacked`. Notebook phụ thuộc cũng giải nén trong working/input_view; benchmark/weights chỉ được liên kết file read-only từ input, không ghi vào mounted inputs. Không tải lại part đã khớp SHA. Báo cáo local đọc bản giải nén đầy đủ.

## Cache và thời gian

RootSIFT cache SQLite lưu SIFT native integer và tọa độ, chuyển float32 RootSIFT khi đọc; packing lossless, đã test so với tính trực tiếp. Cache gallery dùng chung encoder. Matching/RANSAC dùng một cache evidence cho cùng query-view/candidate trong mỗi shard. Pilot lấy top-ranked candidates thật, lưu mean/p95 và dự báo số shard ~3h với margin1.5; số này không là cam kết thời gian.

Extraction FP32 single-image để giữ aspect ratio và đo query uncached. GPU T4 được yêu cầu bằng CLI, dùng CUDA:0; không tuyên bố tận dụng cả hai GPU nếu phiên cấp hai T4. Quota/availability được xử lý bằng trạng thái remote, không đổi model/resolution/precision ngầm.

Retrieval-only đo riêng. Geometry main replay chi phí matching đo được trong cache, không gọi đó là direct end-to-end. Notebook latency GPU đo lại pipeline trực tiếp, query mới decode/corrupt/encode và local extraction trong timer, gallery/index cache được phép: clean + severe mỗi họ, query indices0/27 mỗi dataset, warm-up ngoài mẫu, 2 repeats, 1.120 mẫu toàn bộ. Báo median/p95 riêng, không trộn với composed timings. Độ biến thiên timing có giới hạn do số mẫu này.

## Kết quả và nghiệm thu

`summary.csv`, `conditions.csv`, `per_query.csv`, `sensitivity_clean.csv`, full rankings npz, paired AP deltas, tăng/giảm/không đổi, graph statistics/convergence, cache bytes, direct end-to-end timings và hình severity/quality–latency/failure examples. Corruption metric average realization trước, sau đó trung bình đều 18 condition; từng dataset báo riêng. Legacy AP trapezoidal đúng junk, P@10/Recall@K tính riêng.

Main coverage chính xác **68.200 query-method-condition cells**: cosine + Q3 + G2 + verifiedQE + D2 cho hai model; score fusion/RRF cho cặp model. Thiếu một cell hoặc direct latency chưa đủ sẽ khiến report_status INCOMPLETE. Negative gains và capped convergence được giữ. Report không tuyên bố multi-seed robustness hay tham số tối ưu.

Tests: `test_runtime.py` kiểm tra invariants; `test_integration.py` chạy miniature CPU toàn bộ retrieval stages, hai geometry shards, sensitivity và report; kiểm tra thiếu direct hardware timing bị báo INCOMPLETE. Máy local không có CUDA: full GPU paths phải được kiểm chứng ở Kaggle.
