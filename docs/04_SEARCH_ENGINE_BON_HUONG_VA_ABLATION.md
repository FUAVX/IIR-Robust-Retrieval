# Bốn hướng search engine và thiết kế ablation bắt buộc

Ngày: 30/10/2026 • **Cả bốn hướng đều phải triển khai và đánh giá. Không phải danh sách tùy chọn.**

Đọc [baseline](01_BASELINE_VA_QUY_UOC_THI_NGHIEM.md), [training](02_TRAIN_AUG_ATTENTION_VA_ABLATION.md), [evaluation](03_DANH_GIA_ROBUSTNESS_VA_KET_QUA.md) trước. Đây là đặc tả thực nghiệm đề xuất; các tham số khởi đầu không được mô tả là tối ưu hoặc đã kiểm chứng.

## 1. Kiến trúc và quy tắc chung

```text
Encoder checkpoint → descriptor query/gallery đúng encoder
                  → exact cosine baseline
                  → Q: sửa query và tìm lại
                  → G: ghép local + kiểm chứng top-K
                  → F: gộp các danh sách nhiều encoder
                  → D: lan truyền điểm trên graph gallery
```

Không cần train CNN mới cho Q/G/D. F dùng các encoder M1/M2 đã có. Local feature dùng RootSIFT để tránh cần train một matcher khác; dependency/version thực tế phải ghi trong environment.

Chạy sàng lọc trên E* được chọn bằng development, seed 42; Q/G/D sau đó áp dụng cấu hình cố định cho cả M0–M3. F là multi-encoder, báo riêng số encoder và chi phí.

Gallery không nhãn có thể dùng để xây graph và làm feedback; ground truth chỉ dùng chấm. Không được dùng nhãn để loại kết quả sai trước QE hoặc sửa cạnh graph.

## 2. Q — Query Expansion có kiểm soát

### Cơ chế và giả thuyết

Kết quả đầu của query có thể cung cấp ảnh cùng đối tượng rõ hơn. Gộp thông tin đó vào query rồi tìm lại toàn gallery. Rủi ro là query drift: kết quả đầu sai sẽ kéo query sang đối tượng khác.

Dùng q và d_i đã L2-normalized, s_i=q^T d_i. Chọn top-k có s_i>0. Loại self-match chỉ theo quy tắc định danh có thể biết ở triển khai, áp dụng thống nhất; không dùng danh sách junk/test labels để chọn feedback.

Đặc tả biến thể có residual query:

```text
u_i = 1                           với AQE
u_i = max(s_i,0)^alpha             với alpha-QE
w_i = u_i / sum_j u_j
q_new = L2(q + lambda * sum_i w_i*d_i)
```

Nếu tập rỗng, tổng trọng số hoặc norm quá nhỏ: giữ q. Chỉ một vòng mở rộng. Đây là biến thể có hệ số lambda và normalized weights, cần ghi công thức trong báo cáo; không tuyên bố giống từng chi tiết implementation alpha-QE của paper.

### Gate cụ thể

Gated alpha-QE giữ các ứng viên s_i >= tau. Chỉ mở rộng nếu có ít nhất r ứng viên đạt ngưỡng; nếu không, giữ query gốc. Dùng chính tập đã lọc để tính lại các trọng số w_i.

Gate chỉ dùng tín hiệu retrieval, không dùng ground truth hoặc loại corruption. Tau không là xác suất tin cậy; khả năng gate bảo vệ query khó cần được đo.

### Ma trận bắt buộc

| ID | Cấu hình | Phép so |
|---|---|---|
| Q0 | Cosine, không mở rộng | Mốc |
| Q1 | AQE, alpha=0, cùng k/lambda | Q1−Q0: mở rộng trung bình |
| Q2 | alpha-QE | Q2−Q1: trọng số |
| Q3 | alpha-QE + gate | Q3−Q2: cơ chế từ chối |

Khởi đầu k=10, lambda=0,5, alpha=3, r=3. Chọn tau trên development: lấy quantile {25%,50%,75%} của top-k positive cosine scores development (positive ở đây nghĩa s>0, không phải nhãn relevant), lưu thành các ngưỡng số trước test.

Sensitivity bắt buộc: k={5,10,20}; alpha={1,3,5}; lambda={0,25;0,5;1}; gate off/on và ba tau trên. Dùng từng trục với các tham số còn lại cố định ở cấu hình tham chiếu; không chạy Cartesian toàn bộ. Khi làm ablation gate off/on, giữ cùng k/alpha/lambda, không để gain đến từ retuning thành phần khác.

Ngoài metric chung, lưu gate-open rate, số feedback images, tỷ lệ query AP tăng/giảm và AP delta từng query. Q3 không được coi là thành công chỉ vì gate thường đóng và trở về baseline.

## 3. G — Geometric Re-ranking

### Cơ chế

Lấy top-K bằng global descriptor, ghép local features query–candidate, kiểm tra hình học và xếp hạng lại trong tập này. Global search tạo ứng viên; geometry kiểm tra các chi tiết có bố trí hợp lý.

Đặc tả khởi đầu:

- SIFT tối đa 2.000 keypoints/image, dùng cùng grayscale preprocessing và version; ghi detector params.
- RootSIFT: L1-normalize vector SIFT không âm, lấy căn bậc hai, L2-normalize.
- 2-nearest matching + Lowe ratio 0,75 + mutual matching; lưu n_matches.
- Homography RANSAC, reprojection threshold 3 px trên ảnh local xử lý cạnh dài tối đa 1024; confidence 0,995; max iterations 2.000; seed cố định nếu API hỗ trợ.
- Valid geometry cần ít nhất 8 inliers, homography hữu hạn và không suy biến; không áp dụng homography cho mọi cảnh như chân lý. Với cảnh phi phẳng/góc nhìn lớn có thể thất bại.
- Lưu inlier count, inlier ratio và tỷ lệ vùng query được phủ bởi inliers để phân tích lặp họa tiết. Không dùng coverage như điều kiện mới sau khi nhìn test mà không tăng protocol.

### Score và fallback

Dùng thành phần bounded để không cộng raw inlier count với cosine khác thang đo:

```text
global_score = (cosine+1)/2
local_score = min(n_matches/50,1)
geo_score = (n_inliers/max(n_matches,1)) * min(n_inliers/20,1)
rerank_score = (1-eta)*global_score + eta*extra_score
```

G1 dùng local_score; G2 dùng geo_score với cùng eta. Nếu query không có đủ keypoint để ghép, fallback toàn query về global ranking. Nếu một candidate thiếu local/geometry, extra_score=0 theo policy đã khóa; đây là không có bằng chứng bổ sung, không khẳng định candidate chắc chắn sai. Khi không candidate nào có valid geometry, G2 fallback toàn query về global ranking. Ghi fallback rate.

Chỉ reorder top-K, giữ phần đuôi nguyên; tie-break theo global rank rồi image ID. Quy tắc này bảo toàn tập kết quả nhưng không thể cứu ảnh đúng nằm ngoài top-K.

### Ma trận bắt buộc

| ID | Cấu hình | Mục đích |
|---|---|---|
| G0 | Cosine | Mốc |
| G1 | Re-rank bằng local matches | Lợi ích local evidence |
| G2 | Re-rank bằng geometric inliers | Giá trị kiểm chứng hình học |

Khởi đầu K=50, eta=0,5. Sensitivity K={20,50,100}, eta={0,25;0,5;0,75}, từng trục. G1/G2 cùng candidate set, matching và eta. Không tăng detector budget riêng cho G2.

Báo relevant Recall@K trước re-rank, latency local query extraction, matching/RANSAC, local storage và failure rate. Gallery local cache dùng chung các encoder nếu ảnh/preprocessing giống nhau; query corruption cần local feature riêng đúng ảnh đã biến đổi.

## 4. F — Rank Fusion

### Cơ chế

M1 và M2 tạo hai bảng điểm và danh sách riêng, bằng gallery descriptor tương ứng. RRF cộng đóng góp theo hạng:

```text
RRF(d) = sum_m 1/(c + rank_m(d)), rank bắt đầu từ 1
```

Main experiment dùng full gallery rankings vì gallery khoảng 5k–6k, tránh ứng viên thiếu gây nhiễu ablation. Nếu dùng truncated lists để đo tốc độ, ảnh không xuất hiện đóng góp 0; báo depth riêng và không trộn với main.

Score fusion đối chứng: `(cosine_M1 + cosine_M2)/2` trên cùng gallery ID. Cosine cùng miền số nhưng phân phối có thể khác; đây chính là khác biệt cần so với rank fusion, không tự calibrate bằng test labels.

### Ma trận bắt buộc

| ID | Cấu hình |
|---|---|
| F0 | M1 cosine riêng |
| F1 | M2 cosine riêng |
| F2 | M3 cosine: một model AUG + Attention |
| F3 | Average-score fusion(M1,M2) |
| F4 | RRF(M1,M2) |
| F5 | RRF(M0,M1,M2,M3) |

Khởi đầu c=60; sensitivity c={10,60,100} trên development. Giữ cùng c khi so F4/F5 để thấy ảnh hưởng thêm hệ thống; không kỳ vọng thêm model luôn tốt hơn.

F5 sử dụng cả M3, vì vậy không gọi F5−F4 là hiệu quả riêng một thành phần; đó là thêm hai rankers. F2 với F4 là trade-off một model kết hợp vs hai model ensemble, không phải so cùng compute.

Không cộng/average raw embeddings của hai mạng đã train độc lập. Cache riêng từng encoder. Khi đánh giá nhiều seed, ghép các model cùng seed để giảm biến số và ghi rõ pairing.

Báo quality, số encoder, tổng descriptor/index storage, tổng query feature-extraction time và end-to-end latency. Full RRF không cần train thêm nhưng không miễn phí inference.

## 5. D — Graph Diffusion

### Cơ chế

Graph chỉ gồm gallery images. Mỗi ảnh nối k láng giềng theo cosine, không dùng nhãn. Query tạo seed scores rồi lan truyền trên graph. Đây là graph ranking, không phải diffusion sinh ảnh.

Đặc tả:

1. Descriptor gallery đúng encoder, chuẩn L2.
2. Tìm kNN exact, bỏ self-edge; dùng deterministic image-ID tie-break.
3. Directed weight W_ij=max(cos(d_i,d_j),0)^3 trên cạnh kNN.
4. D1 union graph: W=max(W,W^T), lấy max theo từng phần tử.
5. D2 mutual graph: chỉ giữ cạnh khi cả i→j và j→i tồn tại; weight cosine như trên.
6. Degree D_ii=sum_j W_ij; S=D^(-1/2) W D^(-1/2). Isolated node dùng inverse degree=0, không chia cho 0.
7. y_i=max(cos(q,d_i),0)^3 cho toàn gallery, L1-normalize khi sum>0.
8. Khởi tạo f_0=y; lặp `f_(t+1)=(1-beta)*y+beta*S*f_t`.
9. Dừng khi relative L2 change <1e-6 hoặc 100 iterations; xếp hạng f giảm dần.

Nếu sum y=0 hoặc số học lỗi, fallback cosine và ghi lỗi. Với beta<1, quy tắc chuẩn hóa phù hợp giúp phép lặp ổn định. Kiểm tra convergence thực tế; chạm cap phải được ghi, không âm thầm coi đã hội tụ.

Không thêm query vào gallery hoặc dùng các query test khác để xây graph. Tái dùng gallery graph qua corruption của cùng encoder; encoder/descriptor thay đổi phải xây graph lại.

### Ma trận bắt buộc

| ID | Cấu hình | Phép so |
|---|---|---|
| D0 | Cosine | Mốc |
| D1 | Diffusion union-kNN | Giá trị graph |
| D2 | Diffusion mutual-kNN | Giảm liên kết một chiều |
| Q2 | alpha-QE từ bảng Q | Đối chiếu query feedback và graph |

Khởi đầu k=20, beta=0,9; sensitivity k={10,20,40}, beta={0,5;0,9;0,99}, từng trục. Giữ cùng k/beta khi so D1/D2. Báo số cạnh, isolated-node fraction, connected components, iterations, offline build time, storage và online latency.

Graph affinity dùng toàn gallery không nhãn là transductive retrieval; cần công bố. Không gọi đây là test leakage chỉ vì dùng gallery, nhưng dùng test relevance labels để sửa graph là leakage.

## 6. Thứ tự triển khai

1. Cosine exact/evaluator + descriptor cache.
2. Q1–Q3 (tận dụng descriptor, ít phụ thuộc mới).
3. F3–F5 (khi M1–M3 checkpoint đã sẵn).
4. Local cache/matching/RANSAC rồi G1–G2.
5. Gallery graph + D1–D2.
6. Cross-encoder và pipeline kết hợp bên dưới.

Các bước 3–5 không được bỏ nếu chất lượng bước trước chưa tăng. Mỗi hướng có thể có kết quả âm; phải báo đầy đủ.

## 7. Ma trận tương tác bắt buộc

Sau tuning trên E*, khóa một đại diện Q*, G*, D* tốt nhất theo development trong mỗi họ, kể cả nếu chưa vượt cosine. Áp dụng cùng hyperparameters lên từng encoder, trích lại descriptor/graph đúng encoder.

| Encoder | Cosine | +Q* | +G* | +D* |
|---|---|---|---|---|
| M0 | chạy | chạy | chạy | chạy |
| M1 | chạy | chạy | chạy | chạy |
| M2 | chạy | chạy | chạy | chạy |
| M3 | chạy | chạy | chạy | chạy |

16 cấu hình, trong đó bốn cosine đã có. Fusion báo riêng F0–F5 vì dùng nhiều encoder. Không chọn riêng tham số tối ưu trên test cho mỗi encoder để làm bảng đẹp hơn.

## 8. Pipeline kết hợp bắt buộc: verified QE

Chạy trên E* với config khóa:

```text
C0: cosine
C1: geometric re-ranking (G2)
C2: alpha-QE không geometry (Q2)
C3: cosine top-K → geometric verification → lấy feedback đạt chuẩn
    → alpha-QE trên feedback → search lại toàn gallery
```

C3 dùng same K và valid geometry rule của G2; lấy tối đa k feedback có cosine>0 và valid geometry, theo cosine ban đầu. Trọng số Q2 và lambda giữ nguyên. Nếu dưới r=3 ảnh hợp lệ, trả query gốc. Không rerank lần thứ hai trong C3 để tránh thêm một biến số chưa tách.

Báo C3−C2 (lọc geometry trước feedback), C3−C0 (pipeline đầy đủ), gate rate và latency. Dù tên C1 là re-ranking, việc geometry trong C3 dùng để chọn feedback, không phải cùng vị trí tác động; phải diễn giải đúng.

Cả bốn hướng được thực hiện riêng + cross-encoder, còn pipeline kết hợp bắt buộc này tập trung Q/G. Không yêu cầu chồng cả Q/G/F/D trong một pipeline để được coi là “làm hết”; cách chồng toàn bộ có nhiều thứ tự và biến số không cần thiết để trả lời đóng góp từng hướng.

## 9. Ngân sách và cách đếm runs

Một training run ≠ một retrieval evaluation. Nhiều cấu hình search dùng lại cùng checkpoint và descriptor; không phải mỗi hàng là train lại VGG16.

- Model: 4 nhánh × 3 seed = 12 training runs tổng, trừ các run cũ đủ điều kiện tái sử dụng.
- Retrieval families: Q0–Q3, G0–G2, F0–F5, D0–D2; các hàng cosine trùng được tái sử dụng và ghi reference.
- Sensitivity: thực hiện trên development seed42, từng trục; lưu cả cấu hình kém.
- Test: chạy các ablation fixed-setting đã khóa, đại diện từng họ, cross-encoder, fusion và C0–C3 trên clean + corruption protocol.
- Cùng ảnh corruption và cache đúng model; đo latency end-to-end riêng không dùng query cache.

Không cộng máy móc số hàng vì có đối chứng trùng. Tạo manifest `evaluation_plan.csv` có unique key (checkpoint hashes, method, config hash, dataset, corruption protocol, seed) để tránh chạy lặp và tránh thiếu.

## 10. Checklist nghiệm thu cả bốn hướng

- [ ] Q: AQE, alpha-QE, gate; sensitivity và query drift analysis.
- [ ] G: local-only vs geometry; K sweep, candidate recall và fallback.
- [ ] F: hai model riêng, M3, score fusion, RRF2, RRF4; chi phí đủ encoder.
- [ ] D: union vs mutual, graph stats, beta/k sensitivity, convergence.
- [ ] Cross-encoder 4×4; configs cố định.
- [ ] C0–C3 verified QE pipeline.
- [ ] Clean/corruption AP từng query, aggregate, latency/storage, seeds.
- [ ] Báo kết quả không cải thiện và giới hạn, không chỉ giữ best numbers.

## 11. Nguồn và cách ghi đóng góp

- [Paper nền GeM, whitening, alpha-QE](https://arxiv.org/abs/1711.02512v2).
- [Total Recall: visual query expansion và spatial verification](https://www.robots.ox.ac.uk/~vgg/publications/2007/Chum07b/).
- [Reciprocal Rank Fusion, SIGIR 2009](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/).
- [Efficient Image Retrieval via Decoupling Diffusion into Online and Offline Processing](https://arxiv.org/abs/1811.10907).
- [Revisiting Oxford and Paris](https://arxiv.org/abs/1803.11285).

Các công thức residual QE, gate, score fusion geometry và graph settings cụ thể ở đây là thiết kế triển khai để nhóm kiểm thử, không tuyên bố tái tạo chính xác mọi chi tiết của các bài nguồn. Chưa có kết quả đo chứng minh chúng tăng robustness trong dataset nhóm.
