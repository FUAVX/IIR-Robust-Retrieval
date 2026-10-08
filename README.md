# IIR-Robust-Retrieval

**Instance-Level Image Retrieval — GeM Baseline Reproduction & Robustness Research**

Dự án nghiên cứu của nhóm **FUAVX** nhằm xây dựng và đánh giá một hệ thống *instance-level image retrieval* (IIR), bắt đầu từ baseline **VGG16 + Generalized Mean (GeM) Pooling**, sau đó nghiên cứu cách cải thiện **general robustness** mà vẫn giữ khả năng phân biệt giữa các instance.

> **Trạng thái:** Đang chuẩn bị baseline **B0**. Chưa có kết quả training/mAP được nhóm xác nhận và **chưa chốt phương pháp B1**. Đây là repository nghiên cứu đang phát triển, **không phải** bản implementation mới do nhóm viết từ đầu.

## 1. Bài toán và mục tiêu

**Instance-level image retrieval** là bài toán tìm trong một gallery những ảnh chứa **cùng một đối tượng hoặc địa điểm cụ thể** với ảnh truy vấn (*query*), dù các ảnh có thể khác góc chụp, ánh sáng, độ phân giải, vật che khuất hoặc bối cảnh.

Ví dụ: với một ảnh query chụp một tòa nhà, mô hình cần xếp ảnh chụp **chính tòa nhà đó** ở những góc nhìn khác lên trên ảnh của các tòa nhà khác.

Nhóm theo đuổi hai mục tiêu nối tiếp:

1. **B0 — Reproduction:** Thiết lập, huấn luyện và đánh giá baseline GeM một cách có thể kiểm chứng, dựa trên paper và source code gốc.
2. **B1 — Robustness improvement:** Sau khi phân tích B0, đề xuất **một thay đổi có cơ sở** để giảm suy giảm hiệu năng retrieval dưới nhiều dạng biến thiên của ảnh, đồng thời giữ *instance discriminability* và chi phí tính toán hợp lý.

**General robustness** ở đây không đồng nghĩa với chỉ tăng mAP trên ảnh sạch. Nhóm dự định đo cả hiệu năng retrieval thông thường lẫn mức suy giảm dưới những phép biến đổi có kiểm soát; giao thức chi tiết sẽ được chốt sau khi B0 hoạt động ổn định. B1 hiện **chưa phải** một kiến trúc hoặc loss cụ thể.

## 2. Phương pháp baseline B0

```text
Query / Gallery image
        |
      VGG16                  (CNN feature extractor)
        |
    Feature maps
        |
    GeM Pooling              (global aggregation)
        |
 L2 normalization
        |
  512-D descriptor
        |
Similarity ranking -> retrieval results -> AP / mAP
```

- **Backbone:** VGG16; sử dụng nguồn khởi tạo **Caffe-converted VGG16 feature weights** theo source GeM gốc, không tự thay bằng torchvision classification weights.
- **Pooling:** GeM với tham số `p` học được (giá trị khởi tạo dự kiến `p = 3`).
- **Training data:** `retrieval-SfM-120k`, với các cặp query/positive và hard negatives được xác định từ metadata SfM; dùng *contrastive loss* và hard-negative mining.
- **Evaluation data:** Oxford5k và Paris6k với ground truth retrieval và phép đo **mean Average Precision (mAP)** theo giao thức legacy tương ứng.
- **Cấu hình B0 dự kiến:** Adam, contrastive margin `0.75`, `5` negatives, training image size `362`; evaluation image size `1024`, single-scale, không whitening và query expansion ở lần chạy cơ sở. **Những giá trị này phải được rà soát/chốt thành config trước khi train; chưa phải kết quả thực nghiệm.**
- **Ngân sách:** Ưu tiên Kaggle GPU; có thể cần thiết lập subset theo **SfM clusters** và checkpoint/resume. Không tự lấy random subset từng ảnh vì dễ phá quan hệ sampling của SfM.

B0 của nhóm là **reproduction có ràng buộc tài nguyên**, không mặc định tương đương toàn bộ thiết lập và kết quả công bố trong paper. Mọi khác biệt với paper hoặc upstream phải được ghi trong báo cáo.

## 3. Tiến độ hiện tại

| Hạng mục | Trạng thái xác nhận | Bước tiếp theo |
| --- | --- | --- |
| Import source gốc | Hoàn thành trên `main` (`886c117`) | Giữ `main` làm mốc tham chiếu |
| Khởi tạo cấu trúc nghiên cứu | Hoàn thành trên `B0` (`f00e6b2`) | Tiếp tục tài liệu và config B0 |
| **C0 — Modern API compatibility** | Đã commit (`4245b79`); **chưa test Kaggle runtime** | Kiểm tra môi trường thực tế |
| GitHub `origin` | Đã cấu hình tới `FUAVX/IIR-Robust-Retrieval` | Xác nhận các branch đã push thành công |
| **C1–C4** | Chưa hoàn tất | Triển khai theo thứ tự và kiểm thử |
| SfM/Oxford/Paris datasets | Chưa xác nhận đủ dữ liệu và Kaggle paths | Hoàng kiểm tra, lập dataset manifest |
| B0 training và mAP | **Chưa có kết quả được xác nhận** | Chỉ chạy sau các smoke test |
| B1 | **Chưa lựa chọn phương pháp** | Phân tích robustness/failure của B0 trước |

### C0–C4 nghĩa là gì?

Đây là **các mốc kỹ thuật do nhóm tự đặt**, không phải tên các phase trong paper GeM:

| Mốc | Ý nghĩa | Cần đạt trước khi chuyển bước |
| --- | --- | --- |
| **C0 — API compatibility** | Thay API không còn hỗ trợ: `Image.ANTIALIAS` → `Image.Resampling.LANCZOS`, `np.in1d` → `np.isin`. | Diff và cú pháp hợp lệ; **runtime Kaggle vẫn cần test**. |
| **C1 — Environment / Paths / Device** | Kiểm tra thư viện Kaggle; đọc dataset/weights từ đường dẫn thực tế; không tự download khi dữ liệu đã có; dùng đúng CPU/GPU và thư mục ghi output. | Load được metadata, ảnh, weights và model trên thiết bị đã chọn. |
| **C2 — Checkpoint / Resume** | Lưu và khôi phục model, optimizer, scheduler, epoch, config và các trạng thái cần thiết; kiểm tra không làm lệch lịch learning rate. | Chạy thử ngắt rồi resume, đối chiếu loss/LR/epoch. |
| **C3 — Training stability** | Kiểm tra sampling query-positive-negatives, mining, batch, loss, gradient, cập nhật optimizer và các trường hợp lỗi. | Smoke test vài batch/epoch nhỏ, không lỗi và không có loss bất thường. |
| **C4 — Evaluation validation** | Kiểm tra query crop, preprocessing, ranking, positive/junk ground truth, AP/mAP và cấu hình evaluation. | mAP được tính đúng và có thể chạy lại trên cùng checkpoint/protocol. |

> Hoàn thành C0–C4 **không có nghĩa** là đã reproduce thành công B0. Sau đó nhóm còn phải training, đánh giá, phân tích sự khác biệt với paper và lưu kết quả thực nghiệm.

## 4. Thành viên và trách nhiệm

| Thành viên | Phụ trách chính | Deliverable giai đoạn đầu |
| --- | --- | --- |
| **Phương** | Training & integration | Kaggle environment preflight; C1–C4 integration; checkpoint/resume; cấu hình và run log B0 |
| **Nguyên** | Model / methodology analysis | Đối chiếu paper–source về VGG16, GeM, loss, mining và scheduler; kiểm tra tính đúng đắn training/evaluation |
| **Hoàng** | Data & evaluation | Dataset manifest cho SfM-120k, Oxford5k, Paris6k; xác minh image/metadata/ground truth, Kaggle mount paths |

**Nguyên tắc:** chia *trách nhiệm triển khai*, **không chia kiến thức**. Cả ba đều cần hiểu pipeline và tham gia review. Phân công trên là kế hoạch đang đề xuất để team thống nhất.

## 5. Dữ liệu và môi trường

### Dữ liệu cần thiết

| Dataset | Mục đích | Lưu ý |
| --- | --- | --- |
| `retrieval-SfM-120k` | Training/validation, tuple supervision, SfM clusters | Cần đúng image IDs, cluster IDs, query/positive indices và quy trình mining |
| `oxford5k` | Benchmark retrieval | Có query bounding boxes, relevance/junk labels |
| `paris6k` | Benchmark retrieval | Có ground truth và quy trình AP/mAP tương ứng |

Tên các Kaggle mount path **chưa được xác nhận**. Không commit dữ liệu, pretrained weights hoặc checkpoints lớn vào Git.

### Môi trường hiện đại

Source gốc khai báo các dependency PyTorch rất cũ. Với Kaggle hiện đại, **không chạy** `pip install -r requirements.txt` hay `pip install .` một cách máy móc. Sử dụng cặp PyTorch/torchvision tương thích đã có trên Kaggle, kiểm tra phiên bản và đường dẫn `cirtorch` trước khi chạy.

Từ thư mục gốc repo (trong môi trường có các packages cần thiết):

```bash
python -B - <<'PY'
import sys, torch, torchvision, numpy as np, PIL
from PIL import Image
import cirtorch
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('NumPy:', np.__version__)
print('Pillow:', PIL.__version__)
print('CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('cirtorch:', cirtorch.__file__)
assert hasattr(Image.Resampling, 'LANCZOS')
assert callable(np.isin)
PY
```

Đây là **preflight**, chưa phải lệnh train. Trên Kaggle, dữ liệu thường được mount dưới `/kaggle/input/` (read-only) và output nằm dưới `/kaggle/working/`; đường dẫn cụ thể chỉ được chốt sau khi Hoàng xác nhận dataset layout.

Xem thêm [`configs/B0/README.md`](configs/B0/README.md) và [`report/B0/`](report/B0/) để biết tình trạng cấu hình, kết quả và quyết định thực nghiệm.

## 6. Cách cộng tác trên GitHub

Repository dự kiến: **[FUAVX/IIR-Robust-Retrieval](https://github.com/FUAVX/IIR-Robust-Retrieval)**.

| Branch | Vai trò |
| --- | --- |
| `main` | Source GeM tham chiếu được import; không sửa trực tiếp cho các thử nghiệm B0/B1 |
| `B0` | Branch tích hợp baseline và tài liệu/compatibility patches |
| `B0/kaggle-runtime` | Công việc Phương: Kaggle compatibility, checkpoint, training pipeline |
| `B0/methodology-audit` | Công việc Nguyên: audit và tài liệu phương pháp |
| `B0/data-preparation` | Công việc Hoàng: metadata, data preparation và evaluation |

Các branch theo người ở bảng trên là **tên đề xuất**, không khẳng định đã được tạo.

**Quy trình thực hiện một task:**

```bash
# Cập nhật baseline sau khi remote B0 đã được push
 git fetch origin
 git switch B0
 git pull --ff-only origin B0

# Ví dụ: tạo nhánh của riêng mình
 git switch -c B0/my-task

# Sau khi sửa và kiểm tra
 git status
 git add <files-related-to-your-task>
 git commit -m "docs: describe my task"
 git push -u origin B0/my-task
```

Sau đó mở **pull request: `B0/my-task` → `B0`** để thành viên khác review trước khi merge. Tránh sửa cùng một file trên nhiều branch không phối hợp; không force-push `main`/`B0` và không commit dữ liệu, secrets hoặc checkpoint. Đề xuất đặt `main` làm default branch và bật branch protection khi repository đã sẵn sàng.

## 7. Lộ trình tiếp theo: từ B0 đến B1

1. **Chốt dữ liệu và environment:** Hoàng lập manifest; Phương xác nhận Kaggle packages và GPU.
2. **Hoàn thành C1–C4:** Mọi thay đổi cần có lý do, diff và smoke test; Nguyên kiểm tra phương pháp không bị thay đổi âm thầm.
3. **Chốt B0 configuration:** ghi rõ architecture, pretrained weights, sampling, loss, optimizer, LR schedule, train/eval image sizes, số epoch và budget.
4. **Huấn luyện và đánh giá B0:** lưu run config, seed, commit SHA, checkpoint, log, mAP, phần cứng và các khác biệt với paper.
5. **Xây robustness evaluation:** giữ một giao thức clean và các stress tests theo blur/noise/illumination/compression/occlusion/scale; viewpoint thực cần dữ liệu ảnh đa góc phù hợp. Không diễn giải augmentation nhân tạo thành viewpoint 3D thật.
6. **Phân tích failure modes, chọn B1:** đề xuất thay đổi có giả thuyết rõ ràng, so sánh công bằng với B0 và làm ablation.

**Không có mAP hay mức cải thiện robustness nào được tuyên bố khi nhóm chưa hoàn thành thí nghiệm.**

## 8. Source gốc, paper và ghi nhận tác giả

Phần core `cirtorch/` được **import và kế thừa** từ repository gốc:

- **Original source:** [filipradenovic/cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
- **Primary paper:** Radenović, Tolias, Chum, *Fine-tuning CNN Image Retrieval with No Human Annotation*, TPAMI 2018 — [arXiv:1711.02512](https://arxiv.org/abs/1711.02512)
- **Related paper:** Radenović, Tolias, Chum, *CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples*, ECCV 2016 — [arXiv:1604.02426](https://arxiv.org/abs/1604.02426)
- **Oxford/Paris benchmark reference:** Radenović et al., *Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking*, CVPR 2018 — [arXiv:1803.11285](https://arxiv.org/abs/1803.11285)

Upstream source sử dụng giấy phép **MIT** (xem [`LICENSE`](LICENSE)); nhóm giữ lại copyright/license notice của upstream. Các thay đổi trong repository này được phân biệt với phần code gốc qua Git history và báo cáo thực nghiệm. Tài liệu gốc chi tiết về các lệnh train/test, optional whitening và các pretrained networks nằm tại link upstream ở trên.

---

**Last updated:** 2026-10-08 · **Project phase:** B0 preparation · **Organization:** FUAVX
