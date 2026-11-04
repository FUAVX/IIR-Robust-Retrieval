# Roadmap nghiên cứu và kiểm chứng baseline GeM (B0)

> **Dự án:** IIR-Robust-Retrieval<br>
> **Vai trò theo kế hoạch nhóm:** Methodology & Model Analysis<br>
> **Mục tiêu:** Hiểu phương pháp GeM gốc, đối chiếu paper–source–B0 và hỗ trợ kiểm chứng một reproduction đúng, có thể lặp lại, có kết quả mAP đáng tin cậy.<br>
> **Cách sử dụng:** Đánh dấu `[x]` khi hoàn thành một đầu việc, thêm link/note ở phần *Minh chứng / Ghi chú*.
> **Trạng thái ban đầu:** Các ô chưa đánh dấu là những việc *cần theo dõi*, không khẳng định rằng nhóm chưa từng thực hiện chúng.

## Tài liệu tham khảo

- **Original paper:** Radenović, Tolias & Chum — *Fine-tuning CNN Image Retrieval with No Human Annotation* (CVPR 2018): <https://arxiv.org/pdf/1711.02512>
- **Original source:** <https://github.com/filipradenovic/cnnimageretrieval-pytorch>
- **Survey:** *Deep Learning for Instance Retrieval: A Survey* — tài liệu tổng quan để đặt GeM trong bối cảnh nghiên cứu IIR.
- **Tài liệu nội bộ liên quan:** [research brief](../project/research_brief.pdf), [kế hoạch triển khai B0](implementation_plan.pdf), [audit phương pháp](../../report/B0/B0_methodology_audit.md).

## Tổng quan tiến độ

| Giai đoạn | Mục tiêu | Trạng thái | Minh chứng / Ghi chú |
|---|---|---|---|
| 1. Nghiên cứu original paper | Hiểu motivation, contributions, architecture, training, experiments | ☐ Chưa đánh dấu hoàn thành | |
| 2. Audit original source | Đối chiếu paper với mã nguồn GeM chính thức | ☐ Chưa đánh dấu hoàn thành | |
| 3. Review B0 của nhóm | Kiểm chứng methodology ở C2/C3/C4 | ☐ Chưa đánh dấu hoàn thành | |
| 4. Phân tích reproduction | Kiểm tra mAP, khác biệt và failure cases | ☐ Chờ kết quả B0 | |

---

## Giai đoạn 1 — Nghiên cứu original paper

**Mục tiêu:** Hiểu *tại sao* phương pháp được đề xuất, *nó hoạt động thế nào* và *phần nào cần reproduce*. Ưu tiên đọc paper trước khi đào sâu code.

### 1.1. Motivation & Contributions

- [ ] Đọc **Abstract** và **Introduction**.
- [ ] Nắm bối cảnh Instance-Level Image Retrieval trước GeM.
- [ ] Trả lời: **Trước GeM, các phương pháp retrieval có hạn chế gì?**
- [ ] Trả lời: **Original paper đóng góp những gì? GeM pooling có phải đóng góp duy nhất không?**
- [ ] Trả lời: **Vì sao tác giả fine-tune CNN mà không cần nhãn do con người gán?**

### 1.2. Training Data / SfM-derived Supervision

- [ ] Hiểu cách SfM tạo quan hệ hình học và thông tin chọn ảnh.
- [ ] Hiểu cách xây dựng query–positive pairs.
- [ ] Hiểu điều kiện lựa chọn negative candidates và hard negatives.
- [ ] Phân biệt **nhãn thủ công** với **supervision khai thác từ SfM**.

### 1.3. Model Architecture

- [ ] Phân tích VGG16 convolutional backbone.
- [ ] Giải thích công thức GeM, vai trò tham số `p`.
- [ ] Phân tích L2 normalization và descriptor 512-D.
- [ ] Hiểu cách descriptor được dùng để xếp hạng ảnh.

### 1.4. Training Methodology

- [ ] Hiểu Siamese/shared-weight learning trong mô hình retrieval.
- [ ] Hiểu cấu trúc training tuple và positive/negative sampling.
- [ ] Phân tích contrastive loss, margin và gradient ở mức cần thiết.
- [ ] Phân tích hard-negative mining và thời điểm refresh.
- [ ] Ghi lại các thông số training quan trọng từ **paper**: optimizer, learning rate, scheduler, số epoch, batch/tuple, số query, cấu hình ảnh, initialization… (chỉ ghi những gì paper thực sự cho biết).

### 1.5. Experiments & Evaluation

- [ ] Xác định training/evaluation datasets trong paper.
- [ ] Phân biệt Oxford5k/Paris6k **legacy** và ROxford/RParis.
- [ ] Hiểu evaluation pipeline: query crop → descriptor → similarity → ranking → AP/mAP.
- [ ] Ghi lại kết quả paper và điều kiện tương ứng (scale, whitening, query expansion, v.v.).

**Đầu ra giai đoạn 1:**

- [ ] Bản tóm tắt ngắn: *problem → motivation → contributions → pipeline → training → evaluation*.
- [ ] Danh sách các thông số/chi tiết còn chưa rõ cần kiểm tra trong code.

**Minh chứng / Ghi chú:**

```text
- Link ghi chú:
- Paper sections đã đọc:
- Câu hỏi còn mở:
```

---

## Giai đoạn 2 — Đối chiếu original paper với source GeM

**Mục tiêu:** Xác minh source code thực hiện các ý tưởng trong paper như thế nào, chỉ ra điểm khớp, khác biệt và chưa rõ. **Không mặc định source và paper giống nhau ở mọi chi tiết.**

### File/khối chức năng cần kiểm tra

- [ ] `cirtorch/datasets/traindataset.py` — training tuple, metadata, hard-negative mining.
- [ ] `cirtorch/networks/imageretrievalnet.py` — backbone, GeM, L2, descriptor.
- [ ] `cirtorch/layers/loss.py` — contrastive loss và margin.
- [ ] `cirtorch/examples/train.py` — training loop, optimizer, LR scheduler, checkpoint.
- [ ] `cirtorch/utils/evaluate.py` — AP/mAP và xử lý positive/junk.
- [ ] Kiểm tra đường đi của query/gallery extraction và ranking trong code evaluation/training liên quan.

### Bảng audit cần điền

| Thành phần | Paper setting | Source behavior | TEAM B0 decision | Impact / Open question |
|---|---|---|---|---|
| Backbone | Cần đọc paper | Cần audit source | VGG16 | |
| Pooling | Cần đọc paper | Cần audit source | GeM, 1 scalar `p` trainable | |
| Descriptor | Cần đọc paper | Cần audit source | L2-normalized, 512-D | |
| Training tuple | Cần đọc paper | Cần audit source | 1 query + 1 positive + 5 negatives | |
| Contrastive loss / margin | Cần đọc paper | Cần audit source | Contrastive loss; margin TBD | |
| Hard-negative mining | Cần đọc paper | Cần audit source | Có sử dụng; chi tiết cần xác minh | |
| Optimizer / LR / scheduler | Cần đọc paper | Cần audit source | TBD | |
| Checkpoint / resume | Cần đối chiếu theo mục tiêu reproduction | Cần audit source | C2 cần kiểm chứng | |
| Evaluation | Cần đọc paper | Cần audit source | Oxford5k/Paris6k legacy, single-scale `[1]`, whitening OFF, QE OFF | |

**Đầu ra giai đoạn 2:**

- [ ] Bảng audit có chứng cứ (paper section/page, source file/function/line hoặc commit).
- [ ] Danh sách mismatch và câu hỏi nhóm phải quyết định trước full B0 training.

**Minh chứng / Ghi chú:**

```text
- Source commit đang đối chiếu:
- Link bảng audit:
- Open issues:
```

---

## Giai đoạn 3 — Review implementation B0 của nhóm

**Mục tiêu:** Xác minh B0 đúng với methodology đã chốt và có thể tái lập; không âm thầm thay model/loss/mining/evaluation nhằm tăng điểm.

### C2 — Checkpoint / Resume / LR Scheduler

- [ ] Kiểm tra checkpoint lưu các trạng thái cần thiết để resume.
- [ ] Kiểm tra khôi phục đúng epoch/step và optimizer state.
- [ ] Đối chiếu thời điểm `scheduler.step()` và learning-rate policy đã chốt.
- [ ] Kiểm tra resume không lặp epoch hoặc decay LR ngoài ý muốn.
- [ ] Review log/save–load test do người phụ trách training cung cấp.

### C3 — Training Stability / Tuple / Mining

- [ ] Xác nhận tuple đúng 7 ảnh: 1Q + 1P + 5N.
- [ ] Kiểm tra `target` và quan hệ positive/negative dùng cho loss.
- [ ] Kiểm tra sampling/mining không rỗng, không sai cluster/ràng buộc.
- [ ] Kiểm tra loss hữu hạn, backward hoạt động, gradient hợp lệ.
- [ ] Kiểm tra trọng số thực sự thay đổi sau optimizer step.
- [ ] Review log mini-run trước full training.

### C4 — Retrieval Evaluation / mAP

- [ ] Kiểm tra query bounding-box crop đúng protocol legacy.
- [ ] Kiểm tra đúng thứ tự image IDs, query/gallery descriptors và ranking indices.
- [ ] Kiểm tra logic `ok`, `junk` và AP/mAP.
- [ ] Làm **toy test AP/mAP tính tay** để đối chiếu code.
- [ ] Kiểm tra evaluation cấu hình B0: single-scale `[1]`, whitening OFF, query expansion OFF.
- [ ] Kiểm tra log evaluation ghi rõ checkpoint, dataset, protocol và cấu hình.

**Đầu ra giai đoạn 3:**

- [ ] Review/checklist C2, C3, C4 kèm kết quả pass/fail và bằng chứng.
- [ ] Vấn đề cần sửa hoặc cần quyết định được ghi thành issue/PR rõ ràng.

**Minh chứng / Ghi chú:**

```text
- Branch/commit B0:
- C2 review / log:
- C3 review / log:
- C4 review / toy AP test:
```

---

## Giai đoạn 4 — Phân tích kết quả reproduction (sau khi B0 có run hợp lệ)

**Điều kiện bắt đầu:** Nhóm có checkpoint rõ ràng, log training/evaluation, dataset/protocol đã xác nhận và kết quả mAP có thể tính lại.

- [ ] Xác minh mAP là kết quả thật, đúng benchmark và protocol.
- [ ] So sánh kết quả với original paper **trong điều kiện tương ứng**.
- [ ] Ghi các điểm khác nhau về training budget, initialization, optimizer, mining, scale, whitening và query expansion.
- [ ] Kiểm tra các query có top-ranked retrieval sai.
- [ ] Phân loại failure cases: blur, noise, illumination, occlusion, clutter, scale, viewpoint.
- [ ] Chỉ sau phân tích B0 mới đề xuất giả thuyết cải tiến/ablation cho B1.

**Đầu ra giai đoạn 4:**

- [ ] Báo cáo kết quả B0 có cấu hình, chứng cứ và mức độ so sánh được với paper.
- [ ] Báo cáo failure analysis làm cơ sở lựa chọn hướng B1.

**Minh chứng / Ghi chú:**

```text
- Run ID / commit / checkpoint:
- Dataset & protocol:
- mAP Oxford5k:
- mAP Paris6k:
- Khác biệt so với paper:
- Link failure analysis:
```

---

## Việc nên làm ngay (Next Action)

1. [ ] Mở original paper: <https://arxiv.org/pdf/1711.02512>.
2. [ ] Đọc **Abstract → Introduction**; chưa cần đọc code cùng lúc.
3. [ ] Viết câu trả lời ngắn cho ba câu hỏi:
   - Các phương pháp IIR trước GeM có vấn đề/hạn chế nào?
   - Paper đóng góp gì ngoài (hoặc cùng với) GeM pooling?
   - Vì sao có thể fine-tune CNN mà không cần human annotation?
4. [ ] Ghi lại câu hỏi chưa hiểu để thảo luận từng phần.
5. [ ] Sau khi rõ motivation mới chuyển sang phần SfM, model architecture và training methodology.

> **Nguyên tắc xuyên suốt:** Luôn tách riêng **paper nói gì**, **source code làm gì**, **B0 của nhóm đã chốt gì**, và **điều gì còn chưa được xác minh**.

## Nhật ký tiến độ

| Ngày | Đã thực hiện | Điều đã hiểu / phát hiện | Việc tiếp theo |
|---|---|---|---|
| YYYY-MM-DD | | | |
| YYYY-MM-DD | | | |
| YYYY-MM-DD | | | |
