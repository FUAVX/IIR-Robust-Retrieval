# IEEE paper draft - SEG301

Bản paper tiếng Anh dựa trên kết quả đã có, không training hay chạy lại GPU. Tác giả tạm thời: **Author 1, Author 2, Author 3**, theo yêu cầu. Chưa chỉ định hội nghị hay giới hạn trang chính thức.

## Đọc và chỉnh sửa

- `seg301_ieee_paper.pdf`: bản chính IEEE conference hai cột.
- `seg301_ieee_paper.tex`: entry point; nội dung nằm trong `sections/`.
- `supplement.pdf` / `supplement.tex`: toàn bộ 100 sensitivity rows và secondary metrics.
- `references.bib`: 16 tài liệu tham khảo, bổ sung ViT, DINOv2, DELG và SuperGlobal từ nguồn tác giả.
- `figures/`: biểu đồ vector, dựng từ CSV; bảng nằm trong `tables/`.
- `data/`: CSV và cấu hình dùng cho bài; không chứa weights, API keys hoặc dữ liệu ảnh benchmark.
- `template/`: ZIP template tải về, bản gốc và thông tin nguồn/SHA256.
- `evidence_manifest.json`, `validation.json`: nguồn số liệu và kết quả kiểm tra.

Đường dẫn chia sẻ cho ZIP nguồn B0/R1/R2 cần điền tại
[`artifacts/README.md`](../../artifacts/README.md); các hash gốc nằm trong
`data/source_manifest.json`.

`evidence_manifest.json` giữ đường dẫn artefact gốc để truy xuất provenance và
trường `snapshot` trỏ tới bản CSV/JSON tương ứng trong `data/`. Khi clone repo,
không cần các thư mục `artifacts/results-consolidated` hoặc
`artifacts/kaggle-posttraining` để kiểm tra bảng số liệu đã công bố.

Chạy `python validate_paper.py` từ thư mục này để kiểm tra bảng, hash của bảy
snapshot, tài liệu tham khảo và cấu trúc/text của hai PDF. Build log `.log` không
được commit: script chỉ kiểm tra cảnh báo LaTeX khi log có mặt và ghi `SKIPPED`
nếu không có. Muốn kiểm tra cảnh báo layout trên một clone mới, biên dịch lại
paper và supplement trước khi chạy validator. Kiểm tra này không thay thế việc
xem PDF bằng mắt.

## Template đã tải trước khi viết

Dùng `IEEEtran.cls` nguyên bản V1.8b và `\documentclass[conference]{IEEEtran}` từ gói IEEE conference `conference-latex-template_10-17-19.zip`. Trang nguồn IEEE Author Center dẫn tới trang template IEEE. Các ZIP trực tiếp trên ieee.org trả về response rỗng ở môi trường này; gói thực tế được tải từ bản mirror của hội nghị ICPEA:

https://www.icpea.org/files/conference-latex-template_10-17-19.zip

Xem `template/provenance.json`. Không thay đổi class, lề, cỡ chữ chính hoặc khoảng cách cột. T1 font encoding giúp XeTeX dùng đúng Times theo IEEEtran thay vì fallback Latin Modern.

## Biên dịch

Dùng một trong các cách sau, chạy từ thư mục này:

```bash
tectonic seg301_ieee_paper.tex
tectonic supplement.tex
```

Hoặc với TeX Live/MiKTeX:

```bash
pdflatex seg301_ieee_paper.tex
bibtex seg301_ieee_paper
pdflatex seg301_ieee_paper.tex
pdflatex seg301_ieee_paper.tex
pdflatex supplement.tex
pdflatex supplement.tex
```

Có thể upload ZIP nguồn lên Overleaf và đặt `seg301_ieee_paper.tex` làm main document. File class giữ sẵn trong project; IEEEtran bibliography style là gói tiêu chuẩn của TeX.

## Tái tạo bảng/biểu đồ

```bash
python build_assets.py
python build_revision_assets.py
```

Cần Python, NumPy và Matplotlib. Script dùng CSV có sẵn trong project gốc; nếu không có thư mục artifacts, dùng bản CSV đi kèm trong `data/`. Không chạy encoder hay GPU. Các bảng được sinh ra thành macro trước khi đưa vào tabular, để tránh lỗi LaTeX input trong alignment.

`build_examples.py` chỉ là bước tùy chọn để tạo lại hình ví dụ; cần repo gốc với ảnh benchmark, descriptor và rankings đã lưu. Các ảnh minh họa được chọn theo tiêu chí lớn nhất/nhỏ nhất về AP gain, không được coi là mẫu đại diện ngẫu nhiên.

## Phạm vi kết luận

- Đây là nghiên cứu thực nghiệm các thành phần đã có, không tuyên bố thuật toán mới hay state of the art.
- B0 best, R1/R2 best là bằng chứng lịch sử; best R1 không có weight file trong ZIP nhưng có evaluation/history.
- Search study chỉ dùng R1/R2 latest epoch 100, FP32 inference và TF32 tắt.
- Không suy luận causal ablation từ so sánh training khác budget/precision/runtime.
- Audit diffusion mới giữ nguyên graph/descriptor: seed top-20 thay cho dense seed khắc phục mức giảm mạnh trong bảng lịch sử. Bảng riêng báo kết quả replay toàn bộ query corruption từ cache; bảng chính lịch sử vẫn giữ số gốc để truy vết. Không dùng latency của dense diffusion làm latency cho phiên bản sửa.
- Source augmentation/gate/BF16 và toàn bộ search đã có trong `artifacts/colab-training/` và `artifacts/kaggle-posttraining/`; hướng dẫn kiểm tra mới ở `artifacts/revision-validation/`.
- B0 chưa có robustness/post-training cùng giao thức; một training seed; sensitivity exploratory.
- Precision/recall trong summary cũ trung bình **19 điều kiện: clean + 18 corrupted**; corrupted mAP và worst-condition chỉ dùng **18 corrupted**. Supplement ghi rõ sự khác nhau này.
- Median end-to-end của QE thấp hơn cosine ở một vài nhóm do biến thiên phép đo; bài không coi đó là causal speed-up.

Trước khi nộp hội nghị cần điền tác giả/affiliation thật và áp dụng yêu cầu riêng của venue. Bản hiện tại là draft nội dung đầy đủ để đọc và chỉnh, chưa qua IEEE PDF eXpress hay quy trình nộp bài.
