# SfM100: tách data preparation và tối ưu training pipeline

Lưu ý khi đọc gói `public-handoff/`: đây là snapshot đã xuất bản ngày
29/10/2026. Câu nhắc về `requirements.txt` cũ trong HANDOFF.md nói về phiên
bản repo tại thời điểm bàn giao; [requirements hiện tại](../../docs/installation.md)
đã bỏ ghim PyTorch 1.0. Trên Kaggle vẫn dùng môi trường và preflight của notebook,
không cài lại dependencies local vào session training.

Dùng hai notebook **01 và 02** cho luồng mới. Notebook `notebookc025fac755_optimized.ipynb` là bản v1 chỉ cải thiện logging/vận hành, giữ lại để đối chiếu. File gốc trong Downloads không bị sửa.

Gói bàn giao cho Codex trên máy/tài khoản khác: [IIR B0 Training Handoff](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-training-handoff). Đọc `public-handoff/HANDOFF.md`; dùng `02_train_fresh.ipynb` trong gói để chạy mới không cần checkpoint lịch sử. Ba input public đã được xác minh; checkpoint `linh2709/seg-ep7` hiện không truy cập được qua API và không nằm trong gói. Receipt xuất bản/kiểm tra gói nằm ở `kaggle-api/handoff-public-verification.json` sau khi xác minh hoàn tất.

## 1. Chuẩn bị dataset trên CPU, một lần

Import `01_prepare_sfm100_cpu.ipynb` lên Kaggle, attach full SfM, Oxford, Paris và Caffe feature weights như notebook gốc. Không cần checkpoint hay GPU.

Notebook chọn lại đúng subset seed42, loại hai self-pairs, kiểm tra metadata và giải mã toàn bộ ảnh. Bản local hiện dùng 16 luồng I/O cho stat/decode, tái dùng file sizes đã đo và 8 luồng copy/hash khi export. Probe Kaggle đo 16 ảnh tuần tự 4,41 giây, 4 luồng 1,08 giây trên hai sample khác nhau; đây không phải cam kết tốc độ toàn bộ dataset. Sau đó xuất `/kaggle/working/b0_sfm100_prepared` gồm metadata, provenance, filter, 16.692 ảnh SfM nguyên byte và manifest SHA-256. Ảnh không bị resize hoặc nén lại.

Dataset đã được tạo public ngày 2026-10-29: [IIR B0 SfM100 Prepared Seed42](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm100-prepared), version 1. Kaggle đã giải nén đủ 16.692 ảnh và 4 file metadata/manifest ngay tại root dataset. Attach dataset này vào notebook 02; không cần chạy lại notebook 01. Archive đầy đủ nằm trong thư mục local bị git-ignore `kaggle-api/publish-files/`; checkpoint không có trong dataset. Bước export kiểm tra dung lượng trước khi copy, giữ 2 GiB dự phòng và từ chối ghi đè. Nếu hết disk hoặc phiên bị ngắt, không dùng folder chưa có manifest hoàn tất; chạy lại trong session mới.

## 2. Training trên T4 x2

Import `02_train_sfm100_pipeline.ipynb`, attach dataset đã chuẩn bị, Oxford/Paris, Caffe weights và checkpoint hiện tại. Chọn T4 x2.

- Nếu tự dò input không tìm được duy nhất một manifest, đặt `PREPARED_DIR` trong cell discovery tới thư mục chứa `prepared_manifest.json`.
- Cell đầu đóng gói lại checkpoint đã giải nén, giữ đường dẫn gốc `/kaggle/input/models/linh2709/seg-ep7/pytorch/default/1/latest.pt`; sửa `checkpoint_dir` nếu input của bạn khác. `RESUME_CHECKPOINT` vẫn là `/kaggle/working/latest_resume.pt`.
- Giữ matching `best.pt` nếu controller yêu cầu. Không tự coi latest là best.
- Chạy Run All / Save & Run All trong session mới. Engine mới tự chạy smoke riêng ngay cả khi resume. Smoke không cập nhật checkpoint training chính.
- Fresh run: skip cell đóng gói checkpoint, đặt `RESUME_CHECKPOINT=None` và dùng run folder mới nếu folder cũ có dữ liệu.

Training không tạo lại subset, không copy ảnh, không quét đường dẫn hoặc giải mã toàn bộ 28k ảnh trước khi chạy. Nó kiểm tra identity/metadata, SHA-256 của 64 ảnh SfM và giải mã một sample 128 ảnh. Các ảnh chưa được sample vẫn có thể báo lỗi khi load sau đó. Giữ nguyên phiên bản các dataset benchmark đã dùng khi preparation.

## Những gì đổi trong data/training flow

- Mining lấy mẫu query/pool theo thứ tự RNG gốc, nhưng chỉ đọc/forward mỗi ảnh duy nhất một lần. Với full pool hiện tại: 17.953 -> 15.953 ảnh/epoch. Không cache descriptor train qua optimizer update hoặc qua epoch.
- Descriptor extraction gom ảnh cùng kích thước trong cửa sổ CPU tối đa tám ảnh; mỗi forward tối đa hai ảnh, có thêm giới hạn tổng pixel. Không padding hoặc đổi crop/resize. Trả descriptor đúng thứ tự ban đầu. Benchmark giữ descriptor trên GPU và chuyển kết quả về CPU một lần cuối extraction.
- Sau khi sort trên GPU, chuyển bảng ranks về CPU một lần rồi chọn negatives theo cluster. Tránh đọc từng scalar GPU trong vòng lặp Python. Bảng ranks hiện tại khoảng 255 MB RAM CPU; chưa đổi sang top-k gần đúng.
- Công thức contrastive loss được vector hóa cho tuple q+p+5, tránh quét từng nhãn trên GPU trong Python. Epsilon, margin và phép cộng loss được giữ nguyên.
- Training gom tối đa hai ảnh cùng kích thước trong mỗi tuple. Vẫn backward riêng từng tuple, cộng gradient cho batch và giữ nguyên thời điểm optimizer.step. Không gộp loss theo mean hoặc thay effective batch size.
- Validation dùng lại descriptor vừa mining vì giữa hai bước không có optimizer update. Không đọc/forward tuple images lần hai; giữ thứ tự tuple và drop_last gốc, giải phóng cache sau validation.
- Worker cap tối đa bốn áp dụng cho mọi loader. Không bật persistent_workers cho tuple dataset vì tuples đổi sau mỗi lần mining và worker replicas có thể giữ dữ liệu cũ.
- Giữ logging ít output và timing JSON của bản v1.

Chưa đổi FP32, image size 1024, learning rate, optimizer/scheduler, negative count, benchmark mỗi epoch hoặc model placement GPU0 -> GPU1 -> GPU0. JPEG decode và resize vẫn chạy khi cần đọc ảnh; chuẩn bị một lần không có nghĩa ảnh đã thành tensor sẵn. Không bật AMP/DDP và chưa che latency hai GPU bằng pipeline parallel.

## Checkpoint và mức tương đương

Giữ kiểm tra identity gốc với checkpoint lịch sử. Các thông số operational được ghi riêng; checkpoint mới thêm `pipeline_engine` gồm source hash/settings để chặn resume với engine khác. Dataset được chuẩn bị lại phải có identity khớp checkpoint; sai thì notebook dừng.

Microbatch có thể đổi rounding FP32/cuDNN và thứ hạng khi descriptor gần hòa nhau. Đây là run với pipeline mới, không phải cam kết tái lập bitwise. Khi OOM, notebook dừng; muốn giảm microbatch, sửa rõ `PIPELINE_SETTINGS` trước một run mới. Không tự giảm resolution hoặc đổi precision để chạy tiếp.

## Kiểm chứng

`verify_pipeline.py` đã chạy trên PyTorch 2.8.0 CPU với model CNN + GeM nhỏ và loss lấy từ source snapshot gốc:

- Descriptor theo thứ tự, contrastive loss và gradient của tất cả tham số khớp reference chạy từng ảnh trong sai số quy định. Ví dụ mixed-shape tuple: 7 -> 4 forward calls.
- Loss vector hóa và descriptor gradients khớp loss gốc cho một/năm tuple với descriptor 512-D.
- Nhiều Adam updates với gradient cộng hai tuple/update và train loss khớp reference.
- Extraction theo cửa sổ giữ gallery order và crop/resize query gốc.
- Full/partial-pool mining giữ sampled query, positive, negatives và trạng thái CPU RNG. Chỉ forward unique images.
- Validation reuse giữ loss/drop_last và giải phóng cache.
- Actual subset generator chạy bằng metadata giả có đúng counts lịch sử, loại đúng hai self-pairs. Export giữ nguyên bytes, SHA-256 và từ chối ghi đè.
- Cả hai notebook compile; training không chứa subset generator.

Chưa chạy model VGG16 thật trên Kaggle/T4 hoặc đo throughput, VRAM và mAP. Notebook có smoke trước full run nhưng smoke không bảo đảm mọi ảnh trong toàn bộ dataset đều vừa VRAM. Dataset đã public và xác minh truy cập anonymous, đủ cây thư mục CID. Archive SHA-256, hash từng ảnh và metadata đã kiểm tra; metadata thực tế vượt qua chính các assertions dữ liệu của notebook 02. Xem `kaggle-api/validation-report.json`, `notebook-compatibility.json`, `public-verification.json`. File gốc trong Downloads hiện không còn, nên đối chiếu sử dụng bản baseline đã lưu và notebook 02. Chưa chạy training GPU.

Chạy kiểm tra: `python3 artifacts/kaggle-training/verify_pipeline.py` trong môi trường có torch/numpy/Pillow. Build lại notebook sau khi sửa runtime: `python3 artifacts/kaggle-training/build_pipeline_notebooks.py`. Không sửa runtime mà dùng notebook chưa build lại.
