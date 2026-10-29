# Bàn giao cho Codex trên máy / tài khoản Kaggle khác

Cập nhật: 2026-10-29 (Asia/Ho_Chi_Minh).

## Mục tiêu và trạng thái

Huấn luyện B0 VGG16-GeM trên SfM subset-100 seed 42. Người dùng muốn Codex trên máy khác tiếp nhận công việc và có thể chạy bằng tài khoản Kaggle khác. Gói này cung cấp code và báo cáo; ảnh, benchmark và weights ở ba dataset public bên dưới. Không cần truy cập máy cũ hoặc clone repository để chạy notebook.

**Entry point khuyến nghị: `02_train_fresh.ipynb`.** Bản này chạy mới từ Caffe pretrained weights, không cần checkpoint epoch 7. Nó tự smoke trước full training. Chưa chạy training VGG16 thật trên GPU với engine mới; không có kết quả throughput, VRAM hoặc mAP để cam kết.

Gói code/báo cáo public: https://www.kaggle.com/datasets/ndhoang2310/iir-b0-training-handoff

## Các input cần attach

| Input | Dataset public | Phiên bản đã xác minh |
| --- | --- | --- |
| SfM đã chuẩn bị | https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm100-prepared | 1 |
| Oxford5k + Paris6k | https://www.kaggle.com/datasets/ndhoang2310/iir-b0-benchmarks | 1 |
| Caffe VGG16 feature weights | https://www.kaggle.com/datasets/ndhoang2310/iir-b0-vgg16-weights | 1 |

Cả ba đã được kiểm tra metadata anonymous và `is_private=false`. Không cần full SfM để training: subset đã có đủ ảnh. Không tải 5,84 GB về máy điều khiển chỉ để chạy Kaggle; attach trực tiếp bằng dataset_sources/Add Input. Gói handoff chỉ cần tải một lần và nhỏ hơn nhiều so với ảnh.

Weights phải là `imagenet-caffe-vgg16-features-d369c8e.pth`, SHA-256:

```text
d369c8e593250e421eda192512263dccd35edb5c4b57e97312651bac10a51b7d
```

Notebook tự kiểm tra weights/metadata identity; sai thì dừng. Benchmark và weights không nằm trong dataset subset.

## Tiếp nhận trên máy khác

1. Cài Kaggle CLI trong môi trường Python phù hợp (`python -m pip install kaggle`) nếu chưa có.
2. Cấu hình tài khoản Kaggle của máy mới theo cách CLI hỗ trợ. Không chuyển API key/token của máy cũ. Không đưa credentials vào notebook, dataset, Git hoặc báo cáo.
3. Tải gói nhỏ:

```sh
kaggle datasets download -d ndhoang2310/iir-b0-training-handoff -p handoff --unzip
cd handoff
python verify_bundle.py
```

4. Đọc báo cáo này và các JSON kiểm chứng. Không tự chạy lại preparation hoặc dùng `requirements.txt` cổ của repository.

Prompt có thể gửi Codex mới:

> Đọc HANDOFF.md trong gói ndhoang2310/iir-b0-training-handoff. Kiểm tra bundle-manifest.json, dùng 02_train_fresh.ipynb và ba input public version 1. Chạy trên tài khoản Kaggle của tôi với T4 x2, theo dõi smoke và log trước khi full training. Không tạo lại subset, không đổi FP32/image size/optimizer/loss để né lỗi. Báo chính xác lỗi nếu GPU, quota, weights hoặc identity không phù hợp. Nếu muốn resume, xử lý mục checkpoint trong báo cáo trước.

## Chạy bằng giao diện Kaggle

1. Tạo notebook mới ở tài khoản đích, import `02_train_fresh.ipynb` từ gói.
2. Add Input đủ ba dataset ở bảng trên. Chọn **GPU T4 x2**; Internet có thể tắt vì source đã đóng gói trong notebook.
3. Dùng session sạch và Run All / Save & Run All. Không chạy cell đóng gói checkpoint của bản resume trong một fresh run.
4. Notebook chạy smoke riêng ở resolution 1024 trước full run. Nếu OOM hoặc smoke fail, giữ traceback và log để chẩn đoán. Không tự hạ resolution/precision; thay đổi pipeline settings cần ghi nhận thành run mới.
5. `PREPARED_DIR=None` sẽ tự dò manifest. Nếu dò bị giới hạn/ambiguous, đặt `PREPARED_DIR` tới đúng thư mục input chứa `prepared_manifest.json`; không đoán đường dẫn từ máy cũ. Dataset subset version 1 có manifest và `ims/` ngay root.

Notebook đích là Kaggle Linux có **hai CUDA GPU**. Đây không phải bản chạy training trên Mac/CPU hoặc máy chỉ có một GPU. Máy chạy Codex chỉ dùng để điều khiển Kaggle, không cần GPU.

## Chạy bằng Kaggle API

`kernel-metadata.template.json` đã liệt kê ba dataset public và không có model_sources. Copy thành `kernel-metadata.json`, sửa `YOUR_KAGGLE_USERNAME` thành tài khoản đích và dùng slug riêng. Notebook mặc định private ở tài khoản đích để không tự công khai log/checkpoint mới.

`enable_gpu=true` chưa bảo đảm được cấp hai GPU: chọn accelerator T4 x2 bằng giao diện hoặc tùy chọn accelerator mà Kaggle CLI phiên bản máy mới thực sự hỗ trợ; kiểm tra `kaggle kernels push --help` và preflight. Notebook sẽ dừng nếu thấy ít hơn hai CUDA GPU.

Khi tài khoản/quota/accelerator đã sẵn sàng, `kaggle kernels push -p .` sẽ **bắt đầu thực thi** notebook và dùng quota. Theo dõi bằng `kaggle kernels status USER/SLUG`; dùng streaming logs nếu SDK hỗ trợ. Không cần push code để đọc báo cáo hoặc kiểm tra bundle.

## Cấu hình và output

- Fresh notebook: `RESUME_CHECKPOINT=None`, `RUN_ID='b0_sfm100_vgg16_gem_fresh'`, full training bật, tối đa 100 epoch; mandatory smoke chạy trước.
- FP32; image size/eval size 1024; VGG16-GeM; Adam lr=1e-6, weight decay=1e-6; batch 5 tuple; mỗi tuple q+p+5 negative; margin 0,7; seed 42; benchmark mỗi epoch.
- Model feature stack chia GPU0 → GPU1 → GPU0. Same-shape microbatch tối đa 2, mining forward mỗi ảnh unique một lần, worker cap 4; không padding/resize mới.
- Source cirtorch được nhúng, commit `747166e627e514842f5c5be531f4e0f73612cc90`, có source hashes và LICENSE MIT. Không `pip install .` hoặc cài torch==1.0.0/torchvision==0.2.1 từ requirements lịch sử. Dùng cặp torch/torchvision tương thích do Kaggle cung cấp; notebook có API preflight.
- Output chính: `/kaggle/working/b0_sfm100_notebook/runs/<RUN_ID>/` gồm `latest.pt`, `best.pt`, `history.json`, `run_record.json`, timing/evaluation JSON và `training.log`.
- Log toàn session: `/kaggle/working/b0_sfm100_notebook/logs/`. Console được giảm output để tránh Kaggle UI lag; log đầy đủ nằm trên disk.
- Giữ cả `latest.pt` và `best.pt` khi save output. Model selection dùng validation loss, không tự coi latest là best.

## Resume checkpoint cũ

`02_train_sfm100_pipeline.ipynb` giữ luồng resume lịch sử và đường dẫn model `linh2709/seg-ep7/pytorch/default/1/latest.pt`. API tài khoản hiện tại không truy cập được model này trong lần bàn giao, nên **checkpoint đó không có trong gói public**. Không khẳng định checkpoint đã public hoặc resume epoch 7 dùng được ở tài khoản khác.

Để resume, cần có quyền truy cập bản `latest.pt` thực tế và matching `best.pt` nếu epoch latest không phải epoch tốt nhất. Kiểm tra source/config/dataset identity, RNG/optimizer/scheduler state và pipeline identity theo assertions notebook. Khôi phục checkpoint giải nén chỉ khi input thực sự ở dạng folder `data.pkl` + storage; checkpoint `.pt` nguyên vẹn dùng trực tiếp. Đặt đường dẫn/resume/run-id phù hợp; đừng copy đường dẫn máy cũ rồi bỏ qua assertions. Không thể thay checkpoint bằng pretrained feature weights mà coi như đã resume.

## Những gì đã kiểm chứng

- Job CPU preparation hoàn tất trên Kaggle; toàn bộ 28.147 ảnh SfM/benchmark tham chiếu đã qua Pillow RGB decode tại preparation.
- SfM public: 16.692 ảnh nguyên byte, không resize/recompression. Train 15.953 ảnh/100 cluster/34.337 positive pairs; val 739 ảnh/20 cluster/181 pairs. Hai self-pairs train đã bị loại.
- Archive transport 5.844.749.328 bytes, SHA-256 `bebc6d633893d11d7e5d24204de7dcc40327d5735fc620e143b6c53db577a81a`.
- Đã kiểm tra archive hash, SHA-256/byte count từng ảnh, metadata hash, filter/inventory; metadata thật vượt qua chính assertions dữ liệu notebook 02; train/val không trùng CID hoặc cluster.
- Kaggle subset version 1 ready/public, anonymous access thành công; đủ 16.696 file đã giải nén (16.692 ảnh + 4 metadata/manifest) và mọi đường dẫn CID khớp archive. Không có weights/checkpoint/log trong dataset ảnh.
- Test CPU `verify_pipeline.py` đã chạy: descriptors/loss/gradients/multiple Adam updates, mining tuple order + CPU RNG, validation cache/drop_last, crop/gallery order và export byte equality. Đây là model CNN nhỏ + GeM, không phải GPU VGG16.
- File gốc trong Downloads hiện không còn. Đối chiếu dùng baseline đã lưu `notebookc025fac755_optimized.ipynb`, code preparation thực tế và notebook 02. Không có kiểm chứng độc lập bằng cách lấy full metadata và chọn lại toàn bộ historical subset trên máy mới.
- Bản `02_train_fresh.ipynb` chỉ đổi phần khởi tạo checkpoint, resume=None, run-id và input metadata; không đổi loss/runtime/data contract. Đã compile và kiểm tra tĩnh, chưa chạy GPU. Microbatch có thể thay FP32 rounding và thứ hạng gần hòa; không cam kết bitwise reproduction.

Chứng cứ nằm trong `validation-report.json`, `notebook-compatibility.json`, `public-verification.json`, `portable-dependencies.json`. `notebook-compatibility.json` ghi hash của bản resume notebook 02 đã đối chiếu; hash bản fresh được ghi riêng trong `bundle-manifest.json`.

## File trong gói

- `HANDOFF.md`: báo cáo/entry point cho người hoặc Codex mới.
- `02_train_fresh.ipynb`: notebook khuyến nghị cho tài khoản khác.
- `02_train_sfm100_pipeline.ipynb`: bản resume, cần checkpoint riêng.
- `01_prepare_sfm100_cpu.ipynb`: chỉ dùng khi chủ động tái tạo dataset; không cần cho training hiện tại.
- `notebookc025fac755_optimized.ipynb`: baseline để đối chiếu, không phải entry point portable.
- `pipeline_runtime.py`, `build_pipeline_notebooks.py`, `verify_pipeline.py`: source/build và kiểm tra CPU; notebook training đã nhúng runtime nên không cần import các file này trên Kaggle.
- `kernel-metadata.template.json`, các report JSON, `bundle-manifest.json`, `verify_bundle.py`, `LICENSE`.

Không có API key/token, dữ liệu người dùng riêng, checkpoint huấn luyện hoặc log thô trong gói code. License MIT của code giữ trong LICENSE; dữ liệu/weights giữ license `Other` theo dataset nguồn. Không thay đổi quyền ảnh gốc.
