# Hai run training trên Colab từ ZIP trên Google Drive

- `00_profile_a100_training.ipynb`: chạy trước để đo và chọn cấu hình vận hành chung trên A100.
- `00b_retest_a100_training.ipynb`: retest tập trung sau log v3, đã sửa cache tồn dư; không lặp lại sweep cache lớn/128 OOM.
- `01_run1_vgg16_gem_augmentation.ipynb`: VGG16–GeM + augmentation.
- `02_run2_vgg16_gem_aug_attention.ipynb`: cùng training protocol + spatial attention trước GeM.

Mở từng notebook bằng **Colab → File → Upload notebook**, chọn GPU, điền `SFM_ZIP` ở cell cấu hình rồi Run all. Không cần clone repo, Kaggle credentials hoặc cài requirements cũ. Hai notebook độc lập và không phụ thuộc nhau.

Cấu hình người dùng đã lưu cho cả hai notebook:

```python
SFM_ZIP = "/content/drive/MyDrive/IIR_dataset/prepared_subset100.zip"
BENCHMARK_ZIPS = ["/content/drive/MyDrive/IIR_dataset/iir-b0-benchmarks.zip"]
OUTPUT_ROOT = "/content/drive/MyDrive/SEG301/colab_runs"
CU_PER_HOUR = 5.4
RUN_CU_BUDGET = 110.0
SESSION_HOURS = 12
EPOCHS = 50
PRECISION = "BF16"
MICROBATCH = 8
MAX_BATCH_PIXELS = 8 * 1024 * 1024
WORKERS = 4
EXTRACTION_WINDOW = 8
PREFETCH_FACTOR = 2
CACHE_MODE = "off"
CACHE_GIB = 0.0
TUPLE_FUSION = 1
```

ZIP benchmark là **một file gộp Oxford5k và Paris6k**. Notebook giải nén một lần rồi tự dò hai file metadata trong cây thư mục. Run 1 giữ `RUN_ID="run1_vgg16_gem_augmentation"`, `ATTENTION=False`; run 2 dùng `RUN_ID="run2_attention_bf16_100ep_v1"`, `ATTENTION=True`. Các tham số chung còn lại giống cấu hình người dùng đã cung cấp. Nếu Usage rate thay đổi ở runtime khác, cập nhật `CU_PER_HOUR` theo số thực tế.

**Cập nhật run 2:** `EPOCHS=100`, `SESSION_HOURS=14`, `RESUME=None` để chạy mới; run 1 đang ở epoch 20 và vẫn giữ mục tiêu 50 epoch/12 giờ. Run 2 bắt đầu từ pretrained weights, không từ checkpoint run 1. Sau khi run 2 bị ngắt và đã có checkpoint, giữ RUN_ID rồi đổi `RESUME="auto"`. Theo số đo người dùng 20 epoch/2 giờ, 100 epoch khoảng 10 giờ tổng cộng (~54 CU), hoặc còn khoảng 8 giờ nếu checkpoint đã hoàn tất epoch 20. Giới hạn 14 giờ chừa overhead và không mở rộng giới hạn runtime của Colab; cap CU toàn run vẫn 110.

Để train tiếp checkpoint, giữ **đúng RUN_ID bạn đã dùng thực tế trên Colab** (nếu đã đổi do trùng tên thì không thay về tên mặc định trong file tải mới). Chỉ tăng target EPOCHS và thời lượng operational; dùng notebook/source/config/data cũ của checkpoint. Controller khôi phục model, Adam, RNG và LR theo epoch. Target 100 với checkpoint epoch 20 chạy từ epoch 21 tới 100, không chạy thêm 100 epoch. Không dùng checkpoint run 1 để resume model attention của run 2. Giữ cả latest.pt, hash và các best_epoch_*.pt trong cùng thư mục Drive.

Code model/data/training/checkpoint được trình bày thành các cell Python đọc được. Các cell `%%writefile` ghi code vào module local, rồi cell import nạp module; không còn chuỗi runtime dài với `\\n`/escape. Khi sửa hoặc chạy lại nhóm này, chạy từ cell `%%writefile` đầu tiên (ghi mới) đến hết nhóm (append), tránh append một cell nhiều lần. Source hash kiểm tra nhóm code có được ghép đúng; logic training không đổi so với bản đóng gói trước.

## Input

ZIP do bạn cung cấp chứa subset đã chuẩn bị: 15.953 ảnh train, 739 ảnh val, 100/20 cluster, 34.337/181 positive pairs. Hai train self-pairs phải đã được loại. Notebook không tự chọn lại subset hoặc đổi metadata full thành subset.

Ví dụ cấu trúc ZIP (được phép có thư mục cha bên ngoài):

```text
retrieval-SfM-120k.pkl
prepared_manifest.json                 # có thì kiểm tra, không bắt buộc
needed_files.txt                       # nếu có manifest tham chiếu thì phải đủ
subset_provenance_k100_seed42.txt
ims/
  <CID suffix folders>/<CID>           # hoặc tất cả CID đặt phẳng trong ims/
```

Điền đường dẫn sau khi mount, ví dụ `SFM_ZIP = "/content/drive/MyDrive/datasets/sfm100.zip"`. Notebook copy ZIP về `/content`, giải nén có kiểm tra CRC/path và dùng ảnh local trong training. Local extraction không còn sau khi runtime bị reset; chạy lại sẽ giải nén lại. Nếu file `.pkl`/`ims` đặt khác vị trí, dùng hai override ở cell cấu hình.

`WEIGHTS_PATH=None` tải Caffe VGG16 feature weights từ nguồn gốc, bắt buộc đúng SHA256, cache ở `OUTPUT_ROOT/_weights_cache`. Có thể điền đường dẫn `.pth` trên Drive để không tải mạng. Đây là weights pretrained ImageNet, không phải checkpoint epoch 7.

Oxford/Paris không bắt buộc để training. Nếu có, điền thêm `BENCHMARK_ZIPS`; các archive cần chứa `oxford5k/gnd_oxford5k.pkl` + `oxford5k/jpg/`, tương tự `paris6k`. Metadata benchmark phải đáng tin cậy vì loader dùng pickle. Không có benchmark thì chỉ báo train/val loss, không công bố mAP.

## Cấu hình thí nghiệm

| | Run 1 | Run 2 |
|---|---|---|
| Backbone/khởi tạo | VGG16 / Caffe pretrained | Giống run 1, khởi tạo độc lập |
| Pooling/output | GeM p=3 học được / L2 / 512-D | Giống run 1 |
| Augmentation | Brightness/contrast/color, blur, JPEG, erasing nhẹ | Giống run 1 |
| Attention | Không | Spatial-only gate 7×7, 99 tham số |
| Backbone/GeM LR | 1e-6 / 1e-5 | Giống run 1 |
| Attention LR | Không dùng | 1e-4 |
| Loss | Contrastive, margin 0.7, q+p+5 | Giống run 1 |
| Epoch mục tiêu đã chốt | 50 | 100, có thể tăng target khi resume |
| Giới hạn mềm/session | 12 giờ | 14 giờ |

Ảnh cạnh dài tối đa 1024, đã chốt BF16, microbatch 8, workers 4, window 8, prefetch 2, cache off, fusion 1 cho cả hai run mới. BF16 chỉ autocast backbone/attention; GeM, normalization, contrastive loss, weights và Adam vẫn FP32. Augmentation chỉ ở gradient training; mining/validation/evaluation dùng ảnh sạch. Không crop mạnh hoặc resize vuông. Run 2 gate khởi tạo identity (`2*sigmoid(0)=1`), khác gate CBAM sigmoid nguyên bản; không phải tái hiện toàn bộ CBAM. Giả thuyết cải thiện retrieval chưa được chứng minh.

## Notebook đánh giá/tối ưu A100

### Sửa sau log v3 thực tế

Log v3 có 58 PASS, 4 OOM và 60 SKIPPED_HOST_PLAN. GPU đã lên 30–31,6 GiB, microbatch 128 OOM. Cache RGB chứa đủ 16.692 ảnh chiếm khoảng 29,95 GiB. Phát hiện cache RAM của trial trước còn tồn tại khi resource plan kiểm tra trial cache-off; điều này có thể chặn sai các bước fusion/window tiếp theo. Profiler mới giải phóng cache không tương thích **trước** resource plan và monitor; cấu hình có lần kiểm tra gần nhất thất bại cũng không được giữ lại chỉ dựa vào PASS cũ.

Chạy `00b_retest_a100_training.ipynb` thay vì lặp lại toàn bộ v3. Retest giữ workers=4, prefetch=2, cache off khi so fusion 1/2/5 × microbatch/window 16/16, 32/32, 64/64. Sau đó đo prefetch/window và finalist 3 lần. Giới hạn mềm 45 phút/6 CU. Log in HOST_PLAN kèm lý do/ước lượng queue/RAM và HOST_RAM kèm số đo. Không thử lại microbatch 128 đã OOM hoặc preload cache ~30 GiB. Epoch planning vẫn là ngoại suy; cần xác nhận trên epoch training đầy đủ.

Upload `00_profile_a100_training.ipynb` lên Colab, chọn A100 và Run all. Đường dẫn ZIP/CU rate 5.4 đã điền. Output nằm riêng trong `OUTPUT_ROOT/profile_a100_<timestamp>`, không đọc/ghi model training chính.

- **V3 aggressive:** mặc định sweep tối đa mềm 90 phút / 10 CU, kiểm tra giữa các trial; chưa tính copy/giải nén hoặc GPU idle. Trial đang chạy có thể vượt mốc. Bắt đầu từ winner A100 đã đo: BF16, microbatch 8, workers 4, window 8; vẫn dùng strict FP32 làm numerical reference.
- So BF16/TF32; workers tới 32 (tối đa 2× CPU affinity), prefetch tới 16; RAM cache off/JPEG bytes 8 GiB/RGB 8–24–40–64 GiB; tuple fusion 1/2/5 với microbatch tới 128; extraction window tới 256. Chọn theo giai đoạn, không vét cạn mọi tổ hợp.
- Thử GPU capacity sớm: fusion 2/5 × microbatch 16/32/64/128 cùng cửa sổ tương ứng, cache tắt. Với microbatch 64/128 dùng một worker/prefetch 1 để queue RAM không chặn mất phép thử GPU; các mức nhỏ dùng tối đa bốn workers/prefetch 2. Sau đó thử cache lớn với workers 2/4/8 để tránh lựa chọn workers cao ở bước trước loại mất các tổ hợp cache có lợi.
- Các trial dùng ảnh thật cùng sample và tuple mined từ FP32 pretrained. Kiểm tra descriptor/loss/gradient so FP32, cả hai biến thể model. Guard mặc định: loss relative error ≤5%, gradient cosine ≥0.98, descriptor cosine ≥0.995. Chỉ là sanity checks cục bộ, không chứng minh hội tụ hay mAP tương đương.
- Cùng optimizer step với notebook training: objective là tổng loss của 5 tuple, Adam vẫn step một lần sau 5 tuple. Fusion gộp 2/5 tuple cho backward chung, chỉ cho model không có batchnorm/dropout. Numerical guard kiểm tra cả thay đổi nhóm backward này.
- Warmup 2 bước, đo 10 bước; extraction mẫu 256 ảnh, lặp 3 lần. Hai finalist được đo lại 3 lần, dùng median cho mỗi model; chưa đủ repeats/confirmation thì chỉ xuất PROVISIONAL, không tạo config để áp dụng.
- CUDA synchronize ở ranh giới đo. Đo throughput extraction/training, loader wait, peak allocated/reserved VRAM và nvidia-smi utilization. Đo RAM hệ thống còn lại và used, process-tree USS/RSS; RSS có thể đếm trùng các trang cache shared. USS lấy mẫu mỗi 5 giây để giảm overhead; RAM/RSS mỗi khoảng một giây.
- Preflight queue kiểm tra RAM và `/dev/shm` trước cấu hình workers/prefetch lớn; giữ ≥20%/4 GiB host RAM. Cache bounded theo cap và RAM còn lại; bytes/RGB uint8 chỉ đọc, fork-share giữa workers, không cache ảnh đã augmentation. RGB được resize bằng đúng Pillow path cũ và copy trước transform. Nếu không đủ RAM, bỏ qua cấu hình đó; không cố lấp RAM.
- Stress forward/backward bằng ảnh lớn nhất trong mẫu 2.048 header, lặp thành batch 5 tuple cùng shape; fusion=5 giữ graph tới 35 ảnh trước backward. V3 cho phép tối đa 90% VRAM (khoảng 36 GiB nếu GPU có 40 GiB); cấu hình OOM hoặc vượt cap bị loại. Không bảo đảm mọi ảnh chưa sample sẽ vừa VRAM. Microbatch lớn hơn 35 chủ yếu phục vụ extraction; effective training batch vẫn 5 tuple.
- Thời gian sort/rank dùng descriptor ngẫu nhiên ở đúng kích thước full pool; epoch time được ngoại suy từ throughput mẫu. Candidate tốt nhất được xác nhận bằng mini epoch 50 tuple/pool 1.024, full validation 739 ảnh và ghi checkpoint probe lên Drive.
- Chỉ xuất `recommended_config.py` khi candidate qua cả hai model, đủ 3 phép đo lại và xác nhận. Copy các biến `PRECISION`, `MICROBATCH`, `MAX_BATCH_PIXELS`, `WORKERS`, `EXTRACTION_WINDOW`, `PREFETCH_FACTOR`, `CACHE_MODE`, `CACHE_GIB`, `TUPLE_FUSION`, `EPOCHS` vào cả hai notebook training mới. `BATCH_SIZE`, image size và loss không đổi.
- Giới hạn planning epoch của profiler là 100 thay vì 12; không có full training 100 epoch trong notebook này. Số epoch đề xuất vẫn phụ thuộc CU/epoch đo được, margin và ngân sách 110 CU/run; có thể cần resume qua nhiều session.
- `trials.csv/json`, `selected_candidate.json`, `confirmations.json`, `recommendation.json`, `profile_usage.json` và log đều trên Drive. Nếu hết giờ trước confirmation, kết quả là PROVISIONAL, không tự áp dụng.

Source runtime đã thêm precision modes; checkpoint từ notebook phiên bản cũ bị chặn resume do source hash khác. Nếu đang chạy run cũ, tiếp tục bằng notebook/code cũ đã lưu; chỉ dùng runtime/cấu hình mới cho run mới. Không dùng `probe_run*.pt` làm checkpoint training.

Mining chọn 2.000 positive pairs và lấy 5 hard negatives từ 5 cluster khác nhau; pool clip ở 15.953 ảnh. Mỗi ảnh unique chỉ trích một lần trong mining. Trích ảnh cùng shape theo microbatch cấu hình, không padding. Mặc định backward từng tuple; có thể chọn fusion 2/5 sau benchmark. Cộng loss/gradient của 5 tuple rồi Adam.step (400 updates/epoch). Không dùng loss mean để thay thang gradient.

Validation dùng toàn bộ pool 739 ảnh, lấy 180/181 tuple theo drop_last. Pair order dùng seed validation cố định; negatives vẫn được chọn lại theo model. Reuse descriptors vừa mining để không forward ảnh lần hai. Đây là protocol mới dùng chung cho hai run, không cam kết RNG/bitwise ngang notebook Kaggle cũ. Backbone bỏ classifier và maxpool cuối đúng source cirtorch; dựng trực tiếp bằng torch để không cấp bộ nhớ classifier hoặc phụ thuộc torchvision cũ.

Best chọn bằng clean validation contrastive loss, không chọn bằng test mAP. Benchmark nếu có chạy mỗi 3 epoch; đánh giá best và xuất descriptor ở cuối. Không dùng benchmark test để chọn hyperparameters.

## Lưu Drive, resume và ngân sách

Mặc định output: `/content/drive/MyDrive/SEG301/colab_runs/<RUN_ID>/`.

- `logs/session_*.log`: stdout/stderr trực tiếp trên Drive.
- `latest.pt` + `.sha256`: trạng thái epoch hoàn tất, gồm model/Adam/RNG/config/source/data identity và history.
- `previous_latest.pt`: bản latest trước đó để phục hồi nếu publication bị ngắt.
- `best.pt`: bản tiện dụng của best; các `best_epoch_*.pt` là bản versioned, **giữ chúng khi resume**.
- `epoch_007.pt`: checkpoint mốc 7 epoch để đối chiếu.
- `history.json`, `timing_latest.json`, `training_result.json`, `run_record.json`, `usage_total.json`: theo dõi.
- `smoke/`: model riêng để kiểm tra forward/backward/checkpoint, không dùng làm full training weights.
- Khi có benchmark: `evaluation_*.json` và `*_best_descriptors.npz` (D×N), dùng cho thử search CPU sau.

Checkpoint serialize local, copy bằng `.pending`, kiểm tra hash rồi rename. Bản best versioned được commit trước latest; khi resume, alias best.pt có thể được phục hồi theo latest. Drive mount không bảo đảm ghi giao dịch khi VM mất đột ngột. Nếu hash latest hỏng, đặt `RESUME` đến `previous_latest.pt` trong chính RUN_ID đó. Epoch đang chạy dở phải chạy lại, không resume giữa batch.

Mỗi run mặc định cap 110 CU **ước lượng**, với `CU_PER_HOUR=5.4` theo thông tin người dùng. Nếu để None, không có giới hạn CU tự động. `SESSION_HOURS=12`; controller dự đoán epoch kế tiếp từ timing epoch trước để dừng giữa hai epoch. Không tự tắt một epoch đang chạy và không đọc được billing Colab. Phần mount/copy/preflight ngoài controller và thời gian GPU idle không nằm trong ước lượng; cần theo dõi số dư thực tế. Mục tiêu 50 epoch; thời gian và CU thực tế vẫn cần theo dõi trên full training.

Giữ cùng epoch target/config cho hai run. Thay augmentation, microbatch, precision, model, LR hoặc source phải dùng RUN_ID mới. Chỉ epoch target có thể tăng trên checkpoint hiện có. Không resume run 2 từ run 1. Có thể thêm benchmark sau bằng `RUN_TRAINING=False` và `EVALUATE_BEST_ONLY=True`.

## Kiểm tra local

```bash
python3 artifacts/colab-training/build_notebooks.py
python3 artifacts/colab-training/build_profile_notebook.py
python3 artifacts/colab-training/build_retest_notebook.py
python3 artifacts/colab-training/verify_colab.py
python3 artifacts/colab-training/verify_profile.py
```

Verifier cần torch/torchvision/NumPy/Pillow; kiểm tra compile, embedded source, VGG topology/state keys/forward, gate identity, contrastive loss+gradients, mixed-shape extraction, ZIP, dữ liệu giả có đúng 16.692 file/counts, và training/resume của cả hai biến thể với CNN nhỏ trên CPU. Không tải weights hoặc dữ liệu thật.

Hai run VGG16 thật trên Colab đã hoàn tất 100 epoch; history, checkpoint identities và kết quả evaluation được dẫn từ `../../paper/ieee/data/models_manifest.json` và `../README.md`. Cấu hình 50 epoch ở trên là mặc định khởi tạo notebook; các lượt lịch sử đã tăng target lên 100 khi resume. Notebook bắt buộc smoke ở 1024 trước full training, dừng khi có OOM/lỗi và không tự hạ resolution/precision.

Runtime và notebook có nguồn gốc từ code/pipeline của repo; MIT license gốc được giữ tại `LICENSE` trong thư mục này. Spatial attention tham khảo [CBAM, ECCV 2018](https://arxiv.org/abs/1807.06521).
