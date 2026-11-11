"""Generate standalone Colab notebooks; no repo clone/pip install is required."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNTIME = (ROOT/'colab_runtime.py').read_text()
EVALUATE = (REPO/'cirtorch/utils/evaluate.py').read_text().replace('np.in1d(', 'np.isin(').rstrip('\n') + '\n'


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        body = source.split('\n', 1)[1] if source.startswith('%%writefile ') else source
        compile(body, 'notebook cell', 'exec')
        result.update(execution_count=None, outputs=[])
    return result


for number in [1, 2]:
    attention = number == 2
    epoch_target = 100 if attention else 50
    session_hours = 14 if attention else 12
    file_stem = f'run{number}_vgg16_gem_' + ('aug_attention' if attention else 'augmentation')
    run_id = file_stem
    if attention:
        run_id = 'run2_attention_bf16_100ep_v1'
    resume_default = 'None' if attention else '"auto"'
    title = f'Run {number}: VGG16 + ' + ('spatial attention + ' if attention else '') + 'GeM / robust augmentation'
    cells = [cell('markdown', f'''# {title}

Notebook độc lập cho **Colab GPU**, không cần clone repo hoặc cài requirements cũ.

**Cách chạy:** chọn GPU → điền `SFM_ZIP` trong cell cấu hình → Run all → cấp quyền mount Drive.
ZIP chứa SfM subset seed42 khoảng 16.692 ảnh và metadata. Giải nén vào `/content`, không đọc JPEG từ Drive trong training.
Log, checkpoint và lịch sử lưu trực tiếp về Drive trong `OUTPUT_ROOT/RUN_ID`.

- Run 1: VGG16–GeM, augmentation chỉ ở training.
- Run 2: cùng cấu hình run 1, thêm spatial attention trước GeM.
- Hai run bắt đầu độc lập từ cùng Caffe pretrained weights; **không resume run 2 từ run 1**.
- Mục tiêu {epoch_target} epoch tổng cộng (tính cả epoch trong checkpoint), 2.000 tuples/epoch, 400 Adam updates/epoch, BF16 (GeM/loss FP32), cạnh dài ≤1024, 5 negatives.
- Tốc độ và VRAM chưa được xác minh trên Colab GPU. Smoke ở 1024 chạy trước training; không tự đổi precision/resolution khi OOM.
- Resume chỉ ở ranh giới epoch: epoch đang chạy khi runtime mất sẽ phải chạy lại.

Bạn chỉ cần ZIP SfM để train. Oxford/Paris ZIP là tùy chọn; không có benchmark thì notebook không công bố mAP.
''')]
    if attention:
        cells.append(cell('markdown', '''## Thời lượng run 2 cập nhật theo số đo thực tế

Bạn đo 20 epoch trong 2 giờ, tương đương khoảng 6 phút/epoch. Mục tiêu 100 epoch dự tính khoảng 10 giờ tổng cộng; nếu đã hoàn tất 20 epoch thì còn khoảng 8 giờ cho 80 epoch tiếp theo. Đây là ước tính theo tốc độ đã quan sát, có thể thay đổi ở epoch/benchmark sau.

Run 2 chưa chạy: mặc định RESUME=None, RUN_ID riêng và khởi tạo từ Caffe pretrained weights, không lấy checkpoint run 1. Sau khi run 2 có checkpoint và cần chạy tiếp, giữ RUN_ID rồi đổi RESUME="auto".

Đặt SESSION_HOURS=14 để có dư địa cho smoke, checkpoint và evaluation; RUN_CU_BUDGET=110 vẫn là cap ước lượng tích lũy cho toàn run, không reset khi resume. Ở 5,4 CU/giờ, 10 giờ tương đương khoảng 54 CU; 14 giờ tương đương 75,6 CU. Số đo 6 phút/epoch đến từ run 1; run 2 có attention nên cần xác nhận lại timing. Code không tăng được giới hạn phiên do Colab áp dụng; nếu runtime bị ngắt thì resume cùng run.
'''))
    cells.append(cell('markdown', f'''## 1. Cấu hình — chỉ sửa cell này trước khi Run all

`SFM_ZIP`: đường dẫn ZIP thật trên Drive, ví dụ `/content/drive/MyDrive/datasets/sfm100.zip`.
ZIP phải có `retrieval-SfM-120k.pkl` và `ims/` cùng thư mục (cho phép một thư mục cha bên ngoài).
Trong `ims/` có thể dùng cây CID chuẩn hoặc ảnh đặt phẳng. Không tự tạo lại subset từ full SfM.

Weights: nếu `WEIGHTS_PATH=None`, tải đúng Caffe VGG16 feature weights từ nguồn gốc, kiểm tra SHA256 và cache trên Drive.
Nếu bạn đã có file `.pth`, điền đường dẫn vào `WEIGHTS_PATH`.

`CU_PER_HOUR`: nhập Usage rate thật trong Colab để notebook ước lượng giới hạn 110 CU cho run này.
Để `None` thì chỉ có giới hạn giờ/session; **không thể tự bảo đảm ngân sách 300 CU**.
`SESSION_HOURS={session_hours}` là giới hạn mềm cho phiên notebook; không thay đổi giới hạn do Colab áp dụng. Notebook dừng trước epoch tiếp theo nếu dự tính không đủ giờ.
Run này đặt `EPOCHS={epoch_target}` tổng cộng, không phải train thêm {epoch_target} epoch.

**Train tiếp:** giữ đúng RUN_ID của thư mục checkpoint trên Drive, `RESUME="auto"`, chỉ tăng EPOCHS và thời lượng operational. Không đổi RUN_ID sang tên có "100ep" nếu muốn tiếp tục run cũ. Giữ source/model/augmentation/precision/LR/microbatch và dataset như checkpoint; không bỏ kiểm tra identity.
Nếu checkpoint hoàn tất epoch 20 và target là 100, controller khôi phục Adam/RNG/LR rồi bắt đầu epoch 21, chạy thêm 80 epoch. Giữ toàn bộ thư mục run, gồm latest.pt, hash và các best_epoch_*.pt.
'''))
    cells.append(cell('code', f'''# === DRIVE INPUTS ===
SFM_ZIP = "/content/drive/MyDrive/IIR_dataset/prepared_subset100.zip"
BENCHMARK_ZIPS = ["/content/drive/MyDrive/IIR_dataset/iir-b0-benchmarks.zip"]  # One ZIP containing both Oxford5k and Paris6k
WEIGHTS_PATH = None  # Optional: exact Drive .pth path; None = download + verified Drive cache

OUTPUT_ROOT = "/content/drive/MyDrive/SEG301/colab_runs"
RUN_ID = "{run_id}"  # Keep this unchanged to resume; new name = independent fresh run
RESUME = {resume_default}  # None = fresh; after interruption keep RUN_ID and switch to "auto"

# Optional exact paths inside the extracted /content tree; normally leave None.
SFM_METADATA_OVERRIDE = None
SFM_IMAGES_OVERRIDE = None

# === SHARED EXPERIMENT CONFIG ===
EPOCHS = {epoch_target}
SEED = 42
IMAGE_SIZE = 1024
QUERY_SIZE = 2000
POOL_SIZE = 20000  # Clipped to actual 15,953 training images
BATCH_SIZE = 5  # tuples; q+p+5 negatives per tuple; summed gradients
LR = 1e-6
WEIGHT_DECAY = 1e-6
MARGIN = 0.7
WORKERS = 4
MICROBATCH = 8  # same-shape images only; measured stable baseline on A100
MAX_BATCH_PIXELS = 8 * 1024 * 1024
EXTRACTION_WINDOW = 8
PREFETCH_FACTOR = 2
TUPLE_FUSION = 1  # 1 / 2 / 5 tuples per backward; Adam still steps after five tuples
CACHE_MODE = "off"  # off / bytes (compressed JPEG) / decoded (resized RGB uint8, before augmentation)
CACHE_GIB = 0.0  # Read-only parent RAM cache cap; apply profiler recommendation for a NEW run
PRECISION = "BF16"  # Confirmed A100 baseline; GeM/loss, weights and Adam stay FP32
ATTENTION = {attention!r}  # Run 1=False, Run 2=True
ATTENTION_LR = 1e-4  # separate small newly initialized spatial gate; backbone LR stays 1e-6

# === OPERATIONAL BUDGET / EVALUATION ===
CU_PER_HOUR = 5.4  # User-reported Colab Usage rate; update if a future runtime differs
RUN_CU_BUDGET = 110.0
SESSION_HOURS = {session_hours}
EVAL_EVERY = 3  # Oxford/Paris only if supplied; clean validation still runs every epoch
FINAL_EVALUATION = True  # best.pt, plus saved descriptors for later CPU search experiments
RUN_TRAINING = True
'''))
    cells.append(cell('markdown', '''## 2. Mount Drive

Giữ torch sẵn có của Colab; không cài torch==1.0.0 hoặc torchvision==0.2.1.
Runtime dựng đúng VGG16 convolutional stack, bỏ classifier và max-pooling cuối; không phụ thuộc torchvision.
Code được trình bày trực tiếp trong các cell bên dưới, không nhúng thành chuỗi escape.
Các cell `%%writefile` chỉ ghi code vào module local; cell import sau đó mới nạp module để chạy.
Chạy tuần tự/Run all; nếu sửa một cell code, chạy lại toàn bộ nhóm từ cell đầu để tránh append lặp.
'''))
    cells.append(cell('code', f'''from pathlib import Path
import hashlib
import json
import re
import sys
from google.colab import drive
drive.mount('/content/drive')

assert SFM_ZIP, "Điền SFM_ZIP ở cell cấu hình trước khi chạy."
assert re.fullmatch(r'[A-Za-z0-9_-]+', RUN_ID), "RUN_ID chỉ gồm chữ, số, _ và -."
assert Path(OUTPUT_ROOT).resolve().is_relative_to(Path('/content/drive').resolve()), "Output phải nằm trên Drive."
assert ATTENTION is {attention!r}, "Notebook này dành riêng cho run {number}."
assert EPOCHS > 0 and BATCH_SIZE == 5 and QUERY_SIZE % BATCH_SIZE == 0
assert IMAGE_SIZE == 1024, "Đổi resolution là thí nghiệm khác; sửa rõ protocol nếu chủ động thay."
assert WORKERS >= 0 and MICROBATCH >= 1 and EVAL_EVERY >= 1
assert MAX_BATCH_PIXELS >= IMAGE_SIZE**2 and EXTRACTION_WINDOW >= 1
assert PREFETCH_FACTOR >= 1 and CACHE_MODE in ["off", "bytes", "decoded"]
assert TUPLE_FUSION in [1, 2, 5]
assert CACHE_MODE == "off" or CACHE_GIB > 0
assert PRECISION in ["FP32", "TF32", "BF16"]
assert CU_PER_HOUR is None or CU_PER_HOUR > 0
assert RUN_CU_BUDGET > 0 and SESSION_HOURS > 0

RUN_DIR = Path(OUTPUT_ROOT) / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CODE = Path('/content/iir_colab_code')
LOCAL_CODE.mkdir(exist_ok=True)
'''))
    sections = [
        ('Tiện ích, log Drive và seed', None),
        ('Đọc ZIP và kiểm tra SfM / weights', 'def extract_zip('),
        ('Model VGG16, GeM và spatial attention', 'class SpatialAttention('),
        ('Augmentation và đọc ảnh', 'class ImageTransform:'),
        ('Trích descriptor, loss và hard-negative mining', 'def forward_images('),
        ('Training, validation và optimizer', 'class TupleImages('),
        ('Lưu checkpoint và resume', 'def cpu_tree('),
        ('Benchmark và xuất descriptor', 'def discover_benchmarks('),
        ('Smoke test', 'def _run_smoke_impl('),
        ('Controller chạy epoch và ngân sách', 'def train_run('),
    ]
    offsets = [0 if marker is None else RUNTIME.index(marker) for _, marker in sections] + [len(RUNTIME)]
    for index, (label, _) in enumerate(sections):
        cells.append(cell('markdown', f'### 2.{index+1}. {label}\n\nCode Python dưới đây ghi vào module local; chưa khởi chạy training.\n'))
        command = '%%writefile ' + ('-a ' if index else '') + '/content/iir_colab_code/colab_runtime.py\n'
        cells.append(cell('code', command + RUNTIME[offsets[index]:offsets[index+1]]))
    cells.append(cell('markdown', '### 2.11. Evaluator mAP\n\nProtocol legacy Oxford/Paris; dùng `np.isin` để tương thích NumPy mới.\n'))
    cells.append(cell('code', '%%writefile /content/iir_colab_code/legacy_evaluate.py\n' + EVALUATE))
    cells.append(cell('markdown', '### 2.12. Import runtime và mở log\n\nNạp các module vừa ghi. Training chỉ bắt đầu ở mục 4.\n'))
    cells.append(cell('code', f'''RUNTIME_SOURCE = (LOCAL_CODE/'colab_runtime.py').read_text()
# IPython/Colab may append a terminal newline to a %%writefile cell.
# Canonicalize only terminal newlines; edits inside the evaluator still fail the hash.
EVALUATE_SOURCE = (LOCAL_CODE/'legacy_evaluate.py').read_text().rstrip(chr(10))
RUNTIME_HASH = hashlib.sha256(RUNTIME_SOURCE.encode()).hexdigest()
EVALUATE_HASH = hashlib.sha256(EVALUATE_SOURCE.encode()).hexdigest()
# Exact known transport variant: Colab/editor removed trailing whitespace on each line.
# Preserve the original evaluator identity so existing checkpoints can resume.
if EVALUATE_HASH == 'b2619397085e7dfcbdc78ff31abbc02a2c5ab76014c93eef84f20f3ce251dc6d':
    EVALUATE_HASH = 'ced1d110dd19f3f887e75f18ac9120aaff8900f5d6b52dc8c45c66c823141fd0'
assert RUNTIME_HASH == {hashlib.sha256(RUNTIME.encode()).hexdigest()!r}, "Code đã sửa hoặc append lặp: chạy lại toàn bộ nhóm cell code từ đầu; nếu chủ động sửa, cập nhật hash và dùng RUN_ID mới."
assert EVALUATE_HASH == {hashlib.sha256(EVALUATE.rstrip(chr(10)).encode()).hexdigest()!r}, "Evaluator đã đổi; kiểm tra lại code (đã bỏ qua newline cuối file)."
if str(LOCAL_CODE) not in sys.path:
    sys.path.insert(0, str(LOCAL_CODE))
import importlib
import colab_runtime as rt
import legacy_evaluate
importlib.reload(rt)
importlib.reload(legacy_evaluate)
SESSION_LOG = rt.start_logging(RUN_DIR)

import torch
if not torch.cuda.is_available():
    raise RuntimeError('Chọn Runtime > Change runtime type > GPU rồi chạy lại.')
print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)
print('Drive output:', RUN_DIR)
if CU_PER_HOUR is None:
    print('CU_PER_HOUR chưa nhập: chưa thể ước lượng ngân sách CU; chỉ áp dụng giới hạn giờ/session.')
'''))
    cells.append(cell('markdown', '''## 3. Giải nén ZIP vào local và kiểm tra dataset

Copy ZIP Drive → local, kiểm tra CRC khi giải nén, từ chối path traversal/symlink và chỉ tái sử dụng extraction đã hoàn tất.
Kiểm tra đủ mọi đường dẫn SfM, counts, cặp positive, train/val không trùng cluster/CID; decode mẫu 128 ảnh.
Nếu có prepared manifest: kiểm tra metadata hashes và 64 image hashes. Ảnh ngoài mẫu vẫn có thể lỗi khi loader đọc sau đó.
Benchmark tùy chọn cần `oxford5k/gnd_oxford5k.pkl` + `oxford5k/jpg/`, tương tự `paris6k`.
'''))
    cells.append(cell('code', '''CACHE_ROOT = Path('/content/iir_colab_data')
SFM_ROOT = rt.extract_zip(SFM_ZIP, CACHE_ROOT)
BENCH_ROOTS = [rt.extract_zip(path, CACHE_ROOT) for path in BENCHMARK_ZIPS]
PARTITIONS, DATA_IDENTITY = rt.load_sfm(SFM_ROOT, SFM_METADATA_OVERRIDE, SFM_IMAGES_OVERRIDE)
WEIGHTS = rt.resolve_weights(WEIGHTS_PATH, Path(OUTPUT_ROOT)/'_weights_cache')
BENCHMARKS = rt.discover_benchmarks([SFM_ROOT] + BENCH_ROOTS)
print('Benchmarks:', list(BENCHMARKS) or 'None — chỉ train/validation; mAP đánh giá sau')

CFG = dict(attention=ATTENTION, attention_lr=ATTENTION_LR, epochs=EPOCHS, seed=SEED,
           image_size=IMAGE_SIZE, query_size=QUERY_SIZE, pool_size=POOL_SIZE,
           batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, margin=MARGIN,
           workers=WORKERS, microbatch=MICROBATCH, max_batch_pixels=MAX_BATCH_PIXELS,
           extraction_window=EXTRACTION_WINDOW,prefetch_factor=PREFETCH_FACTOR,cache_mode=CACHE_MODE,
           cache_gib=CACHE_GIB,tuple_fusion=TUPLE_FUSION,
           augmentation_version='mild_photo_blur_jpeg_erase_v1', precision=PRECISION)
DATA_IDENTITY['benchmark_metadata'] = {name: value['metadata_sha256'] for name, value in BENCHMARKS.items()}
IDENTITY = rt.make_identity(CFG, DATA_IDENTITY, RUNTIME_HASH)
IDENTITY['run_id'] = RUN_ID
IDENTITY['evaluator_sha256'] = EVALUATE_HASH
# Benchmark inputs are evaluation-only; adding them later must not invalidate model resume.
IDENTITY['data'].pop('benchmark_metadata', None)
OPERATIONAL = dict(resume=RESUME, scratch='/content/iir_checkpoint_staging',
                   cu_per_hour=CU_PER_HOUR, run_cu_budget=RUN_CU_BUDGET,
                   session_hours=SESSION_HOURS, eval_every=EVAL_EVERY,
                   final_evaluation=FINAL_EVALUATION)

# Refuse to overwrite provenance for an existing run before validating its identity.
identity_path = RUN_DIR/'experiment_identity.json'
if identity_path.exists() and json.loads(identity_path.read_text()) != IDENTITY:
    raise RuntimeError('RUN_ID đã có config/source/data khác. Đặt RUN_ID mới; không ghi đè run cũ.')
rt.atomic_json(identity_path, IDENTITY)
(RUN_DIR/'colab_runtime.py').write_text(RUNTIME_SOURCE)
(RUN_DIR/'legacy_evaluate.py').write_text(EVALUATE_SOURCE)
rt.atomic_json(RUN_DIR/'input_sources.json', dict(sfm_zip=SFM_ZIP, benchmark_zips=BENCHMARK_ZIPS,
               weights=str(WEIGHTS), benchmarks=list(BENCHMARKS)))
print(json.dumps(CFG, indent=2))
'''))
    cells.append(cell('markdown', '''## 4. Smoke → training → checkpoint về Drive

- Smoke riêng ở resolution 1024: một batch có mining/backward/update, validation và checkpoint round-trip.
- Model full training được khởi tạo lại từ pretrained, không dùng trọng số smoke.
- Mining/validation dùng ảnh sạch. Augmentation chỉ dùng trong tuple training.
- Mỗi epoch: 2.000 tuple → 400 Adam updates → clean validation 180 tuple → lưu checkpoint → benchmark nếu đến lịch.
- GeM p dùng LR gấp 10 backbone. Run 2 có gate spatial-only 7×7, 99 tham số, LR riêng 1e-4; gate khởi tạo identity bằng `2*sigmoid`.
- Best chọn theo validation loss. Epoch 7 được giữ riêng để đối chiếu run cũ.
- Trước epoch tiếp theo, dùng timing epoch trước để dự tính còn đủ giờ/CU không. Đây là ước lượng, không phải đồng hồ billing Colab.
- Gián đoạn giữa epoch: giữ checkpoint epoch hoàn tất trước đó; chạy lại cùng RUN_ID với `RESUME='auto'`.
- Không copy checkpoint run 1 sang run 2. Khi kéo dài training chỉ tăng EPOCHS; thay config học/augmentation/source phải dùng RUN_ID mới.

Checkpoint được serialize ở local, copy sang Drive bằng file `.pending`, kiểm tra SHA256 rồi rename.
Drive mount không thay thế một hệ lưu trữ giao dịch; giữ `previous_latest.pt` và các best version để phục hồi nếu bị ngắt lúc ghi.
'''))
    cells.append(cell('code', '''if RUN_TRAINING:
    try:
        TRAINING_RESULT = rt.train_run(CFG, PARTITIONS, WEIGHTS, RUN_DIR, IDENTITY,
                                       OPERATIONAL, BENCHMARKS, legacy_evaluate.compute_map)
    except BaseException:
        import traceback
        traceback.print_exc()  # Also persist smoke/resume failures in the Drive session log.
        sys.stdout.flush()
        sys.stderr.flush()
        raise
else:
    print('RUN_TRAINING=False: chuẩn bị xong, chưa train.')
'''))
    cells.append(cell('markdown', '''## 5. Kiểm tra output / đánh giá riêng sau training

`logs/session_*.log`: log trực tiếp trên Drive. `latest.pt`: resume; `previous_latest.pt`: dự phòng.
`best.pt` và `best_epoch_*.pt`: model chọn bởi validation loss; giữ cả versioned best khi resume.
`history.json`, `timing_latest.json`, `training_result.json`, `run_record.json`: loss/timing/config.
Nếu có Oxford/Paris, final evaluation xuất descriptor `.npz` về Drive để thử search bằng CPU sau.

Bạn có thể thêm benchmark ZIP sau này, đặt RUN_TRAINING=False và chạy cell đánh giá riêng bên dưới.
Cell này không cập nhật model và không chạy smoke. Việc trích descriptor vẫn nên dùng GPU.
'''))
    cells.append(cell('code', '''# Chỉ bật khi muốn đánh giá riêng best.pt; mặc định không chạy lặp sau training.
EVALUATE_BEST_ONLY = False
if EVALUATE_BEST_ONLY:
    if not BENCHMARKS:
        raise RuntimeError('Điền BENCHMARK_ZIPS và chạy lại bước dữ liệu trước.')
    checkpoint = rt.load_checkpoint(RUN_DIR/'best.pt', IDENTITY)
    eval_model = rt.make_model(CFG, WEIGHTS, torch.device('cuda:0'))
    try:
        eval_model.load_state_dict(checkpoint['model'], strict=True)
        metrics = rt.benchmark(eval_model, BENCHMARKS, CFG, RUN_DIR/'evaluation_best.json',
                               legacy_evaluate.compute_map, export=True)
        rt.atomic_json(RUN_DIR/'evaluation_best_identity.json', dict(epoch=checkpoint['epoch'],
                       checkpoint_sha256=rt.sha256(RUN_DIR/'best.pt'), metrics=metrics, identity=IDENTITY))
    finally:
        del eval_model, checkpoint
        import gc
        gc.collect()
        torch.cuda.empty_cache()

print('Output trên Drive:', RUN_DIR)
for name in ['latest.pt', 'best.pt', 'history.json', 'training_result.json', 'timing_latest.json']:
    path = RUN_DIR/name
    print(name, f'{path.stat().st_size/1024**2:.1f} MiB' if path.exists() else 'chưa có')
'''))
    for i, item in enumerate(cells):
        item['id'] = f'colab-run{number}-{i:02d}'
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(accelerator='GPU', colab=dict(name=f'{run_id}.ipynb', provenance=[]),
                                  kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python', version='3.12'),
                                  runtime_sha256=hashlib.sha256(RUNTIME.encode()).hexdigest()))
    target = ROOT/f'{number:02d}_{file_stem}.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    print(target.name)
