"""Build a readable standalone Colab GPU tuning notebook from shared runtime cells."""
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'build_notebooks.py'))
base = json.loads((ROOT/'01_run1_vgg16_gem_augmentation.ipynb').read_text())
PROFILE = (ROOT/'profile_runtime.py').read_text()


def cell(kind,text):
    result = dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
    if kind == 'code':
        compile(text.split('\n',1)[1] if text.startswith('%%writefile ') else text,'profile cell','exec')
        result.update(execution_count=None,outputs=[])
    return result


cells = [cell('markdown', '''# Benchmark A100 v3 — aggressive capacity sweep

Notebook này **đo trên GPU thật**, không chạy full training và không đụng vào checkpoint run 1/run 2.
Chọn A100 GPU, Run all, cấp quyền Drive. Đường dẫn ZIP đã điền theo cấu hình của bạn.

Quy trình: lấy mẫu ảnh thật → mining tuple thật → FP32 reference → so precision → microbatch → workers/window → mini epoch xác nhận cả hai model → đo ghi checkpoint lên Drive → xuất cấu hình.

- Thử FP32 strict, TF32 và BF16 autocast. GeM/loss và trọng số/Adam vẫn FP32.
- Giữ nguyên resolution 1024, batch 5 tuple, summed contrastive loss; không tăng batch để lấp VRAM.
- Cùng ảnh/cặp/augmentation trong phép kiểm tra số học; thử cả model không/có attention.
- Đo wall time có CUDA synchronize, loader wait, peak allocated/reserved VRAM và GPU utilization nếu nvidia-smi có sẵn.
- Cấu hình bị OOM, mất headroom hoặc lệch quá ngưỡng sẽ không được đề xuất.
- Mốc mặc định 90 phút / cap mềm 10 CU, kiểm tra giữa trial; một trial đang chạy có thể vượt mốc.
- Đo RAM hệ thống, RAM còn lại, USS/RSS của parent+workers. RSS cộng nhiều process có thể đếm trùng RAM shared; dùng USS và RAM hệ thống để đánh giá.
- Thử workers tới 32, prefetch tới 16, cache JPEG bytes 8 GiB hoặc resized RGB tới 64 GiB. Giữ ≥20%/4 GiB RAM dự phòng và kiểm tra /dev/shm trước trial.
- Thử gộp 1/2/5 tuple cho backward, microbatch tới 128, extraction window tới 256; Adam vẫn cập nhật sau đúng 5 tuple, loss vẫn sum.
- Chạy các cặp microbatch/window lớn ngay đầu sweep với cache tắt để mức cache/queue RAM không che mất khả năng GPU. Sau đó thử cache lớn với workers 2/4/8.
- Warmup 2 bước, đo 10 bước, extraction lặp 3 lần. Hai finalist được đo lại 3 lần trước khi đề xuất.
- Đây là khảo sát theo từng giai đoạn, không vét cạn mọi tổ hợp và không chứng minh hội tụ/mAP dài hạn.

Log/CSV/JSON và các model probe dùng một lần lưu riêng ở `SEG301/colab_runs/profile_a100_<timestamp>/`.
''')]
cells.append(cell('markdown', '''## 1. Input và giới hạn benchmark

ZIP SfM: 16.692 ảnh + metadata subset đã chuẩn bị. ZIP benchmark gộp cả Oxford và Paris.
Benchmarks chỉ cần đọc metadata để ước lượng chi phí evaluation; notebook không chạy toàn bộ benchmark mAP.
Giới hạn thời gian/CU bên dưới tính từ khi bắt đầu sweep, chưa gồm copy/giải nén ZIP và GPU idle.
'''))
cells.append(cell('code', '''import time
SFM_ZIP = "/content/drive/MyDrive/IIR_dataset/prepared_subset100.zip"
BENCHMARK_ZIPS = ["/content/drive/MyDrive/IIR_dataset/iir-b0-benchmarks.zip"]
WEIGHTS_PATH = None
OUTPUT_ROOT = "/content/drive/MyDrive/SEG301/colab_runs"
RUN_ID = "profile_a100_v3_" + time.strftime("%Y%m%d_%H%M%S")
SFM_METADATA_OVERRIDE = None
SFM_IMAGES_OVERRIDE = None

# Shared training protocol — same as run 1 / run 2
EPOCHS = 100  # Planning upper bound only; this notebook never trains 100 epochs
SEED = 42
IMAGE_SIZE = 1024
QUERY_SIZE = 2000
POOL_SIZE = 20000
BATCH_SIZE = 5
LR = 1e-6
WEIGHT_DECAY = 1e-6
MARGIN = 0.7
ATTENTION_LR = 1e-4
ATTENTION = False  # Placeholder for setup; profiler independently tests BOTH False and True
WORKERS = 4
MICROBATCH = 8
MAX_BATCH_PIXELS = 8 * 1024 * 1024
EXTRACTION_WINDOW = 8
PRECISION = "BF16"  # Start at the previous A100 winner; strict FP32 remains numerical reference
PREFETCH_FACTOR = 2
CACHE_MODE = "off"
CACHE_GIB = 0.0
TUPLE_FUSION = 1
EVAL_EVERY = 3
CU_PER_HOUR = 5.4  # Update to actual Colab Usage rate
RUN_CU_BUDGET = 110.0  # For final per-run epoch projection, not benchmark budget
SESSION_HOURS = 12

# Profiling: staged search, not every possible combination
PROFILE_MAX_MINUTES = 90
PROFILE_CU_BUDGET = 10.0
EXTRACTION_SAMPLE_IMAGES = 256
HEADER_SCAN_IMAGES = 2048
MINING_SAMPLE_POOL = 1024
WARMUP_STEPS = 2
MEASURED_STEPS = 10
EXTRACTION_REPEATS = 3
PRECISION_CANDIDATES = ["BF16", "TF32"]
MICROBATCH_CANDIDATES = [8, 16, 32, 64, 128]
WORKER_CANDIDATES = [4, 8, 12, 16, 24, 32]
WINDOW_CANDIDATES = [8, 16, 32, 64, 128, 256]
PREFETCH_CANDIDATES = [2, 4, 8, 16]
CACHE_CANDIDATES = [("off", 0.0), ("bytes", 8.0), ("decoded", 8.0), ("decoded", 24.0), ("decoded", 40.0), ("decoded", 64.0)]
TUPLE_FUSION_CANDIDATES = [1, 2, 5]
# Early GPU capacity tests, before spending time filling RAM caches
CAPACITY_MICROBATCHES = [16, 32, 64, 128]
CAPACITY_FUSIONS = [2, 5]
# Joint tests avoid a greedy high-worker choice excluding an otherwise fast large cache
CACHE_JOINT_GIB = [24.0, 40.0, 64.0]
CACHE_JOINT_WORKERS = [2, 4, 8]
FINALIST_REPEATS = 3
MIN_RAM_FREE_FRACTION = 0.20
MAX_VRAM_FRACTION = 0.90  # Aggressive probes may use ~36 GiB on A100 40 GiB
CONFIRMATION_QUERIES = 50
CONFIRMATION_POOL = 1024
CONFIRMATION_RESERVE_MINUTES = 12

# Conservative local numerical guards; not a convergence/mAP acceptance test
MAX_LOSS_RELATIVE_ERROR = 0.05
MIN_GRADIENT_COSINE = 0.98
MIN_DESCRIPTOR_COSINE = 0.995
RUN_PROFILE = True
'''))
# Reuse the readable shared runtime and evaluator cells, including Drive mount/import.
started = False
for source_cell in base['cells']:
    text = ''.join(source_cell['source'])
    if text.startswith('## 2. Mount Drive'):
        started = True
    if text.startswith('## 3. Giải nén ZIP'):
        break
    if started:
        copied = dict(source_cell)
        if copied['cell_type'] == 'code':
            text = text.replace('"Notebook này dành riêng cho run 1."',
                                '"Profiler sẽ tự thử cả hai biến thể attention."')
            copied['source'] = text.splitlines(keepends=True)
        cells.append(copied)
cells.append(cell('markdown', '''## 3. Chuẩn bị dữ liệu thật và weights local

Giải nén ZIP về `/content`; không đọc JPEG từ Drive trong timing. Tải/cache đúng Caffe weights nếu chưa có.
Copy weights một lần về local để các lần tạo model không bị ảnh hưởng bởi Drive I/O.
'''))
cells.append(cell('code', '''PREPARATION_STARTED = time.monotonic()
CACHE_ROOT = Path('/content/iir_colab_data')
SFM_ROOT = rt.extract_zip(SFM_ZIP, CACHE_ROOT)
BENCH_ROOTS = [rt.extract_zip(path, CACHE_ROOT) for path in BENCHMARK_ZIPS]
PARTITIONS, DATA_IDENTITY = rt.load_sfm(SFM_ROOT, SFM_METADATA_OVERRIDE, SFM_IMAGES_OVERRIDE)
DRIVE_WEIGHTS = rt.resolve_weights(WEIGHTS_PATH, Path(OUTPUT_ROOT)/'_weights_cache')
import shutil
WEIGHTS = Path('/content/iir_profile_weights.pth')
shutil.copyfile(DRIVE_WEIGHTS, WEIGHTS)
assert rt.sha256(WEIGHTS) == rt.WEIGHTS_HASH
BENCHMARKS = rt.discover_benchmarks([SFM_ROOT] + BENCH_ROOTS)

BASE_CFG = dict(attention=False,attention_lr=ATTENTION_LR,epochs=EPOCHS,seed=SEED,
    image_size=IMAGE_SIZE,query_size=QUERY_SIZE,pool_size=POOL_SIZE,batch_size=BATCH_SIZE,
    lr=LR,weight_decay=WEIGHT_DECAY,margin=MARGIN,workers=WORKERS,microbatch=MICROBATCH,
    max_batch_pixels=MAX_BATCH_PIXELS,extraction_window=EXTRACTION_WINDOW,
    prefetch_factor=PREFETCH_FACTOR,cache_mode=CACHE_MODE,cache_gib=CACHE_GIB,tuple_fusion=TUPLE_FUSION,
    augmentation_version='mild_photo_blur_jpeg_erase_v1',precision=PRECISION)
CONTROLS = dict(max_minutes=PROFILE_MAX_MINUTES,profile_cu_budget=PROFILE_CU_BUDGET,
    cu_per_hour=CU_PER_HOUR,training_cu_per_run=RUN_CU_BUDGET,eval_every=EVAL_EVERY,
    extraction_images=EXTRACTION_SAMPLE_IMAGES,header_scan=HEADER_SCAN_IMAGES,mining_sample_pool=MINING_SAMPLE_POOL,
    warmup_steps=WARMUP_STEPS,measured_steps=MEASURED_STEPS,extraction_repeats=EXTRACTION_REPEATS,
    microbatches=MICROBATCH_CANDIDATES,workers=WORKER_CANDIDATES,windows=WINDOW_CANDIDATES,
    precisions=PRECISION_CANDIDATES,prefetch_factors=PREFETCH_CANDIDATES,caches=CACHE_CANDIDATES,
    tuple_fusions=TUPLE_FUSION_CANDIDATES,finalists=2,finalist_repeats=FINALIST_REPEATS,
    capacity_microbatches=CAPACITY_MICROBATCHES,capacity_fusions=CAPACITY_FUSIONS,
    cache_joint_gib=CACHE_JOINT_GIB,cache_joint_workers=CACHE_JOINT_WORKERS,
    min_ram_free_fraction=MIN_RAM_FREE_FRACTION,
    max_vram_fraction=MAX_VRAM_FRACTION,confirmation_queries=CONFIRMATION_QUERIES,
    confirmation_pool=CONFIRMATION_POOL,confirmation_reserve_minutes=CONFIRMATION_RESERVE_MINUTES)
LIMITS = dict(max_loss_relative_error=MAX_LOSS_RELATIVE_ERROR,
              min_gradient_cosine=MIN_GRADIENT_COSINE,min_descriptor_cosine=MIN_DESCRIPTOR_COSINE)
assert CU_PER_HOUR > 0 and PROFILE_MAX_MINUTES > 0 and PROFILE_CU_BUDGET > 0
assert MEASURED_STEPS >= 2 and WARMUP_STEPS >= 1 and EXTRACTION_REPEATS >= 2
assert 0 < MAX_VRAM_FRACTION < 1
assert CONFIRMATION_QUERIES % BATCH_SIZE == 0 and CONFIRMATION_QUERIES >= BATCH_SIZE
rt.atomic_json(RUN_DIR/'profile_inputs.json', dict(data_identity=DATA_IDENTITY,sfm_zip=SFM_ZIP,
    benchmark_zips=BENCHMARK_ZIPS,preparation_seconds=time.monotonic()-PREPARATION_STARTED,
    runtime_sha256=RUNTIME_HASH,evaluator_sha256=EVALUATE_HASH))
(RUN_DIR/'colab_runtime.py').write_text(RUNTIME_SOURCE)
(RUN_DIR/'legacy_evaluate.py').write_text(EVALUATE_SOURCE)
print('Dữ liệu sẵn sàng; output riêng:', RUN_DIR)
'''))
sections = [('GPU monitor và tiện ích',None),
            ('Chọn mẫu và kiểm tra descriptor/loss/gradient','def select_paths('),
            ('Chọn cấu hình chung và ước lượng epoch','def tune_signature('),
            ('Đo extraction, training và stress VRAM','def trial_impl('),
            ('Mini epoch xác nhận và cấu hình đề xuất','def confirm_impl('),
            ('Lưu báo cáo và điều phối sweep','def write_reports(')]
offsets = [0 if marker is None else PROFILE.index(marker) for _,marker in sections] + [len(PROFILE)]
cells.append(cell('markdown', '## 4. Code benchmark\n\nCác cell chỉ định nghĩa module đo; chưa khởi chạy sweep.\n'))
for i,(label,_) in enumerate(sections):
    cells.append(cell('markdown',f'### 4.{i+1}. {label}\n'))
    cells.append(cell('code','%%writefile '+('-a ' if i else '')+
                      '/content/iir_colab_code/profile_runtime.py\n'+PROFILE[offsets[i]:offsets[i+1]]))
cells.append(cell('markdown', '''## 5. Chạy sweep

FP32 reference không update model. Các trial dùng model riêng và thực hiện vài Adam updates thật để đo đầy đủ.
Không có state nào được dùng để khởi tạo run 1/2. Stress dùng ảnh lớn nhất trong 512 ảnh được đọc header,
lặp thành batch 5 tuple cùng shape; fusion=5 giữ graph tới 35 ảnh để stress VRAM thực sự. Đây không phải training data mới.

Thời gian rank/sort đo ở kích thước full 15.953×2.000 và 739×181, dùng descriptor ngẫu nhiên;
chất lượng numerical/mining/training đo bằng ảnh và tuple thật. Thời gian cả epoch là **ngoại suy**, chưa đo full epoch.
'''))
cells.append(cell('code', f'''PROFILE_SOURCE = (LOCAL_CODE/'profile_runtime.py').read_text().rstrip('\\n')
PROFILE_HASH = hashlib.sha256(PROFILE_SOURCE.encode()).hexdigest()
assert PROFILE_HASH == {hashlib.sha256(PROFILE.rstrip(chr(10)).encode()).hexdigest()!r}, "Code profiler đã đổi hoặc append lặp; chạy lại nhóm 4 từ cell đầu."
import profile_runtime as pr
importlib.reload(pr)
(RUN_DIR/'profile_runtime.py').write_text(PROFILE_SOURCE)
if RUN_PROFILE:
    try:
        RECOMMENDATION = pr.run_profile(BASE_CFG,PARTITIONS,WEIGHTS,BENCHMARKS,RUN_DIR,CONTROLS,LIMITS)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
else:
    print('RUN_PROFILE=False; chưa chạy benchmark.')
'''))
cells.append(cell('markdown', '''## 6. Đọc kết quả và áp dụng

`trials.csv/json`: throughput, loader wait, GPU utilization, VRAM, host RAM/USS/RSS, cache coverage và sai lệch số học của từng cấu hình/model.
`recommendation.json`: đề xuất chung và dự tính giờ/CU/epoch; `confirmations.json`: mini epoch/full validation/Drive write.
Chỉ tạo `recommended_config.py` nếu cả hai model đã qua bước xác nhận và ngân sách dự tính đủ ít nhất một epoch.

Nếu trạng thái PROVISIONAL/NO_SAFE, không áp dụng tự động; xem lỗi hoặc tăng thời gian benchmark.
Sau khi có kết quả, copy cấu hình đề xuất (gồm cache/prefetch/tuple fusion) vào **cả hai notebook training mới** và dùng RUN_ID mới.
Đổi precision/source không thể resume checkpoint của phiên bản notebook cũ. Không cần dùng model probe để train.

Tốc độ mẫu có thể lạc quan do cache/prefetch; projection có margin 25% nhưng không bảo đảm 12 epoch vừa 110 CU.
Numerical guards chỉ kiểm tra batch cố định, không chứng minh BF16/TF32 giữ nguyên mAP sau nhiều epoch.
'''))
cells.append(cell('code', '''report = RUN_DIR/'recommendation.json'
if report.exists():
    recommendation = json.loads(report.read_text())
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
try:
    import pandas as pd
    if (RUN_DIR/'trials.csv').exists():
        table = pd.read_csv(RUN_DIR/'trials.csv')
        display(table.sort_values(['status','estimated_epoch_seconds']))
except ImportError:
    print('Xem trials.csv trên Drive để so sánh các cấu hình.')
snippet = RUN_DIR/'recommended_config.py'
if snippet.exists():
    print('Cấu hình chung đề xuất:')
    print(snippet.read_text())
print('Log và báo cáo:', RUN_DIR)
'''))
for index,item in enumerate(cells):
    item['id'] = f'a100-profile-{index:03d}'
base['cells'] = cells
base['metadata']['colab']['name'] = '00_profile_a100_training.ipynb'
base['metadata']['profile_source_sha256'] = hashlib.sha256(PROFILE.encode()).hexdigest()
target = ROOT/'00_profile_a100_training.ipynb'
target.write_text(json.dumps(base,ensure_ascii=False,indent=1)+'\n')
print(target.name)
