"""Build the two self-contained Kaggle notebooks from the supplied v1 artifact."""
import copy, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
base = json.loads((ROOT/'notebookc025fac755_optimized.ipynb').read_text())
def cell(kind, text):
    value = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code': value.update(execution_count=None, outputs=[])
    return value
def original(i):return copy.deepcopy(base['cells'][i+1])
def text(c):return ''.join(c['source'])
def settext(c,s):c['source']=s.splitlines(keepends=True)
def write(name, cells, gpu):
    notebook = copy.deepcopy(base)
    notebook['cells'] = cells
    notebook['metadata']['kaggle']['isGpuEnabled'] = gpu
    if not gpu:
        notebook['metadata']['kaggle'].pop('accelerator',None)
        notebook['metadata']['kaggle']['dataSources'] = [item for item in notebook['metadata']['kaggle'].get('dataSources',[]) if item['sourceType']=='datasetVersion']
    notebook['metadata']['operational_revision']['version']='pipeline_v2'
    for i,c in enumerate(cells):
        c['id'] = f'pipeline-{i:03d}'
        if c['cell_type']=='code':compile(text(c),f'{name}:cell{i}','exec')
    (ROOT/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print(name)

prep = [cell('markdown', '''# 1. Prepare SfM subset once — CPU
Run this notebook once with the original SfM + Oxford + Paris datasets and Caffe weights attached. No checkpoint or GPU is needed. It selects the same subset, removes self-pairs, validates images, then copies the exact original SfM image bytes into a portable subset folder. Images are not recompressed or resized. The final manifest is written only after export succeeds.

Save the output folder `b0_sfm100_prepared` as a Kaggle Dataset, then attach that dataset to the training notebook. Keep Oxford/Paris and Caffe weights attached separately. This preparation run is intentionally expensive; training no longer performs it.
''')]
for i in range(3,18):
    c=original(i)
    if i==3:
        s=text(c);a=s.index('if not torch.cuda.is_available():');b=s.index("print('Free working disk GB:'",a)
        s=s[:a]+"ENVIRONMENT = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, pillow=PIL.__version__, preparation='CPU')\n"+s[b:]
        settext(c,s)
    if i==17:
        prepared_source = text(c).split('# ------------------------------------------------------------\n# 3. Approved sources')[0]
        a = prepared_source.index("stage('Checking SfM image paths')")
        b = prepared_source.index('# ------------------------------------------------------------\n# 2. Initialize benchmarks', a)
        prepared_source = prepared_source[:a] + """from concurrent.futures import ThreadPoolExecutor
import stat as file_stat
PREPARATION_IO_WORKERS = 16

def _stat_sfm_image(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return path, None
    return path, info.st_size if file_stat.S_ISREG(info.st_mode) else None

stage('Checking SfM paths with bounded parallel I/O')
with ThreadPoolExecutor(max_workers=PREPARATION_IO_WORKERS) as executor:
    stats = list(executor.map(_stat_sfm_image, referenced))
missing_images = [str(path) for path,size in stats if size is None]
if missing_images:
    raise FileNotFoundError(f'{len(missing_images)} SfM images missing; first paths: {missing_images[:10]}')
SFM_IMAGE_BYTES = {str(path): size for path,size in stats}
print('SfM image paths checked:',len(SFM_IMAGE_BYTES))

""" + prepared_source[b:]
        a = prepared_source.index('    for image_id in all_ids:')
        b = prepared_source.index('    # Invalid ranking indices',a)
        prepared_source = prepared_source[:a] + """    for image_id in all_ids:
        assert isinstance(image_id,str) and Path(image_id).name == image_id
    benchmark_paths = [jpg / f'{image_id}.jpg' for image_id in sorted(all_ids)]
    with ThreadPoolExecutor(max_workers=PREPARATION_IO_WORKERS) as executor:
        present = list(executor.map(Path.is_file, benchmark_paths))
    missing = [str(path) for path,ok in zip(benchmark_paths,present) if not ok]
    if missing:
        raise FileNotFoundError(f'{name}: missing benchmark files: {missing[:10]}')

""" + prepared_source[b:]
        a = prepared_source.index("stage(f'Decoding {len(image_paths)} images before training')")
        prepared_source = prepared_source[:a] + """stage(f'Fully decoding {len(image_paths)} images with bounded parallel I/O')
def _decode_preparation_image(path):
    try:
        with Image.open(path) as image:
            image.convert('RGB').load()
    except Exception as error:
        raise RuntimeError(f'Image cannot be decoded: {path}: {error}') from error
    return None

with ThreadPoolExecutor(max_workers=PREPARATION_IO_WORKERS) as executor:
    for index,_ in enumerate(executor.map(_decode_preparation_image,image_paths),1):
        if index % 1000 == 0 or index == len(image_paths):
            print(f'Image decode preflight: {index}/{len(image_paths)}',flush=True)
"""
        settext(c,prepared_source)
    prep.append(c)
prep.append(cell('code','''import shutil
from concurrent.futures import ThreadPoolExecutor

EXPORT = WORKING / 'b0_sfm100_prepared'
if EXPORT.exists():
    raise RuntimeError('Export folder already exists. Use a fresh session or choose another EXPORT; no files are overwritten.')
EXPORT.mkdir()
needed_bytes = sum(SFM_IMAGE_BYTES[str(path)] for path in referenced)
free_bytes = shutil.disk_usage(EXPORT).free
if needed_bytes + 2 * 1024**3 > free_bytes:
    raise RuntimeError(f'Insufficient export space: need image bytes={needed_bytes} plus 2 GiB reserve; free={free_bytes}')
for source in [SFM_PKL, FILTER_PATH, PROVENANCE_PATH]:
    shutil.copyfile(source, EXPORT / source.name)

def copy_verified_image(source):
    relative = source.relative_to(SFM_IMAGES)
    destination = EXPORT / 'ims' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open('rb') as inp, destination.open('xb') as output:
        for chunk in iter(lambda: inp.read(1024*1024), b''):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return relative.as_posix(), dict(bytes=size, sha256=digest.hexdigest())

stage(f'Exporting {len(referenced)} original SfM images ({needed_bytes/1e9:.2f} GB)')
with ThreadPoolExecutor(max_workers=8) as executor:
    exported = dict(executor.map(copy_verified_image, referenced))
manifest = dict(format='b0_prepared_subset_v1', dataset_identity=DATASET_IDENTITY,
                metadata_files={p.name: sha256_file(EXPORT/p.name) for p in [SFM_PKL,FILTER_PATH,PROVENANCE_PATH]},
                sfm_images=exported, full_image_decode_preflight=True,
                note='Original image bytes; no resize/recompression. Training samples hashes and decodes; lazy loading detects other failures.')
manifest_path = EXPORT / 'prepared_manifest.json'
manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2))
print('PREPARATION COMPLETE:', EXPORT)
print('Publish this folder as a Kaggle Dataset, then attach it to the training notebook.')
_session_log_stop()
'''))
write('01_prepare_sfm100_cpu.ipynb',prep,False)

train=[cell('markdown','''# 2. Train from the prepared SfM Dataset — T4 ×2
Requires the output of `01_prepare_sfm100_cpu.ipynb` attached as a Kaggle Dataset, plus Oxford/Paris, Caffe weights and the existing resume checkpoint input. Set `PREPARED_DIR` in the input-discovery cell if automatic discovery is ambiguous. No subset creation or image copying runs here.

Changes: bounded same-shape microbatches (default 2, no padding), one extraction per unique mining image, one bulk GPU-to-CPU transfer for hard-negative selection, validation descriptor reuse while weights are fixed, bounded workers, sampled preflight and stage timings. Full image validation runs in the CPU preparation notebook. Missing/corrupt unsampled images can still fail later during loading.

FP32, image preprocessing, tuple draw order, distinct-cluster negative rule, summed per-tuple gradients, optimizer-step schedule and benchmark frequency remain. Batching can change floating-point rounding and near-tie rankings: this is a new pipeline run, not a claim of bitwise reproduction. Historical checkpoints are accepted subject to existing identity checks. New checkpoints record pipeline source/settings and require those to match on later resumes. GPU memory and speed require Kaggle smoke testing; no automatic precision/resolution fallback occurs.
''')]
for i in range(1,30):
    c=original(i)
    if c['cell_type']=='markdown':
        # Replace stale combined-notebook descriptions.
        if i in (6,13,18,20,22,28):continue
        if i==4:settext(c,'## Pinned source and explicit runtime adapters\nSource files and hashes remain intact; runtime replacements are recorded separately.\n')
    if i==7:
        s=text(c);pos=s.index("stage('Locating metadata/weights")
        setup='''# Exact attached prepared folder, or None for bounded discovery.
PREPARED_DIR = None
if PREPARED_DIR is None:
    matches, truncated = discover_metadata(INPUT_ROOT, ['prepared_manifest.json'])
    if truncated or len(matches['prepared_manifest.json']) != 1:
        raise RuntimeError('Attach the prepared SfM Dataset and set PREPARED_DIR to its exact folder.')
    PREPARED_DIR = matches['prepared_manifest.json'][0].parent
PREPARED_DIR = input_path(PREPARED_DIR)
PREPARED_MANIFEST = json.loads((PREPARED_DIR / 'prepared_manifest.json').read_text())
assert PREPARED_MANIFEST['format'] == 'b0_prepared_subset_v1'
assert PREPARED_MANIFEST['full_image_decode_preflight'] is True
for filename, expected_hash in PREPARED_MANIFEST['metadata_files'].items():
    assert filename in {'retrieval-SfM-120k.pkl', 'needed_files.txt', 'subset_provenance_k100_seed42.txt'}
    actual_hash = hashlib.sha256((PREPARED_DIR / filename).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f'Prepared metadata hash mismatch: {filename}'
PATH_OVERRIDES['sfm_metadata'] = str(PREPARED_DIR / 'retrieval-SfM-120k.pkl')
PATH_OVERRIDES['sfm_images'] = str(PREPARED_DIR / 'ims')
'''
        s=s[:pos]+setup+'\n'+s[pos:]
        s=s.replace("print('Full backup is ignored; whitening metadata is unused.')", "print('Prepared subset loaded; no full-data sampling or image export in training.')")
        settext(c,s)
    if i in (9,10,11,12):continue
    if i==14:
        settext(c,text(c)+'''\nassert DATASET_IDENTITY == PREPARED_MANIFEST['dataset_identity'], 'Prepared data/weights identity differs'
assert expected_filter == set(PREPARED_MANIFEST['sfm_images']), 'Prepared image inventory differs'
# Sample integrity without a repeated 16k-file scan/decode. Use an independent RNG.
for relative in random.Random(42).sample(sorted(expected_filter), min(64,len(expected_filter))):
    path = PREPARED_DIR / 'ims' / relative
    assert sha256_file(path) == PREPARED_MANIFEST['sfm_images'][relative]['sha256'], str(path)
''')
    if i==17:
        s=text(c)
        s=s.replace('for index, path in enumerate(referenced, 1):',"sampled_references = random.Random(42).sample(referenced, min(64, len(referenced)))\nfor index, path in enumerate(sampled_references, 1):")
        s=s.replace("print('SfM image paths checked:', len(referenced))", "print('SfM paths sampled:', len(sampled_references), 'of', len(referenced))")
        s=s.replace('for image_id in all_ids:', 'for image_id in random.Random(42).sample(sorted(all_ids), min(64, len(all_ids))):')
        s=s.replace("stage(f'Decoding {len(image_paths)} images before training')", "image_paths = random.Random(42).sample(image_paths, min(128, len(image_paths)))\nstage(f'Sample-decoding {len(image_paths)} images; full SfM validation was done in preparation')")
        settext(c,s)
    if i==21:
        s=text(c)
        runtime=(ROOT/'pipeline_runtime.py').read_text()
        runtime_hash=hashlib.sha256(runtime.encode()).hexdigest()
        pos=s.index('def preview_mining_pool')
        s=s[:pos]+runtime+f"\nPIPELINE_IDENTITY = dict(settings=PIPELINE_SETTINGS, source_sha256='{runtime_hash}')\n"+s[pos:]
        s=s.replace('super().create_epoch_tuples(net)  # Exact embedded upstream algorithm.', 'pipeline_mine(self, net)  # Same tuple draws; unique descriptor extraction.')
        s=s.replace("    original_train.args = SimpleNamespace(update_every=cfg['update_every'], print_freq=10)", "    original_train.args = SimpleNamespace(update_every=cfg['update_every'], print_freq=10)\n    original_train.train = pipeline_train\n    original_train.validate = pipeline_validate\n    imageretrievalnet.extract_vectors = pipeline_extract_vectors")
        s += "\nimageretrievalnet.extract_vectors = pipeline_extract_vectors\nContrastiveLoss = PipelineContrastiveLoss\n"
        settext(c,s)
    if i==23:
        settext(c, text(c).replace('AUTO_SMOKE = RUN_FULL_TRAINING and RESUME_CHECKPOINT is None', 'AUTO_SMOKE = RUN_FULL_TRAINING  # New engine gets a smoke even on legacy resume.').replace('Fresh full run: automatic separate smoke at the full image size.', 'Pipeline engine: automatic separate smoke at the full image size (including resume).'))
    if i==25:
        s=text(c)
        s=s.replace("    assert state.get('format')", "    if 'pipeline_engine' in state:\n        assert state['pipeline_engine'] == PIPELINE_IDENTITY, 'Pipeline changed since checkpoint'\n    assert state.get('format')")
        s=s.replace("environment=ENVIRONMENT, history=history)", "environment=ENVIRONMENT, history=history, pipeline_engine=PIPELINE_IDENTITY)")
        s=s.replace("operational_revision='low_output_v1'", "pipeline_engine=PIPELINE_IDENTITY, prepared_dataset=str(PREPARED_DIR), operational_revision='pipeline_v2'")
        settext(c,s)
    train.append(c)
write('02_train_sfm100_pipeline.ipynb',train,True)
