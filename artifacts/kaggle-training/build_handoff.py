"""Assemble only reviewed public artifacts; never copy credentials or bulk downloads."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'public-handoff'
DEST.mkdir(exist_ok=True)
FILES = ['01_prepare_sfm100_cpu.ipynb', '02_train_sfm100_pipeline.ipynb',
         'notebookc025fac755_optimized.ipynb', 'pipeline_runtime.py',
         'build_pipeline_notebooks.py', 'verify_pipeline.py']
for filename in FILES:
    shutil.copyfile(ROOT / filename, DEST / filename)
for filename in ['validation-report.json', 'notebook-compatibility.json',
                 'public-verification.json', 'portable-dependencies.json']:
    shutil.copyfile(ROOT / 'kaggle-api' / filename, DEST / filename)
shutil.copyfile(ROOT.parents[1] / 'LICENSE', DEST / 'LICENSE')

notebook = json.loads((ROOT / '02_train_sfm100_pipeline.ipynb').read_text())
fresh = copy.deepcopy(notebook)
fresh['cells'][0]['source'] = '''# Train SfM100 on another Kaggle account — fresh run
Read HANDOFF.md first. Attach the three public input datasets listed there and select T4 x2.
This version starts from Caffe VGG16 weights and requires no historical checkpoint.
It automatically runs a separate smoke before full training; it has not been tested on a real T4 training session.
All source and runtime adapters are embedded. No repository clone or old requirements installation is needed.
'''.splitlines(keepends=True)
fresh['cells'][1] = dict(cell_type='markdown', metadata={}, id='fresh-no-checkpoint',
                        source=['Fresh run: no checkpoint reconstruction. RESUME_CHECKPOINT is None.\n'])
changed = 0
for cell in fresh['cells']:
    if cell['cell_type'] != 'code':
        continue
    text = ''.join(cell['source'])
    if 'RESUME_CHECKPOINT = "/kaggle/working/latest_resume.pt"' in text:
        text = text.replace('RESUME_CHECKPOINT = "/kaggle/working/latest_resume.pt"', 'RESUME_CHECKPOINT = None')
        text = text.replace("RUN_ID = 'b0_sfm100_vgg16_gem'", "RUN_ID = 'b0_sfm100_vgg16_gem_fresh'")
        changed += 1
    cell['source'] = text.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None
assert changed == 1
inputs = ['ndhoang2310/iir-b0-sfm100-prepared',
          'ndhoang2310/iir-b0-benchmarks',
          'ndhoang2310/iir-b0-vgg16-weights']
# Avoid stale author-specific model and full-SfM mounts in the portable notebook.
# kernel-metadata.template.json below is the authoritative attach configuration.
fresh['metadata']['kaggle']['dataSources'] = []
fresh['metadata']['portable_handoff'] = dict(fresh_run=True, inputs=inputs)
(DEST / '02_train_fresh.ipynb').write_text(json.dumps(fresh, ensure_ascii=False, indent=1) + '\n')

kernel = dict(id='YOUR_KAGGLE_USERNAME/iir-b0-sfm100-training',
              title='IIR B0 SfM100 Training', code_file='02_train_fresh.ipynb',
              language='python', kernel_type='notebook', is_private=True,
              enable_gpu=True, enable_tpu=False, enable_internet=False,
              dataset_sources=inputs, competition_sources=[],
              kernel_sources=[], model_sources=[])
(DEST / 'kernel-metadata.template.json').write_text(json.dumps(kernel, indent=2) + '\n')
for path in DEST.glob('*.ipynb'):
    n = json.loads(path.read_text())
    for index, cell in enumerate(n['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'{path.name}:cell{index}', 'exec')

manifest = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(DEST.iterdir()) if p.is_file() and p.name not in {'bundle-manifest.json', 'dataset-metadata.json'}}
(DEST / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Public handoff assembled:', DEST)
