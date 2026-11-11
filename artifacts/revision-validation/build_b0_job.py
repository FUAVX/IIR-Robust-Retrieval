"""Prepare, but do not submit, the bounded Kaggle B0 evaluation job."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    folder = ROOT/'jobs/b0-clean-epoch12-v1'
    folder.mkdir(parents=True, exist_ok=True)
    source = '''import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import hashlib, json, sys, traceback
from pathlib import Path
inputs = Path('/kaggle/input')
weights = list(inputs.rglob('b0_epoch12.pth'))
assert len(weights) == 1, weights
code = weights[0].parent
manifest = json.loads((code/'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((code/name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(code))
from b0_clean_eval import main
output = Path('/kaggle/working/b0_clean')
try:
    main(code, inputs, output)
except Exception as error:
    output.mkdir(parents=True, exist_ok=True)
    (output/'status.json').write_text(json.dumps(dict(status='FAILED', error=str(error))))
    raise
'''
    compile(source, 'evaluate.py', 'exec')
    (folder/'evaluate.py').write_text(source)
    metadata = dict(id='ndhoang2310/iir-revision-b0-clean-epoch12-v1',
        title='IIR Revision B0 Clean Epoch12 v1', code_file='evaluate.py', language='python',
        kernel_type='script', is_private=True, enable_gpu=True, enable_tpu=False,
        enable_internet=False, dataset_sources=['ndhoang2310/iir-revision-b0-inputs-v1',
        'ndhoang2310/iir-b0-benchmarks'], kernel_sources=[], competition_sources=[], model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(folder)
    return folder


if __name__ == '__main__':
    build()
