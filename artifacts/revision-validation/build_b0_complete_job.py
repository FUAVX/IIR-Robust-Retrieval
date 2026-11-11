"""Build the B0 completion job; deliberately does not submit or monitor it."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def build():
    manifest = json.loads((HERE/'results/b0_input_manifest.json').read_text())
    recovery = json.loads((HERE/'results/b0_clean/recovery_manifest.json').read_text())
    refs = []
    for filename in ['all_posttraining_results.csv', 'sparse_diffusion_summary.csv']:
        with (ROOT/'paper/ieee/data'/filename).open() as f:
            for row in csv.DictReader(f):
                if row['model'] not in ['run1_aug_epoch100', 'run2_aug_attention_epoch100']:
                    continue
                if row['method'] not in ['cosine', 'AQE', 'alpha_QE', 'gated_QE', 'diffusion_mutual_top20']:
                    continue
                refs.append(dict(model=row['model'], epoch=100, selection='final_epoch_100',
                    dataset=row['dataset'], method=row['method'], clean_map=float(row['clean_map']),
                    corrupted_map=float(row['corrupted_map']), worst_map=float(row['worst_map']),
                    record_origin='historical_epoch100_fixed_protocol'))
    assert len(refs) == 20 and len({(r['model'], r['dataset'], r['method']) for r in refs}) == 20
    source = (HERE/'run_b0_complete.py').read_text()
    source = source.replace('EXPECTED_INPUT_MANIFEST = {}', 'EXPECTED_INPUT_MANIFEST = '+repr(manifest), 1)
    source = source.replace("OXFORD_CACHE_SHA256 = ''", 'OXFORD_CACHE_SHA256 = '+repr(recovery['source_descriptor_sha256']), 1)
    source = source.replace('REFERENCE_ROWS = []', 'REFERENCE_ROWS = '+repr(refs), 1)
    compile(source, 'run_b0_complete.py', 'exec')
    folder = HERE/'jobs/b0-complete-epoch12-v2'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'run_b0_complete.py').write_text(source)
    metadata = dict(id='ndhoang2310/iir-b0-complete-epoch12-v2',
        title='IIR B0 Epoch12 Complete Clean and Robustness v2', code_file='run_b0_complete.py',
        language='python', kernel_type='script', is_private=True, enable_gpu=True,
        enable_tpu=False, enable_internet=False,
        dataset_sources=['ndhoang2310/iir-revision-b0-inputs-v1', 'ndhoang2310/iir-b0-benchmarks',
                         'ndhoang2310/iir-revision-b0-oxford-numerical-v1'],
        kernel_sources=[], competition_sources=[], model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(folder)
    return folder


if __name__ == '__main__':
    build()
