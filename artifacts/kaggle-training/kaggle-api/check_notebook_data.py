"""Check the downloaded real dataset against the saved training notebook contract."""
import hashlib
import json
from pathlib import Path
import pickle
import re
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent

def sources(path):
    return [''.join(c['source']) for c in json.loads(path.read_text())['cells']]

def check():
    baseline = sources(ROOT.parent / 'notebookc025fac755_optimized.ipynb')
    deployed = sources(ROOT / 'prepare-kernel/prepare.ipynb')
    training = sources(ROOT.parent / '02_train_sfm100_pipeline.ipynb')
    # The actual deployed prep retained the saved baseline's selection and cleanup.
    for marker in ['def make_subset(', '# Remove self-pairs from SfM subset-100']:
        reference = next(s for s in baseline if marker in s)
        actual = next(s for s in deployed if marker in s)
        assert reference == actual, marker
    summary = json.loads((ROOT / 'preparation_result.json').read_text())
    with zipfile.ZipFile(ROOT / 'publish-files/prepared_subset100.zip') as archive:
        manifest = json.loads(archive.read('prepared_manifest.json'))
        assert manifest['dataset_identity'] == summary['dataset_identity']
        assert manifest['full_image_decode_preflight'] is True
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for name, digest in manifest['metadata_files'].items():
                assert name in {'retrieval-SfM-120k.pkl', 'needed_files.txt', 'subset_provenance_k100_seed42.txt'}
                data = archive.read(name)
                assert hashlib.sha256(data).hexdigest() == digest
                (temp / name).write_bytes(data)
            db = pickle.loads((temp / 'retrieval-SfM-120k.pkl').read_bytes())
            # Execute the actual notebook's full metadata assertions, stopping
            # before its external weights and sampled filesystem checks.
            contract = next(s for s in training if s.startswith('EXPECTED_SUBSET ='))
            contract = contract.split('def sha256_file(path):')[0]
            scope = dict(SFM_DB=db, np=np, Path=Path, re=re,
                         PROVENANCE_PATH=temp / 'subset_provenance_k100_seed42.txt',
                         FILTER_PATH=temp / 'needed_files.txt')
            exec(compile(contract, '02_train:metadata_contract', 'exec'), scope)
            assert scope['SUBSET_SUMMARY'] == manifest['dataset_identity']['partitions']
            assert scope['expected_filter'] == set(manifest['sfm_images'])
            identity = manifest['dataset_identity']
            for key, name in [('metadata_sha256', 'retrieval-SfM-120k.pkl'),
                              ('filter_sha256', 'needed_files.txt'),
                              ('provenance_sha256', 'subset_provenance_k100_seed42.txt')]:
                assert identity[key] == hashlib.sha256((temp / name).read_bytes()).hexdigest()
            # Order matters to tuple sampling: retain fingerprints of each list.
            ordered = {part: {key: hashlib.sha256(json.dumps([int(x) if isinstance(x, np.integer) else x for x in db[part][key]], separators=(',', ':')).encode()).hexdigest()
                              for key in ['cids', 'cluster', 'qidxs', 'pidxs']}
                       for part in ['train', 'val']}
    report = dict(passed=True, partitions=scope['SUBSET_SUMMARY'], images=16692,
                  checked_notebook_sha256=hashlib.sha256((ROOT.parent / '02_train_sfm100_pipeline.ipynb').read_bytes()).hexdigest(),
                  archive_sha256=summary['archive_sha256'],
                  deployed_subset_and_cleanup_match_saved_baseline=True,
                  exact_training_metadata_assertions_passed=True,
                  cid_layout_and_filter_match=True, ordered_metadata_sha256=ordered,
                  required_external_weights_sha256=identity['weights_sha256'],
                  limitations=['Original Downloads notebook is no longer present; compared with saved baseline and notebook 02.',
                               'Full source metadata was not downloaded for independent re-sampling; deployed selection code matches saved baseline.',
                               'Weights and Oxford/Paris must be attached separately; no GPU training was run.'])
    (ROOT / 'notebook-compatibility.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: real metadata satisfies notebook 02, all CID paths and partition identities match.', flush=True)
    return report

if __name__ == '__main__':
    check()
