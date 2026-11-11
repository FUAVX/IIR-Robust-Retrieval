"""Verify every completed B0 ranking, then package the numerical release.

No model training, GPU inference or authenticated API calls are performed here.
"""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'artifacts/kaggle-posttraining'))
import runtime as R


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def package():
    base = ROOT/'artifacts/revision-validation'
    compact = base/'outputs/b0-complete-v2/b0_complete_v2'
    raw = base/'outputs/b0-v2-raw'
    files = {}
    for source in [compact, *sorted(raw.iterdir())]:
        if (source/'b0_complete_v2').is_dir():
            source = source/'b0_complete_v2'
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in {'.csv', '.json', '.npz'}:
                continue
            name = str(path.relative_to(source))
            if name in files:
                assert R.digest(files[name]) == R.digest(path), name
            else:
                files[name] = path
    summary = read(compact/'summary.csv')
    views = read(compact/'views.csv')
    per = {(r['dataset'], r['method'], r['condition'], r['query_id']): r
           for r in read(compact/'per_query.csv')}
    assert len(summary) == 10 and len(views) == 310 and len(per) == 17050
    assert json.loads((compact/'status.json').read_text())['status'] == 'COMPLETE'
    groundtruth = {}
    for dataset in ['oxford5k', 'paris6k']:
        path = ROOT/f'data/test/{dataset}/gnd_{dataset}.pkl'
        assert R.digest(path) == R.P.GND_HASHES[dataset]
        with path.open('rb') as f:
            gt = groundtruth[dataset] = pickle.load(f)
        path = files[f'desc_b0_epoch12_{dataset}_robust.npz']
        with np.load(path, allow_pickle=False) as f:
            assert str(f['checkpoint_sha256']) == '8a56bad2e1d585c68de42ff841f061e5469f76792750d8b4993e500294f19f27'
            np.testing.assert_array_equal(f['gallery_ids'], gt['imlist'])
            np.testing.assert_array_equal(f['query_ids'], gt['qimlist'])
            np.testing.assert_array_equal(f['conditions'], [c['name'] for c in R.P.conditions()])
            assert f['gallery'].shape == (512, len(gt['imlist'])) and f['queries'].shape == (512, 1705)
            for key in ['gallery', 'queries']:
                assert np.isfinite(f[key]).all()
                np.testing.assert_allclose(np.linalg.norm(f[key], axis=0), 1, atol=2e-5)
    max_error = 0.
    for index, row in enumerate(views, 1):
        ds, method, condition = row['dataset'], row['method'], row['condition']
        name = f'ranks_{ds}_{condition}_{method}.npz'
        gt = groundtruth[ds]; n = len(gt['imlist'])
        with np.load(files[name], allow_pickle=False) as f:
            ranks = f['ranks']
            assert ranks.shape == (n, 55)
            np.testing.assert_array_equal(f['gallery_ids'], gt['imlist'])
            np.testing.assert_array_equal(f['query_ids'], gt['qimlist'])
        np.testing.assert_array_equal(np.sort(ranks, axis=0), np.broadcast_to(np.arange(n)[:, None], ranks.shape))
        m, metrics = R.measure(ranks, gt)
        assert abs(m-float(row['map'])) < 1e-12
        for qid, metric in zip(gt['qimlist'], metrics):
            saved = per[ds, method, condition, qid]
            for key in ['ap', 'p10', 'recall20', 'recall50', 'recall100']:
                error = abs(metric[key]-float(saved[key]))
                assert error < 1e-12
                max_error = max(max_error, error)
        if index % 50 == 0:
            print('VERIFIED RANKINGS', index, '/', len(views), flush=True)
    assert sum(name.startswith('ranks_') for name in files) == 310
    check_path = compact/'validation_local.json'
    report = json.loads(check_path.read_text())
    report.update(rankings_verified=310, ap_values_independently_rescored=17050,
                  descriptor_files_verified=2, max_metric_recompute_error=max_error,
                  limits='All 310 saved rankings independently rescored; no new direct latency measurement')
    check_path.write_text(json.dumps(report, indent=2)+'\n')
    shutil_target = base/'results/b0_complete_v2/validation_local.json'
    shutil_target.write_bytes(check_path.read_bytes())

    target = base/'publish/b0-complete-v2-results'
    target.mkdir(parents=True, exist_ok=True)
    inventory = dict(kernel='ndhoang2310/iir-b0-epoch12-complete-clean-and-robustness-v2/1',
        scope='Complete B0 epoch12 clean and corruption evaluation; five fixed methods; both legacy datasets',
        excluded=['weights', 'benchmark images', 'duplicate progress memmaps', 'execution logs'],
        files=[dict(path=name, bytes=path.stat().st_size, sha256=R.digest(path)) for name, path in sorted(files.items())])
    archive = target/'b0-complete-v2.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    expected = {row['path']: row for row in inventory['files']}; seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            stream = tar.extractfile(member); h = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
            assert h.hexdigest() == expected[member.name]['sha256']
            seen.add(member.name)
    assert seen == set(expected)
    inventory.update(archive=archive.name, archive_bytes=archive.stat().st_size,
                     archive_sha256=R.digest(archive), payload_files=len(files),
                     payload_bytes=sum(r['bytes'] for r in inventory['files']), archive_verified=True)
    (target/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    (target/'README.md').write_text('''# Complete B0 epoch-12 results

Both Oxford5k and Paris6k are complete: five methods, 31 views per query and
17,050 AP records. All 310 saved rankings were checked as full gallery
permutations and independently rescored. `inventory.json` contains payload
hashes and the local archive hash. Kaggle expands the archive into
`b0-complete-v2/`; validate extracted files against the inventory.

Includes final descriptors, rankings, per-query metrics, aggregates, corruption
pixel hashes, graph matrices/diagnostics, execution environment and provenance.
Weights, benchmark images, duplicate progress arrays and logs are excluded.
This complete v2 supersedes the archived v1 Oxford-only timeout recovery.
Code and the paper are in FUAVX/IIR-Robust-Retrieval on GitHub.
''')
    (target/'dataset-metadata.json').write_text(json.dumps(dict(
        id='ndhoang2310/iir-b0-complete-epoch12-results-v2', title='IIR B0 Complete Epoch12 Results v2',
        licenses=[dict(name='other')],
        description='Complete clean and robustness numerical outputs for validation-selected B0 epoch12. Both Oxford and Paris; 310 full rankings and 17050 independently rescored AP records.'), indent=2)+'\n')
    print(json.dumps({k: v for k, v in inventory.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    package()
