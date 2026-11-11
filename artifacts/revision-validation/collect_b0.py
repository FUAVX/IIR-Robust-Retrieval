"""Check once and collect the bounded B0 job through the credential bridge."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BRIDGE = '/home/hoang/.local/bin/kaggle-account'
KERNEL = 'ndhoang2310/iir-revision-b0-clean-epoch12-v1'
METHODS = {'cosine', 'AQE', 'alpha_QE', 'gated_QE', 'diffusion_mutual_top20'}


def call(account, args):
    result = subprocess.run([BRIDGE, 'run', account, '--', *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stdout+result.stderr).strip())
    return result.stdout


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def recover_oxford(output):
    """Accept a complete dataset only; never label the timed-out job complete."""
    import numpy as np
    sys.path.insert(0, str(ROOT/'artifacts/kaggle-posttraining'))
    import runtime as R
    import pickle
    matches = list(output.rglob('desc_b0_epoch12_oxford5k_clean.npz'))
    assert len(matches) == 1
    folder = matches[0].parent
    status_path = folder/'status.json'
    if not status_path.is_file():
        status_path = folder/'original_job_status.json'
    status = json.loads(status_path.read_text())
    assert status['status'] == 'FAILED' and '30-minute extraction budget exceeded' in status['error']
    expected = json.loads((ROOT/'artifacts/revision-validation/results/b0_input_manifest.json').read_text())
    with np.load(matches[0], allow_pickle=False) as f:
        ids, qids = f['gallery_ids'], f['query_ids']
        assert str(f['checkpoint_sha256']) == expected['source_checkpoint_sha256']
        assert f['gallery'].shape == (512, 5063) and f['queries'].shape == (512, 55)
        assert np.isfinite(f['gallery']).all() and np.isfinite(f['queries']).all()
        np.testing.assert_allclose(np.linalg.norm(f['gallery'], axis=0), 1, atol=2e-5)
        np.testing.assert_allclose(np.linalg.norm(f['queries'], axis=0), 1, atol=2e-5)
    groundtruth = ROOT/'data/test/oxford5k/gnd_oxford5k.pkl'
    assert R.digest(groundtruth) == R.P.GND_HASHES['oxford5k']
    with groundtruth.open('rb') as f:
        gt = pickle.load(f)
    np.testing.assert_array_equal(ids, gt['imlist']); np.testing.assert_array_equal(qids, gt['qimlist'])
    summary, per_query, files = [], [], []
    for method in sorted(METHODS):
        path = folder/f'ranks_oxford5k_{method}.npz'
        with np.load(path, allow_pickle=False) as f:
            ranks = f['ranks']
            np.testing.assert_array_equal(f['gallery_ids'], ids)
            np.testing.assert_array_equal(f['query_ids'], qids)
        assert ranks.shape == (5063, 55)
        np.testing.assert_array_equal(np.sort(ranks, axis=0), np.broadcast_to(np.arange(5063)[:, None], ranks.shape))
        m, metrics = R.measure(ranks, gt)
        summary.append(dict(model='b0_epoch12', dataset='oxford5k', method=method, map=m,
                            p10=float(np.mean([r['p10'] for r in metrics]))))
        per_query.extend(dict(dataset='oxford5k', method=method, query_id=str(qid), **metric)
                         for qid, metric in zip(qids, metrics))
        files.append(dict(file=path.name, sha256=R.digest(path)))
    assert len(summary) == 5 and len(per_query) == 275
    target = ROOT/'artifacts/revision-validation/results/b0_clean'
    target.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.csv', summary), ('per_query.csv', per_query)]:
        with (target/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(data)
    manifest = dict(status='PARTIAL_BUDGET_LIMIT', kernel=KERNEL, version=1,
                    completed_datasets=['oxford5k'], incomplete_datasets=['paris6k'],
                    failed_job_status=status, input_manifest=expected,
                    source_descriptor_sha256=R.digest(matches[0]), rankings=files,
                    recovery='All five complete Oxford rankings checked as full gallery permutations; AP/P@10 recomputed locally',
                    runtime_environment='Not recorded before timeout; no GPU type/version inferred from the request')
    (target/'recovery_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copy2(folder/'graph_oxford5k.json', target/'graph_oxford5k.json')
    validation = ROOT/'artifacts/revision-validation/results/validation.json'
    record = json.loads(validation.read_text())
    record['gpu_b0'] = 'PARTIAL_BUDGET_LIMIT: Oxford five groups/275 AP records verified; Paris incomplete; no additional GPU job'
    validation.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def collect(account, recover_completed=False):
    status = call(account, ['kernels', 'status', KERNEL])
    print(status.strip(), flush=True)
    complete = 'KernelWorkerStatus.COMPLETE' in status
    failed = any(value in status.upper() for value in ['ERROR', 'CANCELLED', 'CANCELED'])
    if not complete and not (failed and recover_completed):
        if any(value in status.upper() for value in ['ERROR', 'CANCELLED', 'CANCELED']):
            print(call(account, ['kernels', 'logs', KERNEL]), flush=True)
            raise RuntimeError('B0 remote evaluation failed; inspect without resubmitting')
        return
    output = ROOT/'artifacts/revision-validation/outputs/b0-clean'
    output.mkdir(parents=True, exist_ok=True)
    print(call(account, ['kernels', 'output', KERNEL, '-p', str(output)]), flush=True)
    if not complete:
        recover_oxford(output)
        return
    matches = list(output.rglob('summary.csv'))
    assert len(matches) == 1
    folder = matches[0].parent
    assert json.loads((folder/'status.json').read_text()) == dict(status='COMPLETE', groups=10, queries_per_group=55)
    summary, query = rows(folder/'summary.csv'), rows(folder/'per_query.csv')
    assert len(summary) == 10 and len(query) == 550
    assert {(r['dataset'], r['method']) for r in summary} == {
        (dataset, method) for dataset in ['oxford5k', 'paris6k'] for method in METHODS}
    for row in summary:
        selected = [r for r in query if r['dataset'] == row['dataset'] and r['method'] == row['method']]
        assert len(selected) == 55 and len({r['query_id'] for r in selected}) == 55
        assert abs(sum(float(r['ap']) for r in selected)/55-float(row['map'])) < 1e-12
    run = json.loads((folder/'run_manifest.json').read_text())
    expected = json.loads((ROOT/'artifacts/revision-validation/results/b0_input_manifest.json').read_text())
    assert run['input_manifest'] == expected
    target = ROOT/'artifacts/revision-validation/results/b0_clean'
    target.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.suffix in {'.csv', '.json'}:
            shutil.copy2(path, target/path.name)
    validation = ROOT/'artifacts/revision-validation/results/validation.json'
    record = json.loads(validation.read_text())
    record['gpu_b0'] = 'COMPLETE: 10 clean groups, 550 AP records, input hashes and aggregate means verified'
    validation.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', default='hoang1')
    parser.add_argument('--recover-completed', action='store_true', help='Recover verified complete Oxford outputs from the known budget timeout')
    args = parser.parse_args()
    collect(args.account, args.recover_completed)
