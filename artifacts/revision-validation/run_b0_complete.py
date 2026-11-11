"""Complete B0 clean/robustness evaluation; reuse Oxford and checkpoint extraction.

This source is embedded by build_b0_complete_job.py. No training or parameter
selection. Run progress is saved locally and can be mounted as a future input.
"""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_name] = '2'

import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

EXPECTED_INPUT_MANIFEST = {}
OXFORD_CACHE_SHA256 = ''
REFERENCE_ROWS = []
METHODS = ['cosine', 'AQE', 'alpha_QE', 'gated_QE', 'diffusion_mutual_top20']
PROTOCOL = 'b0_complete_clean_robust_v2'
SOFT_SECONDS = 3000


class BudgetExpired(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def csv_file(path, rows):
    assert rows
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    pending.replace(path)


def prefix_hash(array, completed):
    return hashlib.sha256(np.ascontiguousarray(array[:, :completed]).tobytes()).hexdigest()


class ArrayStore:
    """Only a flushed, checksummed prefix is accepted after an interruption."""
    def __init__(self, folder, name, shape, identity, resume_root=None):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        self.array_path, self.state_path = folder/(name+'.npy'), folder/(name+'.json')
        self.identity, self.shape = identity, tuple(shape)
        candidates = [self.state_path] if self.state_path.exists() else []
        if resume_root is not None:
            candidates += list(Path(resume_root).rglob(name+'.json'))
        valid = []
        for candidate in candidates:
            state = json.loads(candidate.read_text())
            if state.get('identity') != identity or tuple(state.get('shape', [])) != self.shape:
                raise ValueError('Extraction checkpoint identity mismatch: '+str(candidate))
            n = state['completed']
            assert 0 <= n <= shape[1]
            assert state['array_file'] == name+'.npy'
            source = candidate.parent/state['array_file']
            a = np.load(source, mmap_mode='r', allow_pickle=False)
            assert a.shape == self.shape and a.dtype == np.float32
            if prefix_hash(a, n) != state['prefix_sha256']:
                raise ValueError('Extraction checkpoint prefix is corrupt')
            valid.append((n, candidate, state))
        best = max(valid, key=lambda row: row[0]) if valid else None
        if best and best[1] == self.state_path:
            self.array = np.lib.format.open_memmap(self.array_path, mode='r+')
        else:
            self.array = np.lib.format.open_memmap(self.array_path, mode='w+', dtype=np.float32, shape=self.shape)
            if best:
                source = np.load(best[1].parent/best[2]['array_file'], mmap_mode='r', allow_pickle=False)
                self.array[:, :best[0]] = source[:, :best[0]]
        self.completed = best[0] if best else 0
        self.pixels = best[2].get('pixels', []) if best else []
        self.save()

    def append(self, vector, pixel=None):
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        assert v.shape == (self.shape[0],) and np.isfinite(v).all()
        assert np.isclose(np.linalg.norm(v), 1, atol=2e-5)
        if self.completed >= self.shape[1]:
            raise ValueError('Checkpoint is already complete')
        self.array[:, self.completed] = v
        self.completed += 1
        if pixel is not None:
            self.pixels.append(pixel)

    def save(self):
        self.array.flush()
        atomic_json(self.state_path, dict(identity=self.identity, shape=list(self.shape),
            array_file=self.array_path.name, completed=self.completed,
            prefix_sha256=prefix_hash(self.array, self.completed), pixels=self.pixels))


class Images(Dataset):
    def __init__(self, paths, boxes=None, conditions=None, query_ids=None):
        self.paths, self.boxes = paths, boxes
        self.conditions, self.query_ids = conditions, query_ids

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import runtime as R
        image = R.prepare_image(self.paths[i], None if self.boxes is None else self.boxes[i])
        pixel = None
        if self.conditions is not None:
            image = R.corrupt(image, self.conditions[i], self.query_ids[i])
            pixel = dict(query_id=self.query_ids[i], condition=self.conditions[i]['name'],
                         pixel_hash=R.pixel_hash(image), size=list(image.size))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()/255).permute(2, 0, 1)
        x = (x-x.new_tensor([.485, .456, .406])[:, None, None])/x.new_tensor([.229, .224, .225])[:, None, None]
        return x, pixel


def single(items):
    return items[0]


def fill(model, store, tasks, started, soft_seconds=SOFT_SECONDS, device='cuda', workers=2):
    import runtime as R
    if store.completed == store.shape[1]:
        return
    pending = torch.utils.data.Subset(tasks, range(store.completed, len(tasks)))
    loader = DataLoader(pending, batch_size=1, num_workers=workers,
                        pin_memory=device.startswith('cuda'), collate_fn=single)
    try:
        with torch.inference_mode():
            for x, pixel in loader:
                if time.monotonic()-started > soft_seconds:
                    raise BudgetExpired('50-minute work budget reached; extraction prefix saved')
                v = model(x.unsqueeze(0).to(device)).float().cpu().numpy()
                store.append(R.unit(v)[:, 0], pixel)
                if store.completed % 64 == 0 or store.completed == len(tasks):
                    store.save()
                    print('EXTRACT', store.array_path.name, store.completed, '/', len(tasks),
                          'seconds', round(time.monotonic()-started, 1), flush=True)
    finally:
        store.save()


def evaluate_dataset(dataset, gt, gallery, queries, ids, conditions, output, started, soft_seconds=SOFT_SECONDS):
    import runtime as R
    from diffusion_checks import seeds, propagate
    from scipy import sparse
    graph, graph_stats = R.build_graph(gallery, ids, k=20, mutual=True)
    sparse.save_npz(output/f'graph_{dataset}_mutual20.npz', graph)
    atomic_json(output/f'graph_{dataset}_stats.json', graph_stats)
    view_rows, per_query = [], []
    nq = len(gt['qimlist'])
    assert queries.shape[1] == nq*len(conditions)
    for ci, condition in enumerate(conditions):
        if time.monotonic()-started > soft_seconds:
            raise BudgetExpired('50-minute work budget reached; descriptors and completed rankings retained')
        q = queries[:, ci*nq:(ci+1)*nq]
        scores = gallery.T @ q
        methods = {'cosine': scores}
        for name, params in [('AQE', dict(alpha=0)), ('alpha_QE', dict(alpha=3)),
                             ('gated_QE', dict(alpha=3, tau=.5))]:
            expanded = np.column_stack([R.qe(q[:, j], gallery, ids, k=10, lam=.5, **params)[0] for j in range(nq)])
            methods[name] = gallery.T @ expanded
        seed = seeds(scores, ids, k=20)
        diffused, diagnostics = propagate(graph, seed)
        if diagnostics['max_relative_residual'] > 1e-8:
            raise RuntimeError('Diffusion residual check failed')
        empty = seed.sum(axis=0) == 0
        diffused[:, empty] = scores[:, empty]
        methods['diffusion_mutual_top20'] = diffused
        current_views, current_queries = [], []
        for method in METHODS:
            ranks = R.ranking(methods[method], ids)
            m, metrics = R.measure(ranks, gt)
            row = dict(model='b0_epoch12', dataset=dataset, method=method,
                       condition=condition['name'], family=condition['family'], severity=condition['severity'],
                       realization=condition['realization'], map=m,
                       p10=float(np.mean([r['p10'] for r in metrics])))
            current_views.append(row)
            current_queries.extend(dict(model='b0_epoch12', dataset=dataset, method=method,
                condition=condition['name'], query_id=qid, **metric) for qid, metric in zip(gt['qimlist'], metrics))
            np.savez_compressed(output/f'ranks_{dataset}_{condition["name"]}_{method}.npz',
                                ranks=ranks, query_ids=np.asarray(gt['qimlist']), gallery_ids=ids)
        atomic_json(output/f'conditions/{dataset}_{condition["name"]}.json',
                    dict(status='COMPLETE', views=current_views, per_query=current_queries,
                         diffusion_diagnostics=diagnostics))
        view_rows.extend(current_views); per_query.extend(current_queries)
        csv_file(output/f'views_{dataset}.csv', view_rows)
        csv_file(output/f'per_query_{dataset}.csv', per_query)
        print('EVALUATED', dataset, condition['name'], flush=True)
    return view_rows, per_query


def summarize(views):
    rows = []
    for dataset in ['oxford5k', 'paris6k']:
        for method in METHODS:
            group = [r for r in views if r['dataset'] == dataset and r['method'] == method]
            assert len(group) == 31
            clean = next(r for r in group if r['family'] == 'clean')
            corrupt, p10 = [], []
            for family in ['darkening', 'contrast', 'blur', 'noise', 'jpeg', 'occlusion']:
                for severity in [1, 2, 3]:
                    selected = [r for r in group if r['family'] == family and r['severity'] == severity]
                    assert len(selected) == (3 if family in ['noise', 'occlusion'] else 1)
                    corrupt.append(float(np.mean([r['map'] for r in selected])))
                    p10.append(float(np.mean([r['p10'] for r in selected])))
            rows.append(dict(model='b0_epoch12', dataset=dataset, method=method,
                clean_map=clean['map'], corrupted_map=float(np.mean(corrupt)), worst_map=min(corrupt),
                drop_pp=100*(clean['map']-float(np.mean(corrupt))),
                clean_p10=clean['p10'], corrupted_p10=float(np.mean(p10))))
    return rows


def configure_runtime(inputs):
    weights = list(inputs.rglob('b0_epoch12.pth'))
    assert len(weights) == 1, weights
    code = weights[0].parent
    manifest = json.loads((code/'manifest.json').read_text())
    assert manifest == EXPECTED_INPUT_MANIFEST, 'Unexpected B0 input version'
    for name, expected in manifest['files'].items():
        assert digest(code/name) == expected, name
    sys.path.insert(0, str(code))
    return weights[0], manifest


def main(inputs=Path('/kaggle/input'), output=Path('/kaggle/working/b0_complete_v2')):
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'status.json', dict(status='STARTING', protocol=PROTOCOL))
    try:
        weights, manifest = configure_runtime(inputs)
        import runtime as R
        from encoder_runtime import RetrievalModel
        from PIL import __version__ as pillow_version
        import cv2
        assert torch.cuda.is_available(), 'This extraction job requires a GPU'
        torch.set_num_threads(2); cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        conditions = R.P.conditions()
        assert len(conditions) == 31
        environment = dict(torch=torch.__version__, numpy=np.__version__, pillow=pillow_version,
                           opencv=cv2.__version__, gpu=torch.cuda.get_device_name(), precision='FP32', tf32=False)
        extraction_identity = dict(protocol=PROTOCOL, runner_sha256=digest(Path(__file__)),
                                   checkpoint=manifest['source_checkpoint_sha256'],
                                   files=manifest['files'], conditions=conditions, environment=environment)
        atomic_json(output/'run_manifest.json', dict(protocol=PROTOCOL, input_manifest=manifest,
            environment=environment, script_sha256=digest(Path(__file__)), training=False, tuning=False,
            methods=METHODS, gallery='clean', query_views=conditions, soft_seconds=SOFT_SECONDS,
            qe=dict(k=10, lam=.5, alpha=3, tau=.5, minimum=3),
            diffusion=dict(mutual=True, graph_k=20, seed_k=20, beta=.9, tolerance=1e-12),
            comparison_scope='B0 validation epoch12 vs historical R1/R2 epoch100; descriptive, not training ablation'))
        model = RetrievalModel(attention=False).eval().cuda()
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        model.precision = 'fp32'
        started = time.monotonic()
        all_views, all_queries = [], []
        cache = R.find_one(inputs, 'desc_b0_epoch12_oxford5k_clean.npz')
        assert digest(cache) == OXFORD_CACHE_SHA256, 'Oxford cache checksum mismatch'
        for dataset, expected_n in [('oxford5k', 5063), ('paris6k', 6392)]:
            metadata = R.find_one(inputs, f'gnd_{dataset}.pkl')
            assert digest(metadata) == R.P.GND_HASHES[dataset]
            with metadata.open('rb') as f:
                gt = pickle.load(f)
            ids, qids = np.asarray(gt['imlist']), np.asarray(gt['qimlist'])
            assert len(ids) == expected_n and len(qids) == 55 and len(set(ids)) == len(ids)
            images = metadata.parent/'jpg'
            identity = dict(**extraction_identity, dataset=dataset, groundtruth=digest(metadata),
                            gallery_ids=identity_hash(ids.tolist()), query_ids=identity_hash(qids.tolist()))
            gallery = ArrayStore(output/'checkpoints', 'b0_v2_'+dataset+'_gallery', (512, len(ids)),
                                 identity_hash(dict(identity, kind='gallery')), resume_root=inputs)
            queries = ArrayStore(output/'checkpoints', 'b0_v2_'+dataset+'_queries', (512, 31*len(qids)),
                                 identity_hash(dict(identity, kind='queries')), resume_root=inputs)
            if dataset == 'oxford5k':
                with np.load(cache, allow_pickle=False) as f:
                    assert str(f['checkpoint_sha256']) == manifest['source_checkpoint_sha256']
                    np.testing.assert_array_equal(f['gallery_ids'], ids)
                    np.testing.assert_array_equal(f['query_ids'], qids)
                    if gallery.completed == 0:
                        for v in f['gallery'].T:
                            gallery.append(v)
                        gallery.save()
                    if queries.completed == 0:
                        for j, v in enumerate(f['queries'].T):
                            image = R.prepare_image(images/(str(qids[j])+'.jpg'), gt['gnd'][j]['bbx'])
                            queries.append(v, dict(query_id=str(qids[j]), condition='clean',
                                pixel_hash=R.pixel_hash(image), size=list(image.size)))
                        queries.save()
                print('REUSED Oxford gallery and clean query descriptors', flush=True)
            gallery_tasks = Images([images/(str(i)+'.jpg') for i in ids])
            fill(model, gallery, gallery_tasks, started)
            tasks = [(images/(str(qid)+'.jpg'), gt['gnd'][j]['bbx'], condition, str(qid))
                     for condition in conditions for j, qid in enumerate(qids)]
            query_tasks = Images([t[0] for t in tasks], [t[1] for t in tasks],
                                 [t[2] for t in tasks], [t[3] for t in tasks])
            fill(model, queries, query_tasks, started)
            assert len(queries.pixels) == 31*len(qids)
            atomic_json(output/f'pixels_b0_epoch12_{dataset}.json', queries.pixels)
            np.savez_compressed(output/f'desc_b0_epoch12_{dataset}_robust.npz',
                gallery=gallery.array, queries=queries.array, gallery_ids=ids, query_ids=qids,
                conditions=np.array([c['name'] for c in conditions]),
                checkpoint_sha256=np.array(manifest['source_checkpoint_sha256']))
            views, per_query = evaluate_dataset(dataset, gt, gallery.array, queries.array, ids, conditions, output, started)
            all_views.extend(views); all_queries.extend(per_query)
            atomic_json(output/f'dataset_{dataset}_complete.json', dict(status='COMPLETE',
                        views=len(views), ap_records=len(per_query)))
        assert len(all_views) == 310 and len(all_queries) == 17050
        summary = summarize(all_views)
        csv_file(output/'summary.csv', summary)
        csv_file(output/'views.csv', all_views)
        csv_file(output/'per_query.csv', all_queries)
        comparison = [dict(model=r['model'], epoch=12, selection='validation_loss', dataset=r['dataset'],
            method=r['method'], clean_map=r['clean_map'], corrupted_map=r['corrupted_map'],
            worst_map=r['worst_map'], record_origin=PROTOCOL) for r in summary] + REFERENCE_ROWS
        assert len(REFERENCE_ROWS) == 20
        csv_file(output/'comparison_b0_r1_r2.csv', comparison)
        atomic_json(output/'status.json', dict(status='COMPLETE', protocol=PROTOCOL, datasets=2,
                    methods=5, views_per_query=31, ap_records=17050, seconds=time.monotonic()-started))
        print('COMPLETE B0: clean + full frozen corruption schedule, both datasets', flush=True)
    except BudgetExpired as error:
        atomic_json(output/'status.json', dict(status='PARTIAL_BUDGET_LIMIT', error=str(error),
                    resume='Attach this output as an input to a future run to restore verified descriptor prefixes'))
        print(str(error), flush=True)
    except Exception as error:
        atomic_json(output/'status.json', dict(status='FAILED', error=str(error)))
        raise


if __name__ == '__main__':
    main()
