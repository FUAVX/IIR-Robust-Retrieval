"""One B0 checkpoint, clean legacy benchmarks, FP32. No training or tuning."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from encoder_runtime import RetrievalModel
import runtime as R
from diffusion_checks import seeds, propagate


class Images(Dataset):
    def __init__(self, paths, boxes=None):
        self.paths, self.boxes = paths, boxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = R.prepare_image(self.paths[i], None if self.boxes is None else self.boxes[i])
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()/255).permute(2, 0, 1)
        return (x-x.new_tensor([.485, .456, .406])[:, None, None])/x.new_tensor([.229, .224, .225])[:, None, None]


def extract(model, paths, boxes, started):
    loader = DataLoader(Images(paths, boxes), batch_size=1, num_workers=2, pin_memory=True)
    vectors = np.empty((512, len(paths)), dtype=np.float32)
    with torch.inference_mode():
        for i, x in enumerate(loader):
            if time.monotonic()-started > 1800:
                raise TimeoutError('30-minute extraction budget exceeded; no partial result claimed')
            v = model(x.cuda(non_blocking=True)).float().cpu().numpy()
            assert v.shape == (512, 1) and np.isfinite(v).all()
            vectors[:, i] = R.unit(v)[:, 0]
            if i % 250 == 0 or i+1 == len(paths):
                print('EXTRACT', i+1, '/', len(paths), 'elapsed', round(time.monotonic()-started, 1), flush=True)
    return vectors


def main(code, input_root, output):
    code, input_root, output = map(Path, [code, input_root, output])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((code/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert R.digest(code/name) == expected, name
    assert torch.cuda.is_available(), 'A GPU is required for this bounded extraction job'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = RetrievalModel(attention=False).eval().cuda()
    model.load_state_dict(torch.load(code/'b0_epoch12.pth', map_location='cpu', weights_only=True), strict=True)
    model.precision = 'fp32'
    started = time.monotonic()
    rows, per_query = [], []
    for dataset in ['oxford5k', 'paris6k']:
        metadata = R.find_one(input_root, f'gnd_{dataset}.pkl')
        assert R.digest(metadata) == R.P.GND_HASHES[dataset]
        with metadata.open('rb') as f:
            gt = pickle.load(f)
        images = metadata.parent/'jpg'
        ids, qids = np.asarray(gt['imlist']), np.asarray(gt['qimlist'])
        gallery = extract(model, [images/(i+'.jpg') for i in ids], None, started)
        queries = extract(model, [images/(i+'.jpg') for i in qids], [g['bbx'] for g in gt['gnd']], started)
        np.savez_compressed(output/f'desc_b0_epoch12_{dataset}_clean.npz', gallery=gallery, queries=queries,
                            gallery_ids=ids, query_ids=qids, checkpoint_sha256=np.array(manifest['source_checkpoint_sha256']))
        scores = gallery.T @ queries
        methods = {'cosine': scores}
        # Freeze all settings before running: no sensitivity sweep on test queries.
        for name, params in [('AQE', dict(alpha=0)), ('alpha_QE', dict(alpha=3)),
                             ('gated_QE', dict(alpha=3, tau=.5))]:
            expanded = np.column_stack([R.qe(queries[:, j], gallery, ids, k=10, lam=.5, **params)[0]
                                        for j in range(len(qids))])
            methods[name] = gallery.T @ expanded
        graph, stats = R.build_graph(gallery, ids, k=20, mutual=True)
        diffused, diagnostics = propagate(graph, seeds(scores, ids, k=20))
        methods['diffusion_mutual_top20'] = diffused
        R.js(output/f'graph_{dataset}.json', dict(stats=stats, diagnostics=diagnostics))
        for method, values in methods.items():
            ranks = R.ranking(values, ids)
            m, metrics = R.measure(ranks, gt)
            row = dict(model='b0_epoch12', dataset=dataset, method=method, map=m,
                       p10=float(np.mean([r['p10'] for r in metrics])))
            rows.append(row)
            for qid, metric in zip(qids, metrics):
                per_query.append(dict(dataset=dataset, method=method, query_id=str(qid), **metric))
            np.savez_compressed(output/f'ranks_{dataset}_{method}.npz', ranks=ranks, query_ids=qids, gallery_ids=ids)
            print(json.dumps(row), flush=True)
    for name, data in [('summary.csv', rows), ('per_query.csv', per_query)]:
        with (output/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    R.js(output/'run_manifest.json', dict(input_manifest=manifest, torch=torch.__version__, numpy=np.__version__,
         device=torch.cuda.get_device_name(), seconds=time.monotonic()-started,
         protocol='clean_legacy_v1', query_expansion=dict(k=10, lam=.5, alpha=3, tau=.5),
         diffusion=dict(mutual=True, graph_k=20, seed_k=20, beta=.9, tolerance=1e-12),
         limitations=['one checkpoint', 'legacy ground truth', 'fixed parameters; no tuning',
                      'no corrupted queries, geometry, fusion or direct end-to-end latency']))
    R.js(output/'status.json', dict(status='COMPLETE', groups=len(rows), queries_per_group=55))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.code, args.input, args.output)
