"""Replay one fixed mutual/top-20 method on all archived query views, CPU only."""
import csv
import json
from pathlib import Path
import pickle

import numpy as np
from scipy import sparse

from diffusion_checks import ROOT, historical as R, hash_file, seeds, propagate


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    base = ROOT/'artifacts/kaggle-posttraining/outputs'
    output = ROOT/'artifacts/revision-validation/results'
    output.mkdir(parents=True, exist_ok=True)
    views, per_query, sources = [], [], []
    for model in R.P.MODEL_IDS:
        for dataset in R.P.DATASETS:
            path = base/'iir-post100-extract/unpacked'/f'desc_{model}_{dataset}.npz'
            with np.load(path, allow_pickle=False) as f:
                gallery, queries, ids, qids, names = [f[k] for k in ['gallery', 'queries', 'gallery_ids', 'query_ids', 'conditions']]
            metadata = ROOT/f'data/test/{dataset}/gnd_{dataset}.pkl'
            assert hash_file(metadata) == R.P.GND_HASHES[dataset]
            with metadata.open('rb') as f:
                gt = pickle.load(f)
            np.testing.assert_array_equal(ids, gt['imlist'])
            np.testing.assert_array_equal(qids, gt['qimlist'])
            assert list(names) == [c['name'] for c in R.P.conditions()]
            assert queries.shape[1] == len(qids)*len(names)
            graph_path = base/'iir-post100-diffusion/unpacked'/f'graph_{model}_{dataset}_k20_True.npz'
            graph = sparse.load_npz(graph_path)
            sources.append(dict(model=model, dataset=dataset, descriptors=hash_file(path), graph=hash_file(graph_path)))
            for index, condition in enumerate(R.P.conditions()):
                q = queries[:, index*len(qids):(index+1)*len(qids)]
                scores = gallery.T @ q
                values, diag = propagate(graph, seeds(scores, ids, k=20))
                ranks = R.ranking(values, ids)
                m, metrics = R.measure(ranks, gt)
                views.append(dict(model=model, dataset=dataset, condition=condition['name'], family=condition['family'],
                                  severity=condition['severity'], realization=condition['realization'], map=m,
                                  p10=float(np.mean([r['p10'] for r in metrics])), **diag))
                for qid, metric in zip(qids, metrics):
                    per_query.append(dict(model=model, dataset=dataset, condition=condition['name'], query_id=str(qid), **metric))
                print(model, dataset, condition['name'], round(m*100, 3), flush=True)
    summaries = []
    for model in R.P.MODEL_IDS:
        for dataset in R.P.DATASETS:
            group = [r for r in views if r['model'] == model and r['dataset'] == dataset]
            clean = next(r for r in group if r['family'] == 'clean')
            conditions = []
            for family in R.P.CORRUPTIONS:
                for severity in [1, 2, 3]:
                    matching = [r for r in group if r['family'] == family and r['severity'] == severity]
                    conditions.append(float(np.mean([r['map'] for r in matching])))
            summaries.append(dict(model=model, dataset=dataset, method='diffusion_mutual_top20',
                                  clean_map=clean['map'], corrupted_map=float(np.mean(conditions)), worst_map=min(conditions)))
    write_csv(output/'sparse_diffusion_views.csv', views)
    write_csv(output/'sparse_diffusion_per_query.csv', per_query)
    write_csv(output/'sparse_diffusion_summary.csv', summaries)
    (output/'sparse_diffusion_manifest.json').write_text(json.dumps(dict(
        protocol='revision_mutual_top20_v1', graph_k=20, seed_k=20, beta=.9,
        tolerance=1e-12, max_iterations=1000, training=False, new_encoder_inference=False,
        tuning=False, views=len(views), per_query_records=len(per_query), sources=sources,
        aggregation='realizations within each of 18 family/severity conditions, then equal condition mean',
        limitations=['same historical epoch100 encoders', 'post-review implementation diagnosis', 'no new latency measurements']), indent=2)+'\n')


if __name__ == '__main__':
    main()
