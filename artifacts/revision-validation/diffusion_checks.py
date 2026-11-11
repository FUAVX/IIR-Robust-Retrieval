"""Independent diffusion checks; historical runtime.py remains unchanged.

The top-k seed follows the support rule in the authors' ymake.m. Comparing it
with the archived dense seed diagnoses implementation choices, not parameters.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'artifacts/kaggle-posttraining'))
import runtime as historical


def seeds(scores, ids, k=None):
    """Nonnegative cubed similarities, optionally restricted to top-k, unit mass."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.maximum(scores, 0) ** 3
    if k is not None:
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer or None')
        order = historical.ranking(scores, ids)[:k]
        selected = np.zeros_like(y)
        np.put_along_axis(selected, order, np.take_along_axis(y, order, axis=0), axis=0)
        y = selected
    return y / np.maximum(y.sum(axis=0, keepdims=True), 1e-20)


def propagate(graph, y, beta=.9, tolerance=1e-12, max_iterations=1000):
    if not 0 <= beta < 1:
        raise ValueError('beta must be in [0, 1)')
    f = y.copy()
    for iteration in range(1, max_iterations + 1):
        updated = (1-beta)*y + beta*(graph @ f)
        change = np.linalg.norm(updated-f, axis=0) / np.maximum(np.linalg.norm(f, axis=0), 1e-20)
        f = updated
        if np.all(change < tolerance):
            break
    residual = np.linalg.norm(f-beta*(graph@f)-(1-beta)*y, axis=0)
    relative = residual / np.maximum(np.linalg.norm((1-beta)*y, axis=0), 1e-20)
    if not np.isfinite(f).all():
        raise FloatingPointError('Nonfinite diffusion output')
    return f, dict(iterations=iteration, max_relative_residual=float(relative.max()))


def normalize_affinity(w):
    """Dense reference, deliberately separate from historical sparse code."""
    w = np.asarray(w, dtype=np.float64)
    degree = w.sum(axis=1)
    inv = np.zeros_like(degree)
    inv[degree > 0] = 1 / np.sqrt(degree[degree > 0])
    return inv[:, None] * w * inv[None, :]


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def audit(project, output):
    output.mkdir(parents=True, exist_ok=True)
    base = project / 'artifacts/kaggle-posttraining/outputs'
    extracts = base / 'iir-post100-extract/unpacked'
    graphs = base / 'iir-post100-diffusion/unpacked'
    rows, evidence = [], []
    for model in ['run1_aug_epoch100', 'run2_aug_attention_epoch100']:
        for dataset in ['oxford5k', 'paris6k']:
            path = extracts / f'desc_{model}_{dataset}.npz'
            with np.load(path, allow_pickle=False) as f:
                gallery, queries = f['gallery'], f['queries']
                ids, conditions, qids = f['gallery_ids'], f['conditions'], f['query_ids']
            clean_index = list(conditions).index('clean')
            query = queries[:, clean_index*len(qids):(clean_index+1)*len(qids)]
            groundtruth = project / f'data/test/{dataset}/gnd_{dataset}.pkl'
            expected = historical.P.GND_HASHES[dataset]
            if hash_file(groundtruth) != expected:
                raise ValueError('Unrecognized benchmark metadata')
            with groundtruth.open('rb') as f:
                gt = pickle.load(f)  # locally supplied, hash-verified benchmark
            np.testing.assert_array_equal(ids, gt['imlist'])
            np.testing.assert_array_equal(qids, gt['qimlist'])
            scores = gallery.T @ query
            cosine_map, cosine_rows = historical.measure(historical.ranking(scores, ids), gt)
            rows.append(dict(model=model, dataset=dataset, graph='none', seed='none', method='cosine',
                             map=cosine_map, p10=float(np.mean([r['p10'] for r in cosine_rows])),
                             max_relative_residual=0, max_cg_relative_error=0, iterations=0))
            for mutual in [False, True]:
                graph_path = graphs / f'graph_{model}_{dataset}_k20_{mutual}.npz'
                graph = sparse.load_npz(graph_path)
                if graph.shape != (len(ids), len(ids)) or not np.isfinite(graph.data).all():
                    raise ValueError('Invalid graph')
                asymmetry = graph-graph.T
                assert not asymmetry.nnz or np.max(np.abs(asymmetry.data)) < 1e-12
                assert np.all(graph.diagonal() == 0) and np.all(graph.data >= 0)
                evidence.append(dict(model=model, dataset=dataset, mutual=mutual,
                                     descriptor_sha256=hash_file(path), graph_sha256=hash_file(graph_path)))
                for support in [None, 20]:
                    y = seeds(scores, ids, support)
                    f, diagnostics = propagate(graph, y)
                    # Independent CG solver on three deterministic queries.
                    errors = []
                    for j in [0, len(qids)//2, len(qids)-1]:
                        ref, status = cg(sparse.eye(len(ids), format='csr')-.9*graph,
                                         .1*y[:, j], rtol=1e-11, atol=0, maxiter=2000)
                        assert status == 0, status
                        errors.append(float(np.linalg.norm(ref-f[:, j])/max(np.linalg.norm(ref), 1e-20)))
                    m, metrics = historical.measure(historical.ranking(f, ids), gt)
                    row = dict(model=model, dataset=dataset, graph='mutual' if mutual else 'union',
                               seed='dense' if support is None else 'top20', method='diffusion', map=m,
                               p10=float(np.mean([r['p10'] for r in metrics])),
                               max_cg_relative_error=max(errors), **diagnostics)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    with (output/'diffusion_clean_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(scope='clean epoch100 diagnostic; no parameter selection', beta=.9,
                  graph_k=20, seed_supports=['dense', 20], tolerance=1e-12,
                  solver='fixed point versus independent scipy CG on 3 queries per graph/seed',
                  numpy=np.__version__, sources=evidence, rows=len(rows))
    (output/'diffusion_clean_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/revision-validation/results')
    args = parser.parse_args()
    audit(args.project, args.output)
