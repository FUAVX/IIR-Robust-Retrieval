#!/usr/bin/env python3
"""
scripts/profile_sfm.py — profile full SfM metadata to decide subset size.

Reads retrieval-SfM-120k-full.pkl (fallback: retrieval-SfM-120k.pkl),
prints cluster/image/pair distribution + candidate table for
k in [25, 50, 100, 200] with T4 epoch-time estimates.

Usage:
    python3 scripts/profile_sfm.py --data-root ./data --seed 42
"""

import argparse
import os
import pickle
import random
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_full(data_root):
    sfm = os.path.join(data_root, "train", "retrieval-SfM-120k")
    for name in ("retrieval-SfM-120k-full.pkl", "retrieval-SfM-120k.pkl"):
        p = os.path.join(sfm, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                print(f"[info] loaded {p}")
                return pickle.load(f), p
    raise FileNotFoundError(
        f"No SfM pkl in {sfm}. Run: python3 scripts/download_datasets_local.py --only sfm-meta")


def summarize(part, name):
    cids, clusters = part["cids"], part["cluster"]
    qidxs, pidxs = part["qidxs"], part["pidxs"]
    sizes = Counter(clusters)
    print(f"--- {name}: {len(cids):,} images, {len(sizes):,} clusters, "
          f"{len(qidxs):,} (q,p) pairs")
    print(f"    cluster size: min={min(sizes.values())} "
          f"median={sorted(sizes.values())[len(sizes)//2]} "
          f"max={max(sizes.values())} mean={len(cids)/len(sizes):.1f}")


def simulate_subset(part, k, seed):
    """Same logic as download script: sample k clusters, keep pairs inside."""
    cids, clusters = part["cids"], part["cluster"]
    qidxs, pidxs = part["qidxs"], part["pidxs"]
    cl_to_idx = {}
    for i, c in enumerate(clusters):
        cl_to_idx.setdefault(c, []).append(i)
    rng = random.Random(seed)
    chosen = set(rng.sample(sorted(cl_to_idx.keys()), min(k, len(cl_to_idx))))
    kept = set(i for c in chosen for i in cl_to_idx[c])
    pairs = sum(1 for q, p in zip(qidxs, pidxs) if q in kept and p in kept)
    n_img = len(kept)
    # mining viability: distinct negative clusters available per query
    neg_clusters = len(chosen) - 1  # minus query's own cluster (worst case)
    ok = neg_clusters >= 5
    return n_img, pairs, len(chosen), ok


def est_epoch_min(qsize, poolsize, imsize):
    """Rough T4 estimate: mining 0.075s/img + train 0.075s fwd, 0.15s bwd per img@512.
    Scale by (imsize/512)^2."""
    scale = (imsize / 512) ** 2
    mining = poolsize * 0.075 * scale / 60
    train = qsize * 7 * (0.075 + 0.15) * scale / 60
    return mining + train, mining, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", default="25,50,100,200")
    args = ap.parse_args()

    db, src = load_full(os.path.abspath(args.data_root))
    for part in ("train", "val"):
        if part in db:
            summarize(db[part], part)

    print("\nCandidate subsets (seed=%d, stratified-random same as trainer):" % args.seed)
    print(f"{'k':>5} | {'train imgs':>10} | {'train pairs':>11} | "
          f"{'neg-clu':>7} | {'mining-ok':>8} | {'disk*':>7} | "
          f"{'ep@512(q500/p5k)':>16} | {'ep@768(q1k/p10k)':>16}")
    print("-" * 110)
    for k in [int(x) for x in args.ks.split(",")]:
        n_img, pairs, nc, ok = simulate_subset(db["train"], k, args.seed)
        disk_gb = n_img * 260 / 1e6  # ~260KB mean JPEG
        e1, _, _ = est_epoch_min(500, 5000, 512)
        e2, _, _ = est_epoch_min(1000, 10000, 768)
        # scale epoch by actual pairs cap: if pairs < qsize, epoch shorter
        e1 *= min(1.0, pairs / 500) if pairs else 0
        e2 *= min(1.0, pairs / 1000) if pairs else 0
        print(f"{k:>5} | {n_img:>10,} | {pairs:>11,} | {nc-1:>7} | "
              f"{'OK' if ok else 'THIEU':>8} | {disk_gb:>5.1f}GB | "
              f"{e1:>13.0f}ph | {e2:>13.0f}ph")
    print("\n* disk = train images only (val + Oxford/Paris ~4GB + weights 0.06GB extra).")
    print("* ep = estimated minutes/epoch on T4 (mining+train). Measure real 1 epoch on Kaggle to calibrate.")
    print(f"* source: {src}")


if __name__ == "__main__":
    main()
