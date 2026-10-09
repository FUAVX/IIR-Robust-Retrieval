#!/usr/bin/env python3
"""
scripts/create_sfm_subset.py

Utility to create a cluster-safe subset of retrieval-SfM-120k.
Author: Hoang (Data & Evaluation Preparation) - FUAVX Team.

RULE: No random image splitting! Any subsetting MUST preserve entire 3D SfM clusters
to prevent data leakage and maintain valid hard-negative mining / contrastive supervision.

Usage:
    python3 scripts/create_sfm_subset.py \
        --input-pkl data/train/retrieval-SfM-120k/retrieval-SfM-120k.pkl \
        --output-pkl data/train/retrieval-SfM-120k/retrieval-SfM-subset.pkl \
        --num-clusters 50 \
        --seed 42
"""

import argparse
import os
import pickle
import random
import sys
from typing import Dict, List, Set


def create_subset_for_partition(partition_data: Dict, num_clusters: int, seed: int) -> Dict:
    """Subsets a single partition ('train' or 'val') preserving complete clusters."""
    cids: List[str] = partition_data["cids"]
    clusters: List = partition_data["cluster"]
    qidxs: List[int] = partition_data["qidxs"]
    pidxs: List[int] = partition_data["pidxs"]

    # Group image indices by cluster
    cluster_to_indices: Dict = {}
    for idx, cl in enumerate(clusters):
        cluster_to_indices.setdefault(cl, []).append(idx)

    unique_clusters = list(cluster_to_indices.keys())
    total_clusters = len(unique_clusters)

    if num_clusters >= total_clusters:
        print(f"[!] Requested {num_clusters} clusters >= total {total_clusters}. Keeping all.")
        return partition_data

    # Sample clusters deterministically
    rng = random.Random(seed)
    chosen_clusters: Set = set(rng.sample(unique_clusters, num_clusters))

    # Collect all image indices belonging to chosen clusters
    kept_old_indices: List[int] = []
    for cl in chosen_clusters:
        kept_old_indices.extend(cluster_to_indices[cl])
    kept_old_indices.sort()

    old_to_new_idx: Dict[int, int] = {old_i: new_i for new_i, old_i in enumerate(kept_old_indices)}
    kept_set: Set[int] = set(kept_old_indices)

    # Subet cids and cluster
    subset_cids = [cids[i] for i in kept_old_indices]
    subset_cluster = [clusters[i] for i in kept_old_indices]

    # Filter qidxs and pidxs: keep only pairs where both query and positive are in chosen clusters
    subset_qidxs = []
    subset_pidxs = []
    for q, p in zip(qidxs, pidxs):
        if q in kept_set and p in kept_set:
            subset_qidxs.append(old_to_new_idx[q])
            subset_pidxs.append(old_to_new_idx[p])

    print(
        f"  - Clusters: {len(chosen_clusters)} / {total_clusters} "
        f"({len(chosen_clusters)/total_clusters*100:.1f}%)"
    )
    print(
        f"  - Images:   {len(subset_cids):,} / {len(cids):,} "
        f"({len(subset_cids)/len(cids)*100:.1f}%)"
    )
    print(
        f"  - (q, p):   {len(subset_qidxs):,} / {len(qidxs):,} "
        f"({len(subset_qidxs)/len(qidxs)*100:.1f}%)"
    )

    return {
        "cids": subset_cids,
        "cluster": subset_cluster,
        "qidxs": subset_qidxs,
        "pidxs": subset_pidxs,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate cluster-preserved subset of retrieval-SfM.")
    parser.add_argument("--input-pkl", type=str, required=True, help="Path to retrieval-SfM-120k.pkl")
    parser.add_argument("--output-pkl", type=str, required=True, help="Path to save subset pkl")
    parser.add_argument("--num-clusters", type=int, default=50, help="Number of clusters to keep in train partition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible cluster sampling")
    args = parser.parse_args()

    if not os.path.isfile(args.input_pkl):
        print(f"[ERROR] Input pickle not found: {args.input_pkl}")
        sys.exit(1)

    print(f"Loading {args.input_pkl}...")
    with open(args.input_pkl, "rb") as f:
        db = pickle.load(f)

    subset_db = {}
    for part in ["train", "val"]:
        if part in db:
            print(f"\nProcessing partition '{part}':")
            # For validation, maintain proportional clusters or keep as-is if small
            k = args.num_clusters if part == "train" else max(10, args.num_clusters // 5)
            subset_db[part] = create_subset_for_partition(db[part], num_clusters=k, seed=args.seed)
        else:
            print(f"[!] Warning: partition '{part}' not found in database.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pkl)), exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(subset_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n[SUCCESS] Subset saved successfully to: {args.output_pkl}")


if __name__ == "__main__":
    main()
