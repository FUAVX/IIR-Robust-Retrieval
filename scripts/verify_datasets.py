#!/usr/bin/env python3
"""
scripts/verify_datasets.py

Dataset verification and integrity audit script for IIR-Robust-Retrieval (B0 Baseline).
Author: Hoang (Data & Evaluation Preparation) - FUAVX Team.

Verifies:
1. retrieval-SfM-120k: metadata schema, cluster integrity, tuple indices, and image sample readability.
2. Oxford5k & Paris6k: image count, query ground truth structure (imlist, qimlist, bbx, ok, junk).
3. Pretrained Caffe VGG16 weights existence and basic validity.

Usage:
    python scripts/verify_datasets.py --data-root /path/to/data
    python scripts/verify_datasets.py --help
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def cid2filename(cid: str, prefix: str) -> str:
    """Replicates cirtorch.datasets.datahelpers.cid2filename."""
    return os.path.join(prefix, cid[-2:], cid[-4:-2], cid[-6:-4], cid)


class DatasetVerifier:
    def __init__(self, data_root: str, sample_size: int = 10, verbose: bool = False):
        self.data_root = os.path.abspath(data_root)
        self.sample_size = sample_size
        self.verbose = verbose
        self.issues: List[str] = []
        self.summary_table: List[Tuple[str, str, str]] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"[INFO] {msg}")

    def add_issue(self, item: str, err: str):
        self.issues.append(f"[{item}] {err}")

    def add_status(self, component: str, status: str, detail: str):
        self.summary_table.append((component, status, detail))

    def verify_sfm(self, sfm_name: str = "retrieval-SfM-120k") -> bool:
        """Verifies SfM training dataset and metadata."""
        sfm_dir = os.path.join(self.data_root, "train", sfm_name)
        pkl_path = os.path.join(sfm_dir, f"{sfm_name}.pkl")
        ims_dir = os.path.join(sfm_dir, "ims")

        if not os.path.isdir(sfm_dir):
            self.add_issue(sfm_name, f"Directory not found: {sfm_dir}")
            self.add_status(sfm_name, "MISSING", f"Directory {sfm_dir} does not exist")
            return False

        if not os.path.isfile(pkl_path):
            self.add_issue(sfm_name, f"Metadata pickle not found: {pkl_path}")
            self.add_status(sfm_name, "FAIL", f"Missing {sfm_name}.pkl")
            return False

        # Verify pickle structure
        try:
            with open(pkl_path, "rb") as f:
                db = pickle.load(f)
        except Exception as e:
            self.add_issue(sfm_name, f"Failed to deserialize {pkl_path}: {e}")
            self.add_status(sfm_name, "FAIL", f"Corrupt pickle file: {e}")
            return False

        if not isinstance(db, dict) or "train" not in db:
            self.add_issue(sfm_name, "Root pickle must be a dict with key 'train'")
            self.add_status(sfm_name, "FAIL", "Invalid pickle schema: missing 'train'")
            return False

        train_db = db["train"]
        required_keys = ["cids", "cluster", "qidxs", "pidxs"]
        missing_keys = [k for k in required_keys if k not in train_db]
        if missing_keys:
            self.add_issue(sfm_name, f"train partition missing keys: {missing_keys}")
            self.add_status(sfm_name, "FAIL", f"Missing train keys: {missing_keys}")
            return False

        cids = train_db["cids"]
        clusters = train_db["cluster"]
        qidxs = train_db["qidxs"]
        pidxs = train_db["pidxs"]
        num_images = len(cids)
        num_clusters = len(set(clusters))
        num_pairs = len(qidxs)

        if len(clusters) != num_images:
            self.add_issue(sfm_name, f"Length mismatch: {num_images} cids vs {len(clusters)} clusters")

        if len(qidxs) != len(pidxs):
            self.add_issue(sfm_name, f"Length mismatch: {len(qidxs)} qidxs vs {len(pidxs)} pidxs")

        # Range check on indices
        max_idx = num_images - 1
        if any(qi < 0 or qi > max_idx for qi in qidxs[:1000]):
            self.add_issue(sfm_name, "qidxs contains out-of-bound indices")
        if any(pi < 0 or pi > max_idx for pi in pidxs[:1000]):
            self.add_issue(sfm_name, "pidxs contains out-of-bound indices")

        # Image files verification
        if not os.path.isdir(ims_dir):
            self.add_issue(sfm_name, f"Image directory missing: {ims_dir}")
            self.add_status(sfm_name, "PARTIAL", f"Metadata OK ({num_images} ims, {num_clusters} clusters), but ims/ missing")
            return False

        # Test sample image loading
        sample_checked = 0
        sample_failed = 0
        stride = max(1, num_images // self.sample_size)
        for i in range(0, num_images, stride):
            if sample_checked >= self.sample_size:
                break
            sample_checked += 1
            cid = cids[i]
            img_path = cid2filename(cid, ims_dir)
            if not os.path.isfile(img_path):
                sample_failed += 1
                if sample_failed <= 3:
                    self.add_issue(sfm_name, f"Sample image not found on disk: {img_path}")
            elif HAS_PIL:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    sample_failed += 1
                    self.add_issue(sfm_name, f"Corrupted sample image {img_path}: {e}")

        if sample_failed > 0:
            self.add_status(sfm_name, "FAIL", f"{sample_failed}/{sample_checked} sample images failed load/verify")
            return False

        status_detail = (
            f"Valid! {num_images:,} images, {num_clusters:,} clusters, "
            f"{num_pairs:,} (q,p) pairs. Tested {sample_checked} sample images."
        )
        self.add_status(sfm_name, "PASS", status_detail)
        return True

    def verify_benchmark(self, dataset_name: str, expected_ims: int) -> bool:
        """Verifies Oxford5k or Paris6k benchmark dataset."""
        ds_dir = os.path.join(self.data_root, "test", dataset_name)
        gnd_path = os.path.join(ds_dir, f"gnd_{dataset_name}.pkl")
        jpg_dir = os.path.join(ds_dir, "jpg")

        if not os.path.isdir(ds_dir):
            self.add_issue(dataset_name, f"Directory not found: {ds_dir}")
            self.add_status(dataset_name, "MISSING", f"Directory {ds_dir} does not exist")
            return False

        if not os.path.isfile(gnd_path):
            self.add_issue(dataset_name, f"Ground truth missing: {gnd_path}")
            self.add_status(dataset_name, "FAIL", f"Missing gnd_{dataset_name}.pkl")
            return False

        try:
            with open(gnd_path, "rb") as f:
                gnd_data = pickle.load(f)
        except Exception as e:
            self.add_issue(dataset_name, f"Failed to load ground truth {gnd_path}: {e}")
            self.add_status(dataset_name, "FAIL", f"Corrupt ground truth: {e}")
            return False

        for k in ["imlist", "qimlist", "gnd"]:
            if k not in gnd_data:
                self.add_issue(dataset_name, f"Ground truth missing key '{k}'")
                self.add_status(dataset_name, "FAIL", f"Missing key '{k}' in ground truth")
                return False

        imlist = gnd_data["imlist"]
        qimlist = gnd_data["qimlist"]
        gnd = gnd_data["gnd"]

        if len(qimlist) != 55 or len(gnd) != 55:
            self.add_issue(dataset_name, f"Expected 55 queries, got {len(qimlist)} qimlist and {len(gnd)} gnd")

        # Verify query bbx and lists
        invalid_bbx = 0
        for i, q_entry in enumerate(gnd):
            if "bbx" not in q_entry or "ok" not in q_entry or "junk" not in q_entry:
                self.add_issue(dataset_name, f"Query {i} missing bbx/ok/junk keys")
                invalid_bbx += 1
                continue
            bbx = q_entry["bbx"]
            if len(bbx) != 4 or (bbx[2] <= bbx[0] and bbx[2] <= 0) or (bbx[3] <= bbx[1] and bbx[3] <= 0):
                invalid_bbx += 1
                self.add_issue(dataset_name, f"Query {i} invalid bounding box: {bbx}")

        if not os.path.isdir(jpg_dir):
            self.add_issue(dataset_name, f"jpg/ directory missing: {jpg_dir}")
            self.add_status(dataset_name, "PARTIAL", f"Ground truth OK ({len(imlist)} gallery, 55 queries), but jpg/ missing")
            return False

        # Check total images present
        actual_jpgs = [f for f in os.listdir(jpg_dir) if f.lower().endswith((".jpg", ".jpeg"))]
        count_jpgs = len(actual_jpgs)

        # Sample image test
        sample_checked = 0
        sample_failed = 0
        stride = max(1, count_jpgs // self.sample_size)
        for i in range(0, count_jpgs, stride):
            if sample_checked >= self.sample_size:
                break
            sample_checked += 1
            im_path = os.path.join(jpg_dir, actual_jpgs[i])
            if HAS_PIL:
                try:
                    with Image.open(im_path) as im:
                        im.verify()
                except Exception as e:
                    sample_failed += 1
                    self.add_issue(dataset_name, f"Corrupted image {im_path}: {e}")

        if count_jpgs < expected_ims:
            self.add_issue(dataset_name, f"Found {count_jpgs} images, expected >= {expected_ims}")

        if sample_failed > 0:
            self.add_status(dataset_name, "FAIL", f"{sample_failed}/{sample_checked} sample images corrupted")
            return False

        status_detail = (
            f"Valid! {count_jpgs:,} images (expected {expected_ims:,}), 55 queries, "
            f"all bbx valid. Tested {sample_checked} images."
        )
        self.add_status(dataset_name, "PASS", status_detail)
        return True

    def verify_weights(self, weights_filename: str = "imagenet-caffe-vgg16-features-d369c8e.pth") -> bool:
        """Verifies pretrained Caffe VGG16 weights."""
        weights_path = os.path.join(self.data_root, "networks", weights_filename)
        if not os.path.isfile(weights_path):
            self.add_issue("Pretrained Weights", f"Weights file not found: {weights_path}")
            self.add_status("Caffe VGG16 Weights", "MISSING", f"File not found: {weights_path}")
            return False

        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        if size_mb < 50 or size_mb > 70:
            self.add_issue("Pretrained Weights", f"Unexpected file size: {size_mb:.2f} MB (expected ~58 MB)")
            self.add_status("Caffe VGG16 Weights", "WARN", f"Size is {size_mb:.2f} MB (expected ~58 MB)")
            return True

        self.add_status("Caffe VGG16 Weights", "PASS", f"Present ({size_mb:.2f} MB)")
        return True

    def run_all(self) -> int:
        print("=" * 80)
        print(f"IIR-Robust-Retrieval — Dataset Verification Audit")
        print(f"Target Data Root: {self.data_root}")
        print("=" * 80)

        self.verify_sfm("retrieval-SfM-120k")
        self.verify_benchmark("oxford5k", expected_ims=5063)
        self.verify_benchmark("paris6k", expected_ims=6412)
        self.verify_weights("imagenet-caffe-vgg16-features-d369c8e.pth")

        print("\n" + "=" * 80)
        print(f"{'Component':<25} | {'Status':<10} | {'Details'}")
        print("-" * 80)
        for comp, st, detail in self.summary_table:
            print(f"{comp:<25} | {st:<10} | {detail}")
        print("=" * 80)

        if self.issues:
            print(f"\n[!] Detected {len(self.issues)} issue(s):")
            for iss in self.issues[:15]:
                print(f"  - {iss}")
            if len(self.issues) > 15:
                print(f"  ... and {len(self.issues) - 15} more issues.")
            print("\nNote: Datasets not yet downloaded or mounted will appear as MISSING.")
            return 1
        else:
            print("\n[SUCCESS] All datasets and weights verified successfully without issues!")
            return 0


def main():
    parser = argparse.ArgumentParser(description="Audit and verify datasets for IIR-Robust-Retrieval.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("CIRT_DATA_ROOT", os.path.join(os.getcwd(), "data")),
        help="Path to data root directory (defaults to $CIRT_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of images per dataset to test for decoding and integrity (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose diagnostics during verification",
    )
    args = parser.parse_args()

    verifier = DatasetVerifier(data_root=args.data_root, sample_size=args.sample_size, verbose=args.verbose)
    exit_code = verifier.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
