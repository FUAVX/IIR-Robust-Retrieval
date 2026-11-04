# B0 Dataset Manifest and Preparation Guide

**Author / Responsible:** Hoang (Data & Evaluation Preparation)  
**Team:** FUAVX (Phuong · Nguyen · Hoang)  
**Branch:** `b0-data-preparation`  
**Status:** Initial specification and verification guide for B0 reproduction  

---

## 1. Overview and Principles

Baseline B0 aims to faithfully reproduce the GeM retrieval pipeline from *Fine-tuning CNN Image Retrieval with No Human Annotation* (Radenovic, Tolias, Chum, TPAMI 2018).

To guarantee strict reproducibility and avoid experimental errors, data preparation follows three core principles:

1. **No Cluster Leakage:** The training supervision in `retrieval-SfM-120k` is built on 3D Structure-from-Motion (SfM) clusters. Never randomly split images between train, validation, or test sets. Any dataset subsetting must be done **strictly by entire cluster IDs**.
2. **Benchmark Protocol Fidelity:** Initial B0 evaluation strictly uses the **original/legacy Oxford5k and Paris6k** protocols (with query bounding box cropping, positive image ranking, and junk image exclusion), **not** the revisited (ROxford5k/RParis6k) protocols.
3. **No Heavy Artifacts in Git:** Image archives, extracted JPEG files, metadata pickles, and `.pth` weight files must remain outside the Git repository. They are managed via `.gitignore`, local cache, or read-only mounts under Kaggle `/kaggle/input/`.

---

## 2. Dataset Inventory and Provenance

### 2.1 Training & Validation: `retrieval-SfM-120k`

| Attribute | Details |
| :--- | :--- |
| **Name** | `retrieval-SfM-120k` (and optional 30k derivative) |
| **Purpose** | Siamese training with contrastive loss & hard-negative mining |
| **Source Host** | Faculty of Electrical Engineering, Czech Technical University (CMP FELK CVUT) |
| **Image Archive URL** | `http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/ims.tar.gz` |
| **Metadata DB URL** | `http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-120k.pkl` |
| **Whitening DB URL** | `http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/retrieval-SfM-120k-whiten.pkl` |
| **Archive Size** | ~11.5 GB compressed (`ims.tar.gz`) |
| **Extracted Size** | ~13–15 GB (~121,142 images) |
| **Directory Structure** | Images are organized in 4-character hex subdirectories based on image CID:<br>`ims/<hex1>/<hex2>/<hex3>/<hex4>/<CID>.jpg` |

#### Metadata Schema (`retrieval-SfM-120k.pkl`):
The file is a serialized Python dictionary with top-level keys:
- `'train'`: Dictionary containing:
  - `cids` (list of str): Unique identifier for each image. Filename resolved via `cirtorch.datasets.datahelpers.cid2filename(cid, ims_root)`.
  - `cluster` (list of int/str): Cluster ID assigned to each image from 3D SfM reconstruction.
  - `qidxs` (list of int): Pool of indices into `cids` designated as training queries.
  - `pidxs` (list of int): Positive matching image index for each query in `qidxs`.
- `'val'`: Dictionary containing corresponding validation partitions for queries, positives, and candidate clusters.

---

### 2.2 Evaluation Benchmark 1: `Oxford5k`

| Attribute | Details |
| :--- | :--- |
| **Name** | `oxford5k` (Oxford Buildings Dataset) |
| **Purpose** | Landmark retrieval test benchmark |
| **Source Host** | Visual Geometry Group (VGG), University of Oxford |
| **Image Archive URL** | `https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz` |
| **Ground Truth URL** | `http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/oxford5k/gnd_oxford5k.pkl` |
| **Archive Size** | ~1.9 GB |
| **Image Count** | 5,063 images (55 query landmarks across 11 buildings) |
| **Target Directory** | `<DATA_ROOT>/test/oxford5k/jpg/` |

#### Ground Truth Schema (`gnd_oxford5k.pkl`):
Dictionary with keys:
- `imlist` (list of str): List of 5,063 image base names (without `.jpg`).
- `qimlist` (list of str): List of 55 query image names.
- `gnd` (list of dicts, length 55): For each query:
  - `bbx` (tuple/list of 4 floats/ints): Query region bounding box `[x0, y0, x1, y1]` or `[x, y, w, h]` used to crop the query before descriptor extraction.
  - `ok` (list of int): Indices into `imlist` containing positive ground-truth images.
  - `junk` (list of int): Indices into `imlist` containing ambiguous/occluded images (must be removed from rank list before computing AP).

---

### 2.3 Evaluation Benchmark 2: `Paris6k`

| Attribute | Details |
| :--- | :--- |
| **Name** | `paris6k` (Paris Dataset) |
| **Purpose** | Landmark retrieval test benchmark |
| **Source Host** | Visual Geometry Group (VGG), University of Oxford |
| **Image Archives URL**| Part 1: `https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz`<br>Part 2: `https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz` |
| **Ground Truth URL** | `http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/paris6k/gnd_paris6k.pkl` |
| **Archive Size** | ~2.0 GB (Part 1: ~1.1 GB, Part 2: ~0.9 GB) |
| **Image Count** | 6,412 images (55 query landmarks across 11 Paris buildings) |
| **Target Directory** | `<DATA_ROOT>/test/paris6k/jpg/` |

#### Ground Truth Schema (`gnd_paris6k.pkl`):
Structure identical to `gnd_oxford5k.pkl` (`imlist`, `qimlist`, `gnd` with `bbx`, `ok`, `junk`).

---

### 2.4 Pretrained Backbone Weights: Custom Caffe VGG16

| Attribute | Details |
| :--- | :--- |
| **File Name** | `imagenet-caffe-vgg16-features-d369c8e.pth` |
| **Source Host** | CMP FELK CVUT: `http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/` |
| **Target Directory** | `<DATA_ROOT>/networks/` |
| **File Size** | ~58 MB |
| **Critical Requirement** | **Do NOT replace with torchvision standard VGG16 weights.** The original GeM paper relies on weights converted directly from Caffe ImageNet models (which use BGR color order and caffe mean centering `[103.939, 116.779, 123.68]`). Using standard PyTorch ImageNet weights without tuning alters the feature geometry and invalidates exact reproduction. |

---

## 3. Directory Layout Specification

### 3.1 Standard Local Layout (`DATA_ROOT`)

```text
data/
├── networks/
│   └── imagenet-caffe-vgg16-features-d369c8e.pth
├── train/
│   └── retrieval-SfM-120k/
│       ├── retrieval-SfM-120k.pkl
│       ├── retrieval-SfM-120k-whiten.pkl
│       └── ims/
│           ├── 0/ ...
│           └── ...
└── test/
    ├── oxford5k/
    │   ├── gnd_oxford5k.pkl
    │   └── jpg/
    │       ├── all_souls_000000.jpg
    │       └── ...
    └── paris6k/
        ├── gnd_paris6k.pkl
        └── jpg/
            ├── defense_000000.jpg
            └── ...
```

---

## 4. Kaggle Integration Strategy (Handoff to Phuong for C1)

In Kaggle environments:
1. `/kaggle/input/` is strictly **read-only**.
2. `/kaggle/working/` is writable for checkpoints, run logs, and outputs.

### 4.1 Recommended Kaggle Mount Options

**Option A (Separate Dataset Mounts - Recommended):**
- `/kaggle/input/retrieval-sfm-120k/` -> contains `train/retrieval-SfM-120k/`
- `/kaggle/input/oxford5k/` -> contains `test/oxford5k/`
- `/kaggle/input/paris6k/` -> contains `test/paris6k/`
- `/kaggle/input/gem-caffe-weights/` -> contains `imagenet-caffe-vgg16-features-d369c8e.pth`

**Option B (Single Unified Mount):**
- `/kaggle/input/iir-robust-data/` structured matching the standard `DATA_ROOT` above.

### 4.2 Decoupling Requirement for C1
`cirtorch/utils/general.py:get_data_root()` currently hardcodes `<repo_root>/data`.  
For C1, Phuong should introduce:
- An environment variable `CIRT_DATA_ROOT` or CLI argument `--data-root` that defaults to `get_data_root()`, but allows overriding to `/kaggle/input/...`.
- If datasets are split across multiple input folders on Kaggle, support `--train-data-root`, `--test-data-root`, and `--weights-path`.

---

## 5. Verification Tools and Checklist

A dedicated verification script is located at [`scripts/verify_datasets.py`](../../scripts/verify_datasets.py).

### Verification Checklist (verified 2026-10-10, exp01 — all PASS, see `experiments/B0/exp01-hoang-full-data.md`):
- [x] **SfM-120k Metadata:** `retrieval-SfM-120k.pkl` loadable with pickle; keys `'train'` and `'val'` present.
- [x] **SfM-120k Images:** Image paths resolved by `cid2filename` exist and sample images can be read by PIL without decompression errors.
- [x] **Oxford5k:** 5,063 JPEG images present in `jpg/`; 55 queries in `gnd_oxford5k.pkl`; all `bbx` have positive area; positive images exist in `imlist`.
- [x] **Paris6k:** 6,412 JPEG images present in `jpg/`; 55 queries in `gnd_paris6k.pkl`; all `bbx` have positive area; positive images exist in `imlist`.
- [x] **Pretrained Weights:** `imagenet-caffe-vgg16-features-d369c8e.pth` present and non-empty.
