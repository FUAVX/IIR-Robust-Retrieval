# exp01 — Hoang: full B0 data preparation (verified 4/4 PASS)

Data-preparation record (no training in this experiment).
Template: `report/B0/B0_results.md`. `TBD` = training-side, owned by Phuong/Nguyen.

| Field | Value |
| --- | --- |
| Experiment ID | exp01-hoang-full-data |
| Git commit | 1299d30 (+ untracked `scripts/download_datasets_local.py`, `scripts/profile_sfm.py`; to be committed with this record) |
| Branch and working-tree state | `b0-data-preparation`; `data/` + `*.tgz` ignored by `.gitignore`, outside Git |
| Environment | Ubuntu (Acer Aspire A315-56), Python 3.12.3, PIL + numpy; PyTorch NOT required for data prep — TBD (Kaggle env recorded at train time) |
| GPU | N/A (local CPU preparation; train target: Kaggle T4 16GB) |
| Dataset/subset | FULL: `retrieval-SfM-120k` train 91,642 imgs / 551 clusters / 181,697 (q,p) pairs + val 6,403 imgs / 162 clusters / 1,691 pairs; `oxford5k` 5,063 imgs / 55 queries; `paris6k` 6,412 imgs / 55 queries; Caffe VGG16 weights 56.14 MB. Local: `./data/` (layout per `dataset_manifest.md` §3) |
| Number of training clusters | FULL = 551 (train). Subset study (seed 42): k=25 → 6,399 imgs/10,734 pairs; k=50 → 9,662/20,012; k=100 → 15,953/34,339; k=200 → 35,767/69,692. All mining-viable (≥5 neg clusters). Recommendation: FULL for B0 train; k=100 as baseline-lite fallback |
| Queries per epoch | TBD (train; proposed: qsize 1000 for imsize 768) |
| Tuple configuration | Agreed: q + p + 5 negatives (7 images), TEAM B0 |
| Mining configuration | Agreed: hard-negative mining; pool size TBD (proposed 10k for imsize 768). Refresh each epoch via `create_epoch_tuples()` |
| Optimizer | TBD (train) |
| Learning rate | TBD (train) |
| Weight decay | TBD (train) |
| Scheduler | TBD (train) |
| Epochs (planned and completed) | Data prep only: 0 train epochs |
| Model-selection criterion | TBD (train) |
| Image size, batch size, preprocessing, init, loss margin | Proposed for train: imsize 768 (fallback 512), rest TBD |
| Seed/randomness policy | Subset sampling seed 42 (`random.Random(42)`, sorted cluster IDs). Provenance: `data/train/retrieval-SfM-120k/subset_provenance_k*_seed42.txt` |
| Exact configuration/command reference | `python3 scripts/download_datasets_local.py --only benchmarks` → `--only sfm-meta --num-clusters 100 --seed 42` → `--only sfm-images --full` (aria2 -x8) → `python3 scripts/profile_sfm.py --data-root ./data --seed 42` → `python3 scripts/verify_datasets.py --data-root ./data --sample-size 20 --verbose` |
| Runtime | Oxford ~15 min; Paris ~5 min; weights/pkl slow single-conn (10–40 KB/s); ims.tar.gz 37.7 GB via aria2 -x8 @~15–20 MB/s ≈ 35 min; full extract ~10 min |
| Checkpoint | N/A (data prep) |
| Evaluation protocol | Legacy `oxford5k` + `paris6k` (gnd keys `ok/junk`, NOT revisited `easy/hard`), single-scale `[1]`, whitening OFF, QE OFF — for Nguyen C4 |
| Oxford5k mAP | TBD (train/eval) |
| Paris6k mAP | TBD (train/eval) |

## Deviations from paper

- FULL local data (not a subset) prepared for B0; subset ladder (25/50/100/200) studied for time-budgeted fallback. Subset configs are TEAM B0 deviations to be recorded per run; they trade cluster diversity for epoch time, comparability to full-paper numbers is approximate.
- Proposed train imsize 768/512 (vs source default 1024) for T4 12h-session fit — to be confirmed by Phuong with timing evidence.

## Compatibility changes

- `cirtorch/datasets/datahelpers.py`: `Image.ANTIALIAS` → `Image.Resampling.LANCZOS` (Pillow ≥10 removal). Pre-existing in branch (C0).
- `cirtorch/utils/evaluate.py`: `np.in1d` → `np.isin` (NumPy deprecation). Pre-existing in branch (C0).
- `scripts/download_datasets_local.py`: removed GNU-tar `--checkpoint-action` (not portable; broke full extract with exit 2). Validation: full 117,369-file extract succeeded after removal.
- Env-only (no repo change): `aria2c -x8` required for `cmp.felk.cvut.cz` (single-connection throttled to ~KB/s; 8 connections ≈ 15–20 MB/s).

## Observations

- Per-connection throttle measured: curl single-conn ~740 B/s over 44 min (15 files); aria2 8-conn ~15–20 MB/s same host/same file. HEAD/`Accept-Ranges: bytes` OK (server alive, 40,528,074,099 B).
- Cluster-size skew: train max 2,082 / median 68 / mean 166.3 — random subset draws vary strongly; stratified sampling recommended for future subset work.
- Subset epoch-time finding: epoch cost depends on (qsize, poolsize, imsize), NOT on k, as long as subset covers qsize/poolsize (even k=25 has 10,734 pairs ≥ qsize 1000). Decision metric is diversity + disk, not epoch minutes.
- Full archive holds 117,369 files vs 98,045 metadata cids (train+val) — ~19k extra files outside supervision; harmless for training (loader follows pkl).

## Failures

1. `curl | tar` stream stall (0 bytes/60 s) — cause: per-conn throttle; fix: aria2 download-then-extract. No data lost.
2. Frankenstein `retrieval-SfM-120k.pkl` (`UnpicklingError: invalid load key, '7'`) — cause: `wget -c`/aria2 resume onto locally-subsetted pkl; fix: restored from `retrieval-SfM-120k-full.pkl` backup, script now never resumes pkl (fresh download or backup restore).
3. `tar --checkpoint-action` exit 2 — cause: non-portable flag; fix: plain `tar -xzf`, progress via file count.
4. `tail -f /tmp/full_imgs.log` "not found" confusion — cause: opened too early; file existed from 22:39. No action.

## Artifact references

- Local data (outside Git): `./data/` — `train/retrieval-SfM-120k/{retrieval-SfM-120k.pkl,retrieval-SfM-120k-full.pkl,retrieval-SfM-120k-whiten.pkl,needed_files.txt,subset_provenance_*.txt,ims/ (117,369 files)}`, `test/{oxford5k,paris6k}/{gnd_*.pkl,jpg/}`, `networks/imagenet-caffe-vgg16-features-d369c8e.pth`.
- Logs: `/tmp/full_imgs.log` (aria2 37.7 GB), `/tmp/full_extract2.log` (empty = clean extract).
- Verify output: 4/4 PASS (SfM 91,642/551/181,697; Oxford 5,063; Paris 6,412; weights 56.14 MB), 20 samples each.
- Scripts (in Git after commit): `scripts/download_datasets_local.py` (`--full` mode), `scripts/profile_sfm.py`.
