# External artifact access

## Training checkpoints

The three original ZIPs are in the shared Drive folder below. Match the exact
filename and archive SHA-256 before loading a checkpoint.

| Artifact ngoài Git | SHA-256 của file lưu trữ | Access URL |
| --- | --- | --- |
| `B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip` — B0 epoch 12, chứa `best.pt`, history và evaluation JSON | `d58bcaa735f2dde7eac132e38c8726882166ea00589f1fa45581469ee8fa2f73` | [Google Drive folder](https://drive.google.com/drive/folders/16lsh6PdOzAlSKRmuTt7iCcIHH0kDMTxY) |
| `run1_vgg16_gem_augmentation_v2-20261030T082558Z-1-002.zip` — R1 augmentation | `76c52b0d73eaa2e87df97f3b3fadfe9ad7726a93031c9eb75a81ad8abac760a8` | [Google Drive folder](https://drive.google.com/drive/folders/16lsh6PdOzAlSKRmuTt7iCcIHH0kDMTxY) |
| `run2_attention_bf16_100ep_v1-20261030T082559Z-1-001.zip` — R2 augmentation + gate | `1b89fb9dcdff9caa9b589e069e497602d55fe39f1d435b0ec911791bd74321fb` | [Google Drive folder](https://drive.google.com/drive/folders/16lsh6PdOzAlSKRmuTt7iCcIHH0kDMTxY) |

The Drive filenames were documented as visible on 2026-11-05; this revision
uses the local copies and does not claim a new independent Drive download.
See `reproduction/B0/epoch_012/README.md` for B0 checkpoint provenance and
`paper/ieee/data/models_manifest.json` for R1/R2 identities.

## Raw post-training numerical outputs — public download

[Download the historical numerical outputs on Kaggle](https://www.kaggle.com/datasets/ndhoang2310/iir-post100-raw-numerical-outputs-v1).

The inventory lists **5,992 original files** (1,025,514,236 payload bytes):
per-query AP, aggregate and sensitivity CSVs, complete ranking NPZs,
gallery/query descriptors, graph matrices, convergence diagnostics and timing
records. These are the historical epoch-100 outputs, including dense-seed
diffusion. Revised sparse-seed measurements are committed separately under
`artifacts/revision-validation/results/`.

Kaggle expanded the uploaded tar.gz. Download the dataset, then place the
contents of `post100-numerical-outputs/` under
`artifacts/kaggle-posttraining/outputs/`, preserving each job's `unpacked/`
directory. `raw_inventory.json` gives the original relative paths, byte counts
and SHA-256 values. The local upload archive was fully read back and verified:
909,081,588 bytes, SHA-256
`f7e1bb3108977d86b986bbe5e125f107b0ff6df70286e7718d177e8f146f044d`.
The Kaggle download is a different container; validate its extracted files
against the inventory rather than comparing its ZIP hash with this tar hash.

Access checked on **2026-11-11 without credentials or cookies**: the inventory
matches the local original byte-for-byte. A ranking NPZ, an Oxford descriptor
NPZ and the report summary CSV were downloaded publicly and matched their
recorded hashes. This is a sampled remote check; every remote file was not
downloaded again. The machine-readable record is
[`public_raw_verification.json`](revision-validation/results/public_raw_verification.json).

The release omits duplicate tar transport, rebuildable SQLite feature/matching
caches, example benchmark images, logs and model weights. It preserves all
unpacked numerical CSV/JSON/NPZ/MD outputs. Rebuilding local caches uses the
published code and original benchmark images.

## Input datasets and source

| Input | Access URL |
| --- | --- |
| Prepared SfM subset-100 | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm100-prepared) |
| Oxford5k and Paris6k benchmarks | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-benchmarks) |
| Caffe VGG16 feature weights | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-vgg16-weights) |
| Portable B0 training handoff | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-training-handoff) |

These are the input locations recorded by the historical handoff. Caffe VGG16
feature weights have SHA-256
`d369c8e593250e421eda192512263dccd35edb5c4b57e97312651bac10a51b7d`.
The training/search source and independent diffusion checks are linked from
[`revision-validation/README.md`](revision-validation/README.md).
