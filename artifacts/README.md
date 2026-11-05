# External artifact access

Ba ZIP trong bảng dưới dùng chung thư mục Google Drive do chủ repo cung cấp.
Trước khi chạy lại từ checkpoint, tải đúng tên file và đối chiếu SHA-256.
Kiểm tra quyền truy cập bằng tài khoản mà giảng viên sẽ dùng. Không đưa API
token, mật khẩu, checkpoint hoặc ZIP lớn vào Git.

| Artifact ngoài Git | SHA-256 của file lưu trữ | Access URL |
| --- | --- | --- |
| `B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip` — B0 epoch 12, chứa `best.pt`, history và evaluation JSON | `d58bcaa735f2dde7eac132e38c8726882166ea00589f1fa45581469ee8fa2f73` | [Google Drive folder](https://drive.google.com/drive/folders/16lsh6PdOzAlSKRmuTt7iCcIHH0kDMTxY) |
| `run1_vgg16_gem_augmentation_v2-20261030T082558Z-1-002.zip` — R1 augmentation | `76c52b0d73eaa2e87df97f3b3fadfe9ad7726a93031c9eb75a81ad8abac760a8` | [Google Drive folder](https://drive.google.com/drive/folders/16lsh6PdOzAlSKRmuTt7iCcIHH0kDMTxY) |
| `run2_attention_bf16_100ep_v1-20261030T082559Z-1-001.zip` — R2 augmentation + gate | `1b89fb9dcdff9caa9b589e069e497602d55fe39f1d435b0ec911791bd74321fb` | [Google Drive folder](https://drive.google.com/drive/folders/16lsh6PdOzAlSKRmuTt7iCcIHH0kDMTxY) |

Drive access check — 2026-11-05: the supplied folder returned HTTP 200 without
credentials or cookies, with title `SEG Result - Google Drive`. All three exact
ZIP filenames in the table were visible. The links open the shared folder;
select the corresponding ZIP by its full filename. ZIP downloads and SHA-256
values have not been independently checked, and access from the instructor's
account has not been tested. The raw-output download location remains pending.

The [B0 evidence index](../reproduction/B0/epoch_012/README.md) describes the
first ZIP and its checkpoint hash. The [paper source manifest](../paper/ieee/data/source_manifest.json)
records the R1/R2 archive identities; the paper's aggregate CSV/JSON snapshots
are already committed under [`paper/ieee/data/`](../paper/ieee/data/). To repeat
raw post-training inference or inspect every saved ranking, the team must also
provide the corresponding raw outputs or their own verified download location.
Add that location here when available: **PENDING_URL_POSTTRAINING_RAW_OUTPUTS**.

## Artifact location audit — 2026-11-04

Reviewed all 33 commits reachable from the available local branches,
remote-tracking refs and reflogs, including `main`, `b0-data-preparation`,
`b0-reproduction-evidence`, `b1-robustness-experiments` and
`training-evaluation-report`. This included historical README files, scripts,
notebook source/saved outputs, PDF text/hyperlinks and committed source ZIPs.
No download URL identifying any of the four pending artifacts was found during
that repository search. On 2026-11-05, the repository owner supplied the Drive
folder now linked for B0/R1/R2 above; the raw-output placeholder remains pending.
The location-evidence table below preserves the earlier repository findings.

| Artifact | Location evidence | Verification / remaining gap |
| --- | --- | --- |
| B0 | [Evidence index](../reproduction/B0/epoch_012/README.md), [checkpoint inventory](../reproduction/B0/epoch_012/results/checkpoint_inventory.csv) and [B1 handoff](../docs/B1/00_README_BAN_GIAO.md) identify historical `reproduce/output_kaggle/` paths and local `C:\Users\DELL\Downloads\iir.zip`. | The local `iir.zip` exists; its checkpoint, history and epoch-12 evaluation match the [analysis manifest](../reproduction/B0/epoch_012/results/analysis_manifest.json). Its archive hash differs from the required B0 ZIP; no share/download URL was found. |
| R1 | [Source manifest](../paper/ieee/data/source_manifest.json) identifies the exact ZIP name and hash; [model manifest](../paper/ieee/data/models_manifest.json) identifies the epoch-100 checkpoint. | Archive identity is documented, but its storage path, byte count and download URL were not found. |
| R2 | [Source manifest](../paper/ieee/data/source_manifest.json) identifies the exact ZIP name and hash; [model manifest](../paper/ieee/data/models_manifest.json) identifies the epoch-100 checkpoint. | Archive identity is documented, but its storage path, byte count and download URL were not found. |
| Raw post-training outputs | [Evidence manifest](../paper/ieee/evidence_manifest.json) and [example builder](../paper/ieee/build_examples.py) refer to `artifacts/kaggle-posttraining/outputs/`. | Historical local paths only; this directory is absent from the inspected worktrees. No raw-output archive identity or download URL was found. |

The inspected `iir.zip` is 558,489,051 bytes, with SHA-256
`75f0154dfdcfdeffef8597c86d4b6262da1689a6374f86dfc7e84f67d69d167f`.
Its `b0_sfm100_notebook/runs/b0_sfm100_vgg16_gem/best.pt` is 176,631,073 bytes
and has the expected checkpoint SHA-256
`8a56bad2e1d585c68de42ff841f061e5469f76792750d8b4993e500294f19f27`.
`history.json` and `evaluation_epoch_012.json` also match their manifest hashes.
This verifies those B0 contents, not the identity of the ZIP requested above.

Raw-output recovery should follow these paths relative to the historical
`artifacts/kaggle-posttraining/outputs/` directory:

- `iir-post100-report/unpacked/`: `per_query.csv`, `conditions.csv`,
  `end_to_end_summary.csv` and `summary.csv`.
- `iir-post100-extract/unpacked/desc_run1_aug_epoch100_oxford5k.npz`:
  descriptors consumed by the example builder.
- `iir-post100-geometry-oxford5k-*/unpacked/results/`: JSON results and matching
  NPZ rankings consumed by that builder. The wildcard is a local file pattern,
  not a Kaggle notebook identifier or URL.

These exact paths are the example builder's minimum inputs, not a complete
inventory of the post-training suite. Its `per_query.csv` reader requires
`model`, `dataset`, `name`, `phase`, `method`, `query_id` and `ap`; the descriptor
NPZ reader uses `gallery_ids`, `conditions`, `gallery` and `queries`. Each result
JSON identifies the model, method, phase and condition; its same-stem NPZ
contains `query_ids` and `ranks`. CSV summaries cannot replace these rankings.
The [paper's coverage description](../paper/ieee/sections/results.tex) and source
manifest report 68,200 main per-query AP records plus 5,500 sensitivity records
(73,700 total), covering both epoch-100 encoders, fusion and both benchmarks.
Recover the whole historical `outputs/` directory and its job manifests;
the checked-in example only selects R1/Oxford examples. The missing job source
and output inventories prevent an exact enumeration of every raw-result file.

An expanded local search also inspected readable files under the workspace,
Windows Downloads, Documents/Codex and OneDrive, and the member names of 160 ZIPs
under the workspace, Downloads and Documents/Codex. It found no additional
matching checkpoint/raw-output bundle. Downloads contains an
`evidence_manifest.json` with the same seven source hashes, but no download URL.
The historical `/home/dacekey/SEG301_Fall26/research` root and the Kaggle/Colab
mount paths are absent here. Some unrelated package/database directories were
unreadable; the search does not establish absence outside the readable scope.

Download the exact three ZIPs from the supplied Drive folder and compare their
SHA-256 values with the first table. For raw outputs, obtain the complete saved
output bundle and record its filename, byte count and SHA-256. Supply its share
link and verify downloads using the recipient's account before replacing the
remaining placeholder.

## Public Kaggle inputs documented by the handoff

| Input | Access URL |
| --- | --- |
| Prepared SfM subset-100 | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm100-prepared) |
| Oxford5k and Paris6k benchmarks | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-benchmarks) |
| Caffe VGG16 feature weights | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-vgg16-weights) |
| Portable B0 training handoff bundle | [Kaggle dataset](https://www.kaggle.com/datasets/ndhoang2310/iir-b0-training-handoff) |

These are the locations recorded in the [Kaggle handoff](kaggle-training/public-handoff/HANDOFF.md).
Check access from the recipient's account before submission. The Caffe weight
file should have SHA-256
`d369c8e593250e421eda192512263dccd35edb5c4b57e97312651bac10a51b7d`.

Access check on 2026-11-04: all four dataset pages and their Kaggle metadata
endpoints returned HTTP 200 without credentials or cookies. Each metadata
response matched the expected dataset slug and reported `isPrivate=false`.
This verifies public page/metadata visibility; file downloads and access from
the instructor's account were not tested. These input/code datasets do not
establish a download location for the trained B0/R1/R2 artifacts or raw outputs.

After filling a ZIP URL, download it and compare its SHA-256 with the value in
the table. A matching filename alone does not establish artifact identity.
