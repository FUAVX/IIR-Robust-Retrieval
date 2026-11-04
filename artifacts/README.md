# External artifact access

**Điền URL chia sẻ vào cột `Access URL` của bảng đầu tiên trước khi gửi repo
cho người cần chạy lại từ checkpoint.** Ba giá trị `PENDING_URL_*` là vị trí
cần thay; URL phải truy cập được bằng tài khoản mà giảng viên sẽ dùng. Không
đưa API token, mật khẩu, checkpoint hoặc ZIP lớn vào Git.

| Artifact ngoài Git | SHA-256 của file lưu trữ | Access URL — điền tại đây |
| --- | --- | --- |
| `B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip` — B0 epoch 12, chứa `best.pt`, history và evaluation JSON | `d58bcaa735f2dde7eac132e38c8726882166ea00589f1fa45581469ee8fa2f73` | **PENDING_URL_B0_ZIP** |
| `run1_vgg16_gem_augmentation_v2-20261030T082558Z-1-002.zip` — R1 augmentation | `76c52b0d73eaa2e87df97f3b3fadfe9ad7726a93031c9eb75a81ad8abac760a8` | **PENDING_URL_R1_ZIP** |
| `run2_attention_bf16_100ep_v1-20261030T082559Z-1-001.zip` — R2 augmentation + gate | `1b89fb9dcdff9caa9b589e069e497602d55fe39f1d435b0ec911791bd74321fb` | **PENDING_URL_R2_ZIP** |

The [B0 evidence index](../reproduction/B0/epoch_012/README.md) describes the
first ZIP and its checkpoint hash. The [paper source manifest](../paper/ieee/data/source_manifest.json)
records the R1/R2 archive identities; the paper's aggregate CSV/JSON snapshots
are already committed under [`paper/ieee/data/`](../paper/ieee/data/). To repeat
raw post-training inference or inspect every saved ranking, the team must also
provide the corresponding raw outputs or their own verified download location.
Add that location here when available: **PENDING_URL_POSTTRAINING_RAW_OUTPUTS**.

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

After filling a ZIP URL, download it and compare its SHA-256 with the value in
the table. A matching filename alone does not establish artifact identity.
