# Notebook entrypoints

| Task | Notebook | Status |
| --- | --- | --- |
| Start a new B0 SfM subset-100 training run on Kaggle T4 x2 | [`artifacts/kaggle-training/public-handoff/02_train_fresh.ipynb`](../artifacts/kaggle-training/public-handoff/02_train_fresh.ipynb) | Recommended portable entrypoint; read the [handoff](../artifacts/kaggle-training/public-handoff/HANDOFF.md) and attach its three public inputs. The new engine has not been validated on a full GPU run. |
| Resume a matching historical checkpoint | [`02_train_sfm100_pipeline.ipynb`](../artifacts/kaggle-training/02_train_sfm100_pipeline.ipynb) | Requires the actual compatible checkpoint and identity checks; see the [pipeline notes](../artifacts/kaggle-training/README.md). |
| Review the earlier B0 setup and smoke design | [`kaggle_train_B0_sfm100.ipynb`](kaggle_train_B0_sfm100.ipynb) | Historical, opt-in notebook. It has no saved run output and its full-run choices are unresolved. |
| Recreate the saved B0 epoch-12 analysis | [`evaluation.ipynb`](../reproduction/B0/epoch_012/evaluation.ipynb) | Needs the external B0 artifact ZIP for sections 1–6; see the [evidence instructions](../reproduction/B0/epoch_012/README.md). |

The `artifacts/kaggle-training/public-handoff/` copies are deliberately
self-contained for transfer to another Kaggle account. Keep their manifests and
matching notebook/runtime files together when updating that bundle.
