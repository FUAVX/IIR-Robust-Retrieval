# B0 reproduction: saved evaluation at epoch 12

This folder preserves the team's historical analysis of a VGG16-GeM run trained on the SfM subset-100. The selected checkpoint was epoch 12 (lowest validation loss); the saved snapshot completed 14 of 100 planned epochs. The reported legacy single-scale mAP, without whitening or query expansion, is **81.060% on Oxford5k** and **77.124% on Paris6k**. These are results derived from saved Kaggle evaluation output; this branch does not rerun image inference.

- [Evaluation notebook](evaluation.ipynb) contains the analysis code and saved outputs.
- [Presentation summary](results/presentation_summary.md), the CSV tables, the analysis manifest, and PNG figures are under [`results/`](results/).
- [Evaluation presentation](../../../docs/B0/evaluation_report.pdf) is the final PDF only.

## Provenance and limits

The original inputs remain outside Git. The local ZIP `B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip` contains the run history, evaluation JSON, dataset provenance and `best.pt`; its SHA-256 is `d58bcaa735f2dde7eac132e38c8726882166ea00589f1fa45581469ee8fa2f73`. The `best.pt` inside it has SHA-256 `8a56bad2e1d585c68de42ff841f061e5469f76792750d8b4993e500294f19f27`, matching [`results/analysis_manifest.json`](results/analysis_manifest.json). The source snapshot used for that run was commit `747166e627e514842f5c5be531f4e0f73612cc90`.

The shared Drive download location for this ZIP is listed in the [external artifact
index](../../../artifacts/README.md).

The notebook's saved outputs and exported inventory retain their original
`reproduce/` paths as historical evidence. Its current input cell accepts the
extracted ZIP in `outputs/B0_VGG16_GeM_Evaluation/` or an explicit
`ARTIFACT_ROOT_OVERRIDE`. From the repository root, extract the external ZIP
with `unzip B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip -d outputs/`,
then run `evaluation.ipynb` from inside the repository. New exports go to
`outputs/b0-evaluation/epoch_012/` by default and do not overwrite the saved
results in Git. `OUTPUT_DIR_OVERRIDE` can select another destination.

Sections 1–6 need NumPy, pandas, Matplotlib, and IPython plus the extracted
artifact. Rerunning image inference also needs PyTorch, torchvision, Pillow,
and benchmark images. No model file, raw log, ZIP bundle, or LaTeX source is
committed here. The broader [paper draft](../../../paper/ieee/seg301_ieee_paper.pdf)
cites this B0 result but evaluates additional models separately.
