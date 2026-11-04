# IIR-Robust-Retrieval

**Research on robust and discriminative representations for instance-level image retrieval.**

## Overview

Instance-level Image Retrieval (IIR) aims to retrieve images depicting the same specific object or landmark as a query image. This task becomes challenging when images differ in viewpoint, scale, illumination, occlusion, background clutter, or visual quality.

**IIR-Robust-Retrieval** investigates how to improve the robustness of learned image descriptors across these visual variations while preserving their ability to distinguish different instances.

## Research Objectives

- Establish a reproducible **GeM-based retrieval baseline**.
- Evaluate retrieval accuracy and descriptor robustness under diverse visual variations.
- Analyze retrieval failures and the trade-off between robustness and instance discriminability.
- Explore and validate improvements through controlled experiments and ablation studies.

## Approach

The starting point is a CNN-based image retrieval pipeline built on **VGG16**, **Generalized Mean (GeM) pooling**, and **L2-normalized global descriptors**. The baseline uses SfM-derived training supervision and contrastive learning with hard-negative mining.

The research follows a baseline-first methodology:

1. **B0 — Baseline:** Reproduce and evaluate the GeM retrieval pipeline.
2. **Robustness Analysis:** Measure performance under visual variations and identify systematic failure modes.
3. **B1 — Improvement:** Design and evaluate an evidence-driven robustness enhancement against B0.

The original B1 planning documents describe proposed experiments. The later paper draft reports exploratory augmentation, attention-gate, and post-training retrieval experiments; read their evidence and limitations before treating a planned ablation as completed.

## Current evidence

- The saved B0 subset-100 run completed 14 of 100 planned epochs. Its validation-selected checkpoint at epoch 12 reached **81.06% Oxford5k mAP** and **77.12% Paris6k mAP** under the legacy single-scale protocol, without whitening or query expansion. The [B0 reproduction record](reproduction/B0/epoch_012/README.md) links the notebook, per-query tables, figures, and checkpoint provenance.
- The [IEEE paper draft](paper/ieee/README.md) reports additional training and search experiments. It is a draft for team review, with provisional author names and explicit limits on cross-run comparisons. The team intends to update it.

## Datasets and Evaluation

- **retrieval-SfM-120k:** Training supervision based on Structure-from-Motion (SfM) reconstructions.
- **Oxford5k / Paris6k:** Instance-level image retrieval benchmarks.

Retrieval performance is evaluated using **mean Average Precision (mAP)** under the appropriate benchmark protocols. Robustness experiments additionally examine how retrieval performance changes across visual conditions and transformation severities.

## Repository

The repository builds upon the original [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) implementation. Research code, configurations, and experiments are organized around the GeM baseline and subsequent robustness investigations.

For implementation details and reproducibility notes:

- [`configs/B0/`](configs/B0/) and [`report/B0/`](report/B0/): baseline specification, data audit, and methodology.
- [`docs/B0/`](docs/B0/) and [`docs/project/`](docs/project/): B0 plan, evaluation presentation, and project brief.
- [`docs/B1/`](docs/B1/): B1 robustness and retrieval handoff. This is a dated experiment specification, not a record of completed results.
- [`reproduction/B0/epoch_012/`](reproduction/B0/epoch_012/): saved B0 evaluation and compact evidence; model files and raw logs remain outside Git.
- [`notebooks/`](notebooks/): notebook entrypoint guide, including the historical B0 smoke notebook.
- [`artifacts/kaggle-training/`](artifacts/kaggle-training/) and [`paper/ieee/`](paper/ieee/): portable training handoff and later paper draft, respectively.

## References and Attribution

This project uses and adapts the open-source implementation associated with:

- Radenović, F., Tolias, G., & Chum, O. **Fine-tuning CNN Image Retrieval with No Human Annotation.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2018. [Paper](https://arxiv.org/abs/1711.02512)

**Original source:** https://github.com/filipradenovic/cnnimageretrieval-pytorch

The original authors retain credit for their methods and implementation. See [`LICENSE`](LICENSE) for the applicable repository license.
