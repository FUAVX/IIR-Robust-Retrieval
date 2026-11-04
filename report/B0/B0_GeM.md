# B0: VGG16-GeM baseline specification

Status at the time this specification was written: agreed method and documentation
skeleton only. Later B0 evidence is indexed in [B0_results.md](B0_results.md).

## Traceability

- At skeleton initialization, `main` represented the imported upstream source;
  current `main` also contains the team's subsequent research materials.
- The historical `B0` branch contained the constrained baseline reproduction
  structure before the later branches were merged.
- At skeleton initialization, both branches point to imported baseline commit
  `886c117070f72cd3d0a91fad30097143233f86e8`.
- The separate untouched reference repository is
  `~/projects/cnnimageretrieval-pytorch`.
- Preserve the original `cirtorch` implementation. Future compatibility changes
  (for example, dependency/API adaptation) must be documented separately from
  methodological changes (for example, sampling, loss, or evaluation changes).
  Each record should identify its category, reason, affected files, commit,
  validation, and effect on comparability. Do not silently treat a methodological
  change as a compatibility fix.

## TEAM B0: currently agreed pipeline

`VGG16 -> GeM with one shared trainable p -> L2 normalization -> 512-D descriptor`

GeM uses one scalar `p` shared across channels, not a separate parameter per
channel. The Siamese branches share the same network weights.

Training:

- Siamese/shared weights.
- Contrastive loss.
- SfM-derived prepared supervision.
- Each tuple contains one query, one positive, and five negatives
  (`q + p + n1 + n2 + n3 + n4 + n5`; seven images).
- Hard-negative mining.

Initial evaluation:

- Oxford5k (`oxford5k`) and Paris6k (`paris6k`).
- Metric: mean average precision (mAP).
- Single-scale evaluation (`[1]`).
- Whitening OFF, including local/network whitening and descriptor post-processing.
- Query expansion OFF.

These are the original Oxford5k/Paris6k benchmarks. Revisited Oxford/Paris are
different protocols and are not the agreed initial B0 evaluation.

## PAPER SETTING, SOURCE DEFAULT, and TEAM B0

- **PAPER SETTING** means a setting verified for the specific paper experiment
  being reproduced. The exact paper experiment and its full settings are **TBD**.
  Do not infer paper settings from parser defaults or copy published scores into
  B0 results.
- **SOURCE DEFAULT** means the behavior of this imported source when an option is
  omitted; it is evidence about the code, not a team agreement.
- **TEAM B0** means a decision explicitly agreed for this reproduction. Only the
  pipeline above is currently agreed; unresolved values below remain **TBD**.

The upstream [README](../../README.md) contains paper-oriented example commands,
including a ResNet101 training example and evaluation with multiscale/whitening.
Those examples are distinct from both CLI defaults and our VGG16 B0 decisions.
The applicable VGG16 paper settings still need verification before claiming a
numerically matched paper reproduction.

The following source observations come from
[train.py](../../cirtorch/examples/train.py),
[test.py](../../cirtorch/examples/test.py),
[traindataset.py](../../cirtorch/datasets/traindataset.py),
[pooling.py](../../cirtorch/layers/pooling.py), and
[imageretrievalnet.py](../../cirtorch/networks/imageretrievalnet.py).

| Setting | SOURCE DEFAULT / observed source behavior | TEAM B0 |
| --- | --- | --- |
| Architecture | Training CLI: `resnet101` | VGG16 |
| Pooling | Training CLI: `gem`; `GeM` has one trainable scalar `p`, initialized to 3 | GeM with one shared trainable `p`; initialization TBD |
| Descriptor | VGG16 output dimension is 512; retrieval network applies L2 normalization | L2-normalized 512-D descriptor |
| Loss | Training CLI: contrastive, margin 0.7 | Contrastive; margin TBD |
| Training data | Training CLI: `retrieval-SfM-120k` | SfM-derived prepared supervision; dataset/subset TBD |
| Tuple | Training CLI: five negatives | Query + positive + five negatives |
| Hard-negative pool | Training CLI: 20,000 candidate images | Hard-negative mining; pool size TBD |
| Number of training clusters | Determined by prepared data, not a fixed CLI cluster count | TBD |
| Queries per epoch | Training CLI: 2,000; dataset constructor default: 2,000 | TBD |
| Epochs | Training CLI: 100 | TBD |
| Mining refresh frequency | Training calls `create_epoch_tuples()` at the start of each epoch | TBD |
| Optimizer | Training CLI: Adam | TBD |
| Learning rate | Training CLI: `1e-6`; global GeM `p` parameter group uses 10 times that rate | TBD, including parameter-group rates |
| Scheduler policy | `ExponentialLR`, gamma `exp(-0.01)`, stepped before training each epoch | TBD |
| Weight decay | Training CLI: `1e-6`; global GeM `p` parameter group has zero decay | TBD, including parameter-group policy |
| Model-selection criterion | Lowest validation loss by default; lowest training loss if validation is disabled | TBD |
| Evaluation datasets | Training CLI: `roxford5k,rparis6k`; standalone test CLI: `oxford5k,paris6k` | Oxford5k and Paris6k |
| Evaluation scale | Standalone test CLI: `[1]` | Initial evaluation: single-scale |
| Whitening | Training whitening flags are off; test whitening is unset by default | OFF |
| Query expansion | Standalone test ranks extracted descriptors without query expansion | OFF |

The dataset constructor's historical docstring lists 1,000 queries and a 10,000
image mining pool, while its actual signature uses 2,000 and 20,000. Source
observations above refer to executable defaults, not those stale docstring values.

Additional run settings, including image size, batch size, preprocessing,
initialization/weights, validation split, seed policy, and environment, are **TBD**.
Record their final values before running an experiment. No final training
configuration is implied by this document.

## Records and artifacts

Use the [experiment template](../../experiments/B0/experiment_template.md) for
new lightweight records in `experiments/B0/`. Record the exact commit, configuration,
data identity, compatibility changes, and methodological deviations for each run.
Large datasets, checkpoints, weights, descriptors, and generated outputs belong
outside Git; record their external/local locations in the run record.
