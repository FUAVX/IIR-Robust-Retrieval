# B0 experiment results template

**TEMPLATE ONLY — no experiment has been recorded in this document.**

Copy this template into a lightweight record in `experiments/B0/` for each run.
Replace placeholders only with verified configuration or observed results.
`TBD` means unresolved/not recorded; it does not mean zero or a successful run.
For an actual failed run, explain unavailable metrics in the failures section.

| Field | Value |
| --- | --- |
| Experiment ID | TBD |
| Git commit | TBD |
| Branch and working-tree state | TBD |
| Environment (OS, Python, PyTorch, torchvision, CUDA/cuDNN, dependency versions) | TBD |
| GPU (model, count, memory) | TBD |
| Dataset/subset (prepared supervision identity, version, split, local/external location) | TBD |
| Number of training clusters | TBD |
| Queries per epoch | TBD |
| Tuple configuration (query, positive, negatives) | TBD |
| Mining configuration (selection, pool size, refresh frequency, cluster constraints) | TBD |
| Optimizer (including parameter groups) | TBD |
| Learning rate (including parameter groups) | TBD |
| Weight decay (including parameter groups) | TBD |
| Scheduler (policy, parameters, step timing) | TBD |
| Epochs (planned and completed) | TBD |
| Model-selection criterion | TBD |
| Image size, batch size, preprocessing, initialization, loss margin | TBD |
| Seed/randomness policy | TBD |
| Exact configuration/command reference | TBD |
| Runtime (elapsed training and evaluation time) | TBD |
| Checkpoint (local/external location, epoch, checksum) | TBD |
| Evaluation protocol (scale, whitening, query expansion, benchmark version) | TBD |
| Oxford5k mAP (state fraction or percent) | TBD |
| Paris6k mAP (state fraction or percent) | TBD |

## Deviations from paper

TBD — identify the verified PAPER SETTING and the actual TEAM B0 choice for each
deviation; explain why and how comparability is affected.

## Compatibility changes

TBD — record compatibility adaptations separately from methodological changes,
with commit references and validation evidence.

## Observations

TBD — record measured behavior and supporting lightweight evidence.

## Failures

TBD — record errors, interrupted runs, missing metrics, and corrective actions.

## Artifact references

TBD — link to local/external logs, datasets, checkpoints, weights, and descriptors.
Keep large artifacts outside Git; do not embed them in this record.
