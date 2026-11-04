# B0 reproduction result: SfM subset-100

This is the recorded historical run, not a new inference run. The analysis uses
saved Kaggle history and evaluation JSON. The selected checkpoint is **epoch 12**
because it had the lowest validation contrastive loss; the snapshot completed
**14 of 100 planned epochs**. See the [evidence index](../../reproduction/B0/epoch_012/README.md),
[analysis manifest](../../reproduction/B0/epoch_012/results/analysis_manifest.json),
and [per-query results](../../reproduction/B0/epoch_012/results/query_AP_epoch_012.csv).

| Field | Recorded value |
| --- | --- |
| Source snapshot | `747166e627e514842f5c5be531f4e0f73612cc90` |
| Model | VGG16 features, one shared trainable GeM `p`, L2-normalized 512-D descriptor |
| Training data | SfM subset-100 seed 42: 15,953 images, 100 clusters, 34,337 valid positive pairs |
| Validation data | 739 images, 20 clusters, 181 pairs; 180 complete tuples processed per epoch |
| Tuples and mining | 2,000 training tuples per epoch; one query, one positive, five negatives; requested pool 20,000, effective pool 15,953 |
| Optimizer and loss | Adam; backbone initial LR `1e-6`, GeM initial LR `1e-5`; contrastive loss margin `0.7` |
| Image and precision | Long-edge cap 1024; FP32; saved configuration reports Tesla T4 |
| Checkpoint selection | Lowest validation loss: epoch 12, loss `0.103847` |
| Evaluation | Legacy Oxford5k and Paris6k, 55 queries each; single scale, no whitening or query expansion |
| Oxford5k | 5,063 gallery images; **81.060% mAP** |
| Paris6k | 6,392 evaluated gallery images; **77.124% mAP** |

The earlier [dataset manifest](dataset_manifest.md) counts 6,412 supplied Paris
JPEGs. The [dataset owner's page](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
states that 20 supplied images are corrupt. The official
`gnd_paris6k.pkl` used by this project has SHA-256
`def916cf4f197a182d9a7840cdb6294489fa57c2121f9742a1bcc6f2209bb345`
and an `imlist` of 6,392 images. The archived B0 `run_record.json` records that
same ground-truth hash and gallery count. Thus, 6,412 is the supplied JPEG
count and 6,392 is the evaluated gallery count for this ground-truth file; the
run did not silently lose 20 gallery IDs. We have not individually matched the
excluded filenames to the owner's corrupt-image list.

The selected `best.pt` has SHA-256
`8a56bad2e1d585c68de42ff841f061e5469f76792750d8b4993e500294f19f27`.
The external ZIP containing the original run artifacts has SHA-256
`d58bcaa735f2dde7eac132e38c8726882166ea00589f1fa45581469ee8fa2f73`.
Checkpoints, benchmark images, and raw logs are not committed to Git.

This is a subset reproduction with a different training budget and settings from
the original paper; it is not an exact reproduction of the paper's full table.
The saved analysis did not rerun image inference, and this record does not claim
completion of 100 epochs or a controlled B1 ablation. The [saved presentation
summary](../../reproduction/B0/epoch_012/results/presentation_summary.md) gives
learning curves, query AP distribution, and the limits of interpretation.

For a new run, copy the [experiment record template](../../experiments/B0/experiment_template.md)
and fill every field from that run's own configuration and evidence.
