# Short revision checks

The historical training and post-training source is now under
`artifacts/colab-training/` and `artifacts/kaggle-posttraining/`. The latter's
five runtime files match the hashes committed with the original paper. Keep
them unchanged when reproducing the archived tables and figures.

## Diffusion diagnosis (CPU only)

```sh
python -m unittest discover -s artifacts/revision-validation -p 'test_*.py'
OPENBLAS_NUM_THREADS=2 python artifacts/revision-validation/diffusion_checks.py
```

The second command needs the original descriptor/graph outputs under
`artifacts/kaggle-posttraining/outputs/` and hash-verified benchmark metadata
under `data/test/`. `results/diffusion_clean_audit.csv` contains the saved run;
its JSON companion records source hashes and diagnostic settings.

For R1/Oxford, the archived union graph with dense query seeds reproduces
23.673% clean mAP. Restricting the seeds to top-20, keeping that graph and beta
unchanged, gives 76.747%. Mutual top-20 gives 77.461%, compared with cosine's
74.000%. R1/Paris mutual top-20 gives 77.264%, compared with cosine's 64.288%.
These clean-cache diagnostics supersede the interpretation that diffusion
itself caused the historical collapse. They do not replace the historical
corruption measurements with unmeasured corrected results.

The historical formula for symmetric graph normalization is consistent with
the authors' `transition_matrix.m`. The important seed-support difference is
visible in their `ymake.m`:

- <https://github.com/ahmetius/diffusion-retrieval/blob/master/transition_matrix.m>
- <https://github.com/ahmetius/diffusion-retrieval/blob/master/ymake.m>
- <https://github.com/ahmetius/diffusion-retrieval/blob/master/knngraph.m>

Tests check a two-node closed-form solution, disconnected/isolated nodes,
independent dense affinity construction, gallery permutation/ties and agreement
with a direct linear solve. The real-cache audit compares fixed-point solutions
with SciPy CG for three fixed queries per case. Top-20 is a fixed diagnostic
support, not the winner of a test-set parameter search. The audit tightens the
solver tolerance for **both** dense and sparse seeds; its timing is not the
historical runtime timing.

`replay_sparse_diffusion.py` extends only the fixed mutual/top-20 method to all
31 cached views. It produces 124 view-level rows and 6,820 per-query AP records,
with realization-then-condition averaging. R1's equal-dataset corrupted mAP is
76.272%, versus 66.82% cosine and 71.05% verified QE in the archived study.
These revised quality results have no new direct end-to-end latency estimate.
Run `paper/ieee/build_revision_assets.py` to regenerate the new paper table.

## B0 validation-selected checkpoint (one bounded T4 job)

```sh
python artifacts/revision-validation/prepare_b0.py
python artifacts/revision-validation/build_b0_job.py
```

`prepare_b0.py` requires a compatible PyTorch/torchvision environment and the
original B0 ZIP in the repository root. It checks the source checkpoint hash,
epoch and selection rule, remaps only `pool.p` to `p`, strictly loads all weights
and compares the exported model against the upstream network on two input
shapes. The measured maximum descriptor error was 2.98e-8 on CPU.

The generated dataset is private. Submit the generated script only through the
existing `kaggle-account` bridge; request `NvidiaTeslaT4` and `-t 2400`. It uses
FP32, disables TF32, and stops extraction after 1,800 seconds. It does not train,
run sensitivity or local geometry, or change the historical epoch-100 notebooks.
Settings are fixed before the new run: QE k=10/lambda=.5; alpha=0/3; cosine gate
tau=.5; mutual diffusion graph k=20, query seed k=20, beta=.9.

Run `python artifacts/revision-validation/collect_b0.py --account hoang1` to
check once. Once the remote job is complete it downloads the outputs, verifies
the input manifest, requires all ten model/dataset/method groups and 550 unique
per-query records, and recomputes each mean AP before exporting compact results.
It never resubmits the job.

### Actual bounded-run outcome

Version 1 reached the 30-minute extraction limit partway through Paris. The
remote job is **ERROR**, not a successful two-dataset evaluation. Oxford's
5,063 gallery descriptors, 55 query descriptors and all five complete rankings
had already been saved. `collect_b0.py --recover-completed` can recover only
that known timeout case: it verifies checkpoint and image IDs, checks every
ranking as a full gallery permutation, and recomputes 275 per-query AP records.
The accepted record is `results/b0_clean/recovery_manifest.json`, with status
`PARTIAL_BUDGET_LIMIT`; no Paris result or runtime environment is invented.

Oxford mAP: cosine **81.060%**, AQE **83.774%**, alpha/gated QE **83.761%**,
mutual/top-20 diffusion **78.348%**. This confirms a practical QE gain on the
validation-selected B0, while the same sparse diffusion settings do not improve
this encoder. No additional GPU job is launched in this revision.

The [Oxford numerical outputs are public](https://www.kaggle.com/datasets/ndhoang2310/iir-revision-b0-oxford-numerical-v1).
All 12 payloads were downloaded without credentials and hash-checked. The release
includes descriptors, every full ranking, per-query metrics and the original
failed job status; `verify_public_b0.py` reproduces the download checks.

## Raw numerical outputs

`package_raw_outputs.py` creates and verifies a checksummed archive of every
unpacked CSV/JSON/NPZ/MD file. It omits duplicated transport files, rebuildable
SQLite caches, benchmark images, logs and checkpoint weights. The archive is
909,081,588 bytes; SHA-256
`f7e1bb3108977d86b986bbe5e125f107b0ff6df70286e7718d177e8f146f044d`.
The original outputs are never removed. See `artifacts/README.md` for the
verified download location when publication completes.
