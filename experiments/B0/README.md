# B0 experiment records

Keep only lightweight experiment records in Git, such as Markdown summaries,
small configuration snapshots, and references to external artifacts.

Start each new run record from the [experiment template](experiment_template.md).
Record its experiment ID, exact Git commit and working-tree state, environment,
data/subset provenance, agreed settings, observations, and failures.
The saved subset-100 training and evaluation run is recorded in
[`report/B0/B0_results.md`](../../report/B0/B0_results.md), with compact evidence
under [`reproduction/B0/epoch_012/`](../../reproduction/B0/epoch_012/README.md).

Large datasets, checkpoints, model weights, descriptor dumps, and full generated
outputs belong outside Git, for example in ignored `data/` or `outputs/` storage
or an external artifact location. Include paths/checksums in records rather than
copying those artifacts here.

The imported upstream source is the historical starting point; current `main`
also contains the team's B0 preparation, reproduction record, B1 planning, and
paper draft. Document compatibility changes separately from methodological
changes, with commit references for each.
