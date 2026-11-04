# B0 experiment records

Keep only lightweight experiment records in Git, such as Markdown summaries,
small configuration snapshots, and references to external artifacts.

Start each run record from [B0_results.md](../../report/B0/B0_results.md).
Record its experiment ID, exact Git commit and working-tree state, environment,
data/subset provenance, agreed settings, observations, and failures.
There are no experiment results yet.

Large datasets, checkpoints, model weights, descriptor dumps, and full generated
outputs belong outside Git, for example in ignored `data/` or `outputs/` storage
or an external artifact location. Include paths/checksums in records rather than
copying those artifacts here.

`main` represents imported upstream source; `B0` contains the constrained
baseline reproduction. Document compatibility changes separately from
methodological changes, with commit references for each.
