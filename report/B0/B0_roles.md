# B0 team roles

| Team member | Responsibility | Focus |
| --- | --- | --- |
| Phuong | Training & Implementation | Training setup, implementation traceability, and run reproducibility. |
| Hoang | Data | Prepared SfM supervision, dataset/subset provenance, and benchmark preparation. |
| Nguyen | Analysis & Implementation | Evaluation analysis, experiment records, and implementation review. |

**"Divide implementation responsibility, not knowledge."**

All three must understand the complete pipeline: prepared SfM supervision,
query/positive/negative tuples, hard-negative mining, shared-weight training,
VGG16-GeM descriptors, L2 normalization, and Oxford5k/Paris6k mAP evaluation.
Responsibility indicates who leads the work; it does not limit who understands
or reviews it.

Review decisions together against [B0_GeM.md](B0_GeM.md). Keep compatibility
changes and methodological changes documented separately, and share failures
and observations as well as successful results.
