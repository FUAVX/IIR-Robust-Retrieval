"""Package only the verified, completed Oxford part of the bounded B0 run."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
raw = ROOT/'artifacts/revision-validation/outputs/b0-timeout-v1/b0_clean'
compact = ROOT/'artifacts/revision-validation/results/b0_clean'
target = ROOT/'artifacts/revision-validation/publish/b0-oxford-results'
target.mkdir(parents=True, exist_ok=True)
recovery = json.loads((compact/'recovery_manifest.json').read_text())
assert recovery['status'] == 'PARTIAL_BUDGET_LIMIT'
assert recovery['completed_datasets'] == ['oxford5k']
for path in sorted(raw.glob('*.npz')):
    assert 'oxford5k' in path.name
    shutil.copy2(path, target/path.name)
for path in sorted(compact.iterdir()):
    if path.suffix in {'.csv', '.json'}:
        shutil.copy2(path, target/path.name)
(target/'original_job_status.json').write_bytes((raw/'status.json').read_bytes())
(target/'README.md').write_text('''# B0 epoch-12 clean Oxford outputs

All five Oxford rankings and the complete gallery/query descriptors were saved
before the original Kaggle job hit its 30-minute extraction cap during Paris.
The complete Oxford rankings were checked as full permutations of the gallery,
and all 275 per-query AP values were recomputed independently after download.
The parent job remains ERROR / PARTIAL_BUDGET_LIMIT. No Paris result is included.

The validation-selected checkpoint SHA-256 and input/source hashes are recorded
in recovery_manifest.json. summary.csv and per_query.csv contain the accepted
measurements. inventory.json checksums every released payload. No benchmark
images, model weights, credentials or execution logs are included.
''')
files = sorted(p for p in target.iterdir() if p.is_file() and p.name not in ['inventory.json', 'dataset-metadata.json'])
inventory = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(target/'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
(target/'dataset-metadata.json').write_text(json.dumps(dict(
    id='ndhoang2310/iir-revision-b0-oxford-numerical-v1',
    title='IIR B0 Oxford Validation Checkpoint Results', licenses=[dict(name='other')],
    description='Complete Oxford-only numerical outputs recovered from a bounded B0 epoch12 job. Paris timed out and is excluded. Includes descriptors, rankings, per-query metrics and checksums.'), indent=2)+'\n')
print('Packaged', len(files), 'payload files,', sum(row['bytes'] for row in inventory.values()), 'bytes')
