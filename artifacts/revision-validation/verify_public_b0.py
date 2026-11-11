"""Download and checksum every small public B0 Oxford payload anonymously."""
import hashlib
import json
from pathlib import Path
from verify_public_raw import fetch_file

ROOT = Path(__file__).resolve().parents[2]
REF = 'ndhoang2310/iir-revision-b0-oxford-numerical-v1'
local = (ROOT/'artifacts/revision-validation/publish/b0-oxford-results/inventory.json').read_bytes()
remote = fetch_file('inventory.json', ref=REF)
assert remote == local
inventory = json.loads(remote)
verified = []
for name, expected in inventory.items():
    payload = fetch_file(name, ref=REF)
    assert len(payload) == expected['bytes'], name
    assert hashlib.sha256(payload).hexdigest() == expected['sha256'], name
    verified.append(name)
report = dict(dataset=REF, url='https://www.kaggle.com/datasets/'+REF,
              anonymous=True, all_payload_hashes_match=True, files=verified,
              scope='Oxford only; the original two-dataset job hit its budget during Paris')
(ROOT/'artifacts/revision-validation/results/public_b0_verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
