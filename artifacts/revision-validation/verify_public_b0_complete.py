"""Check public access to the complete B0 inventory and representative payloads."""
import hashlib
import json
from pathlib import Path

from verify_public_raw import fetch_file

ROOT = Path(__file__).resolve().parents[2]
REF = 'ndhoang2310/iir-b0-complete-epoch12-results-v2'


def main():
    path = ROOT/'artifacts/revision-validation/publish/b0-complete-v2-results/inventory.json'
    local = path.read_bytes()
    remote = fetch_file('inventory.json', ref=REF)
    assert remote == local
    inventory = json.loads(remote)
    entries = {r['path']: r for r in inventory['files']}
    names = ['summary.csv', 'per_query.csv', 'ranks_paris6k_clean_diffusion_mutual_top20.npz',
             'desc_b0_epoch12_paris6k_robust.npz']
    verified = []
    for name in names:
        payload = fetch_file('b0-complete-v2/'+name, ref=REF)
        expected = entries[name]
        assert len(payload) == expected['bytes'], name
        assert hashlib.sha256(payload).hexdigest() == expected['sha256'], name
        verified.append(expected)
    report = dict(dataset=REF, url='https://www.kaggle.com/datasets/'+REF,
                  anonymous_inventory_matches=True, payload_files=inventory['payload_files'],
                  payload_bytes=inventory['payload_bytes'], anonymous_samples=verified,
                  archive_sha256=inventory['archive_sha256'],
                  verification_scope='Full local archive checked before upload; public inventory and four numerical payloads downloaded anonymously and hash-checked; not every remote payload downloaded again')
    output = ROOT/'artifacts/revision-validation/results/public_b0_complete_verification.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
