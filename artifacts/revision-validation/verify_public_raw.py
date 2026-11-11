"""Verify public raw-output downloads without credentials or cookies."""
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REF = 'ndhoang2310/iir-post100-raw-numerical-outputs-v1'


def fetch_file(name):
    url = 'https://www.kaggle.com/api/v1/datasets/download/'+REF+'/'+urllib.parse.quote(name, safe='')
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    # Some Kaggle endpoints wrap a single file in a ZIP.
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            files = [n for n in z.namelist() if not n.endswith('/')]
            if len(files) == 1 and Path(files[0]).name == Path(name).name:
                payload = z.read(files[0])
    return payload


def main():
    path = ROOT/'artifacts/revision-validation/publish/raw-outputs/raw_inventory.json'
    local = path.read_bytes()
    remote = fetch_file('raw_inventory.json')
    assert remote == local, 'Public inventory differs from the locally verified inventory'
    inventory = json.loads(remote)
    samples = [next(row for row in inventory['files'] if '/results/' in row['path'] and row['path'].endswith('.npz')),
               next(row for row in inventory['files'] if '/desc_' in row['path'] and row['path'].endswith('.npz')),
               next(row for row in inventory['files'] if row['path'].endswith('/summary.csv'))]
    verified = []
    for row in samples:
        # Kaggle automatically expands an uploaded tar.gz into this directory.
        payload = fetch_file('post100-numerical-outputs/'+row['path'])
        assert len(payload) == row['bytes']
        assert hashlib.sha256(payload).hexdigest() == row['sha256']
        verified.append(dict(path=row['path'], bytes=len(payload), sha256=row['sha256']))
    report = dict(dataset=REF, url='https://www.kaggle.com/datasets/'+REF,
                  anonymous_inventory_verified=True, anonymous_sample_files=verified,
                  payload_files=len(inventory['files']), payload_bytes=inventory['payload_bytes'],
                  local_archive_sha256=inventory['archive_sha256'],
                  note='Local tar archive fully verified before upload; Kaggle expanded it. Public inventory and three representative files downloaded and hash-checked; not every remote file downloaded again.')
    output = ROOT/'artifacts/revision-validation/results/public_raw_verification.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
