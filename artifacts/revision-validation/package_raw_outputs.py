"""Package historical numerical outputs without duplicate transport/caches.

No upload is performed. SQLite feature/evidence caches and benchmark example
images are omitted; all unpacked CSV, JSON and NPZ measurements are included.
"""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def package():
    outputs = ROOT/'artifacts/kaggle-posttraining/outputs'
    folder = ROOT/'artifacts/revision-validation/publish/raw-outputs'
    folder.mkdir(parents=True, exist_ok=True)
    allowed = {'.csv', '.json', '.npz', '.md'}
    files = sorted(p for p in outputs.rglob('*') if p.is_file() and
                   'unpacked' in p.relative_to(outputs).parts and p.suffix in allowed)
    assert files
    status = outputs/'iir-post100-report/unpacked/report_status.json'
    assert json.loads(status.read_text())['status'] == 'COMPLETE'
    inventory = dict(scope='Historical epoch-100 numerical outputs, original files',
        included='all unpacked CSV/JSON/NPZ/MD; gallery/query descriptors, full rankings, metrics, graph matrices and diagnostics',
        excluded=['duplicate tar transport', 'SQLite local-feature/evidence caches (rebuildable)',
                  'benchmark example images', 'execution logs', 'checkpoint weights'],
        files=[dict(path=str(p.relative_to(outputs)), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    archive = folder/'post100-numerical-outputs.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(outputs)), recursive=False)
    # Verify every archived payload in a streaming pass, without extracting it.
    expected = {row['path']: row for row in inventory['files']}
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            stream = tar.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
            assert h.hexdigest() == expected[member.name]['sha256'], member.name
            seen.add(member.name)
    assert seen == set(expected)
    inventory.update(archive=archive.name, archive_bytes=archive.stat().st_size,
                     archive_sha256=sha(archive), archive_verified=True,
                     payload_bytes=sum(row['bytes'] for row in inventory['files']))
    (folder/'raw_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    source = ROOT/'artifacts/revision-validation/evidence/models_manifest.json'
    (folder/'models_manifest.json').write_bytes(source.read_bytes())
    (folder/'README.md').write_text('''# Historical post-training numerical outputs

Extract `post100-numerical-outputs.tar.gz` and retain its job/unpacked paths.
`raw_inventory.json` lists every file's byte count and SHA-256, plus the archive
hash. All numerical unpacked outputs are preserved, including descriptors,
full rankings, per-query AP and aggregate results. The source code is in the
repository's `artifacts/kaggle-posttraining/`; its hashes are recorded in
`models_manifest.json`.

This is the historical epoch-100 study, including the original dense-seed
diffusion results. New diffusion diagnostics and B0 evaluation are separate.
It excludes duplicated transport archives, rebuildable SQLite local-feature
and matching caches, example benchmark images, logs and model weights.
These exclusions do not remove the saved per-query rankings or measurements.
The numerical files are research outputs; underlying datasets retain their
original terms. This bundle contains no benchmark image collection.
''')
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(
        id='ndhoang2310/iir-post100-raw-numerical-outputs-v1',
        title='IIR Post100 Raw Numerical Outputs v1', licenses=[dict(name='other')],
        description='Historical retrieval descriptors, full rankings, per-query metrics, graph diagnostics and checksums. Excludes rebuildable SQLite caches, images and model weights.'), indent=2)+'\n')
    print(json.dumps({k: v for k, v in inventory.items() if k != 'files'}, indent=2))
    print('payload_files', len(files), flush=True)


if __name__ == '__main__':
    package()
