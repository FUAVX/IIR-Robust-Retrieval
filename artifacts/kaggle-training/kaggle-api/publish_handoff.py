"""Publish the reviewed, small handoff bundle and verify public file bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT.parent / 'public-handoff'
DATASET = 'ndhoang2310/iir-b0-training-handoff'
api = KaggleApi()
api.authenticate()
subprocess.run([sys.executable, str(FOLDER / 'verify_bundle.py')], check=True)
manifest = json.loads((FOLDER / 'bundle-manifest.json').read_text())
allowed = set(manifest) | {'bundle-manifest.json', 'dataset-metadata.json'}
assert {p.name for p in FOLDER.iterdir()} <= allowed
assert sum(p.stat().st_size for p in FOLDER.iterdir()) < 10 * 1024 * 1024
deps = json.loads((FOLDER / 'portable-dependencies.json').read_text())
for ref in ['ndhoang2310/iir-b0-sfm100-prepared', 'ndhoang2310/iir-b0-benchmarks', 'ndhoang2310/iir-b0-vgg16-weights']:
    assert deps[ref]['public'] and deps[ref]['anonymous'] and deps[ref]['version'] == 1
metadata = dict(id=DATASET, title='IIR B0 Training Handoff', licenses=[dict(name='other')],
                subtitle='Portable notebooks, public input links and a Vietnamese Codex handoff report',
                description='''Start with HANDOFF.md, then use 02_train_fresh.ipynb on Kaggle T4 x2. This small bundle includes self-contained notebooks, runtime source, CPU checks, integrity manifest and verification reports. It does not include the bulk image data; attach the three public version-1 datasets linked in HANDOFF.md: ndhoang2310/iir-b0-sfm100-prepared, ndhoang2310/iir-b0-benchmarks, ndhoang2310/iir-b0-vgg16-weights.

The fresh notebook needs no historical checkpoint. The resume notebook is included separately, but its old third-party checkpoint is not available in this bundle. No GPU training or mAP result is claimed. Fresh notebook schema/compile and CPU numerical tests passed; all 16,692 prepared image hashes and the actual training metadata contract were checked. No credentials, raw execution logs or trained checkpoints are published here.

Code copyright/license notices are preserved in LICENSE (MIT). Input data/weights retain their source license category Other. See HANDOFF.md for exact scope, limitations, setup and commands.''')
(FOLDER / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
result_path = ROOT / 'handoff-publish-result.json'
if not result_path.exists():
    result = api.dataset_create_new(str(FOLDER), public=True, quiet=True, convert_to_csv=False, dir_mode='skip')
    out = {k: getattr(result, k, None) for k in ['url', 'status', 'error', 'error_code']}
    result_path.write_text(json.dumps(out, default=str, indent=2))
    print(json.dumps(out, default=str), flush=True)
    assert not out.get('error') and out.get('url'), out
else:
    previous = json.loads(result_path.read_text())
    assert previous.get('url') and not previous.get('error'), previous

for attempt in range(60):
    try:
        status = api.dataset_status(DATASET)
        if status != 'ready':
            print('Handoff processing:', status, flush=True)
            time.sleep(20)
            continue
        with api.build_kaggle_client() as client:
            q = ApiGetDatasetRequest()
            q.owner_slug, q.dataset_slug = DATASET.split('/')
            row = client.datasets.dataset_api_client.get_dataset(q)
        assert row.is_private is False
        r = requests.get('https://www.kaggle.com/api/v1/datasets/metadata/' + DATASET, timeout=60)
        r.raise_for_status()
        response = r.json()
        info = response['info']
        info = json.loads(info) if isinstance(info, str) else info
        assert not response.get('hasErrorMessage') and info['datasetSlug'] == DATASET.split('/')[1]
        # One small file-list request; do not rescan the 16k-image input datasets.
        page = api.dataset_list_files(DATASET, page_size=100)
        files = getattr(page, 'files', None)
        if files is None:
            files = page.dataset_files
        names = {f.name for f in files}
        assert not page.next_page_token
        assert names == set(manifest) | {'bundle-manifest.json'}, names
        # Verify every published payload byte through the authenticated download API.
        download_dir = ROOT / 'handoff-public-check'
        download_dir.mkdir(exist_ok=True)
        for name in sorted(names):
            api.dataset_download_file(DATASET, name, path=str(download_dir), force=True, quiet=True)
            local = download_dir / name
            assert local.is_file(), name
            assert hashlib.sha256(local.read_bytes()).hexdigest() == hashlib.sha256((FOLDER / name).read_bytes()).hexdigest(), name
        report = dict(dataset=DATASET, url='https://www.kaggle.com/datasets/' + DATASET,
                      ready=True, public=True, anonymous_metadata=True,
                      version=row.current_version_number, files=sorted(names),
                      all_published_file_hashes_match=True)
        (ROOT / 'handoff-public-verification.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        break
    except requests.HTTPError as error:
        if error.response.status_code not in (403, 404, 429):
            raise
        print('Waiting for public handoff access, HTTP', error.response.status_code, flush=True)
        time.sleep(20)
else:
    raise TimeoutError('Handoff public verification is incomplete.')
