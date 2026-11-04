"""Confirm processing status, visibility without credentials, and prepared layout."""
import json,pathlib,requests,zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
ROOT=pathlib.Path(__file__).resolve().parent
DATASET='ndhoang2310/iir-b0-sfm100-prepared'
api=KaggleApi();api.authenticate()
try:
 status=api.dataset_status(DATASET)
except requests.HTTPError as error:
 if error.response.status_code in (403,404):
  print('Dataset access is not available yet (HTTP '+str(error.response.status_code)+'); publication is not verified.',flush=True)
  raise SystemExit(2)
 raise
print('Dataset processing status:',status,flush=True)
if status!='ready':raise SystemExit(2)
with api.build_kaggle_client() as client:
 request=ApiGetDatasetRequest();request.owner_slug=DATASET.split('/')[0];request.dataset_slug=DATASET.split('/')[1]
 row=client.datasets.dataset_api_client.get_dataset(request)
assert row.is_private is False,'Dataset is still private'
# Anonymous public API call uses no API key, cookies, or Authorization header.
r=requests.get('https://www.kaggle.com/api/v1/datasets/metadata/'+DATASET,timeout=(30,120))
r.raise_for_status()
public_info=r.json()
assert not public_info.get('hasErrorMessage'), 'Anonymous metadata access was denied'
info=public_info.get('info',{})
if isinstance(info,str):info=json.loads(info)
assert info.get('datasetSlug')==DATASET.split('/')[1], 'Anonymous metadata points to a different dataset'
# Complete file inventory verifies archive extraction rather than only uploaded zip.
names=[];token=None
while True:
 page=api.dataset_list_files(DATASET,page_token=token,page_size=1000)
 files=getattr(page,'files',None)
 if files is None:files=page.dataset_files
 names.extend(f.name for f in files)
 token=page.next_page_token
 if not token:break
 print('Listed files:',len(names),flush=True)
required={'prepared_manifest.json','retrieval-SfM-120k.pkl','needed_files.txt','subset_provenance_k100_seed42.txt'}
# Kaggle may mount extracted archive contents under an archive-named folder.
manifest_name=next((n for n in names if n=='prepared_manifest.json' or n.endswith('/prepared_manifest.json')),None)
assert manifest_name is not None,'Prepared manifest has not been extracted'
prefix=manifest_name[:-len('prepared_manifest.json')]
assert all(prefix+n in names for n in required)
images=[n for n in names if n.startswith(prefix+'ims/')]
assert len(images)==16692,f'Expected 16692 extracted image files, got {len(images)}'
with zipfile.ZipFile(ROOT/'publish-files/prepared_subset100.zip') as archive:
 manifest=json.loads(archive.read('prepared_manifest.json'))
assert set(images)=={prefix+'ims/'+relative for relative in manifest['sfm_images']}, 'Extracted CID paths differ from the verified manifest'
report=dict(dataset=DATASET,url='https://www.kaggle.com/datasets/'+DATASET,status=status,
            is_private=False,anonymous_visibility=True,images=len(images),files=len(names),prepared_prefix=prefix,
            current_version=getattr(row,'current_version_number',None))
(ROOT/'public-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
