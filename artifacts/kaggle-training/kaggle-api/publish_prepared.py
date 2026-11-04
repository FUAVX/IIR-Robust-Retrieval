"""Download verified CPU output and create the explicitly requested public dataset.
Uses the user's existing Kaggle credentials locally; never embeds credentials in a kernel.
"""
import argparse,hashlib,json,os,pathlib,time,zipfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
ROOT=pathlib.Path(__file__).resolve().parent
KERNEL='ndhoang2310/iir-b0-sfm100-cpu-preparation'
DATASET='ndhoang2310/iir-b0-sfm100-prepared'

def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def api_client():
 api=KaggleApi();api.authenticate();return api

def output_files(api):
 result_files={};token=None
 with api.build_kaggle_client() as client:
  while True:
   request=ApiListKernelSessionOutputRequest();request.user_name=KERNEL.split('/')[0];request.kernel_slug=KERNEL.split('/')[1]
   request.page_size=100;request.page_token=token
   result=client.kernels.kernels_api_client.list_kernel_session_output(request)
   result_files.update({entry.file_name:entry for entry in result.files})
   token=result.next_page_token
   if not token:break
 return result_files

def download():
 api=api_client()
 status=str(api.kernels_status(KERNEL).status)
 if not status.endswith('.COMPLETE'):raise RuntimeError('Kernel is not complete: '+status)
 files=output_files(api)
 required=['preparation_result.json','prepared_subset100.zip']
 for name in required:
  if name not in files:raise RuntimeError('Missing completed output: '+name)
 target=ROOT/'publish-files';target.mkdir(exist_ok=True)
 summary_path=ROOT/'preparation_result.json'
 response=requests.get(files[required[0]].url,timeout=(30,120));response.raise_for_status()
 summary=response.json()
 assert summary['status']=='COMPLETE' and summary['images']==16692
 summary_path.write_text(json.dumps(summary,indent=2))
 path=target/'prepared_subset100.zip'
 if path.exists() and path.stat().st_size==summary['archive_bytes'] and digest_file(path)==summary['archive_sha256']:
  print('Archive already downloaded and verified.',flush=True);return
 part=path.with_suffix('.zip.partial')
 offset=part.stat().st_size if part.exists() else 0
 headers={'Range':f'bytes={offset}-'} if offset else {}
 r=requests.get(files[required[1]].url,headers=headers,stream=True,timeout=(30,120));r.raise_for_status()
 mode='ab' if offset and r.status_code==206 else 'wb'
 if mode=='wb':offset=0
 free=__import__('shutil').disk_usage(target).free
 if summary['archive_bytes']-offset+1024**3>free:raise RuntimeError('Insufficient local disk for output download.')
 started=time.monotonic();last=started
 with part.open(mode) as f:
  for block in r.iter_content(chunk_size=4*1024*1024):
   if not block:continue
   f.write(block);offset+=len(block)
   now=time.monotonic()
   if now-last>=20:
    print(f'Downloaded {offset/1e9:.2f}/{summary["archive_bytes"]/1e9:.2f} GB',flush=True);last=now
 assert part.stat().st_size==summary['archive_bytes'],'Archive length mismatch'
 assert digest_file(part)==summary['archive_sha256'],'Archive SHA256 mismatch'
 part.replace(path)
 print('Archive downloaded and SHA256 verified:',path,flush=True)

def validate():
 path=ROOT/'publish-files/prepared_subset100.zip'
 summary=json.loads((ROOT/'preparation_result.json').read_text())
 assert path.stat().st_size==summary['archive_bytes']
 assert digest_file(path)==summary['archive_sha256']
 started=time.monotonic();last=started
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  assert len(names)==len(set(names)),'Duplicate archive entries'
  assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in names)
  manifest=json.loads(z.read('prepared_manifest.json'))
  assert manifest['format']=='b0_prepared_subset_v1' and manifest['full_image_decode_preflight'] is True
  assert manifest['dataset_identity']==summary['dataset_identity']
  images=manifest['sfm_images'];assert len(images)==16692
  expected={'ims/'+n for n in images}|set(manifest['metadata_files'])|{'prepared_manifest.json'}
  assert set(names)==expected,'Unexpected files in public payload'
  for n,h in manifest['metadata_files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  for i,(name,info) in enumerate(images.items(),1):
   h=hashlib.sha256();size=0
   with z.open('ims/'+name) as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block);size+=len(block)
   assert size==info['bytes'] and h.hexdigest()==info['sha256'],name
   now=time.monotonic()
   if now-last>=20:print(f'Verified image hashes: {i}/16692',flush=True);last=now
 report=dict(verified_images=len(images),archive_sha256=summary['archive_sha256'],dataset_identity=manifest['dataset_identity'])
 (ROOT/'validation-report.json').write_text(json.dumps(report,indent=2))
 print('Validated all 16,692 image hashes, metadata and public payload inventory.',flush=True)
 return report

def publish():
 report=validate()
 if (ROOT/'publish-result.json').exists():
  raise RuntimeError('Publication was already attempted; inspect publish-result.json before retrying.')
 metadata=dict(title='IIR B0 SfM100 Prepared Seed42',id=DATASET,licenses=[dict(name='other')],
  subtitle='Fixed SfM subset: 100 train clusters and 20 validation clusters',
  description='''Prepared SfM subset for the IIR B0 VGG16-GeM experiments.\n\nTrain: 15,953 images, 100 clusters, 34,337 positive pairs after removing two self-pairs. Validation: 739 images, 20 clusters, 181 positive pairs. Total: 16,692 original images. Seed 42. Images are unchanged original bytes; no resize or recompression.\n\nIncludes retrieval-SfM-120k.pkl, needed_files.txt, subset_provenance_k100_seed42.txt, prepared_manifest.json and the ims CID directory tree. All images passed full Pillow RGB decode during preparation; each image has a SHA-256 and byte count in the manifest. The archive is a transport container for the complete directory tree.\n\nSource: https://www.kaggle.com/datasets/ndhoang2310/iir-b0-sfm-full\nPreparation: https://www.kaggle.com/code/ndhoang2310/iir-b0-sfm100-cpu-preparation\nProject: https://github.com/FUAVX/IIR-Robust-Retrieval\n\nThis is a subset of the source SfM training data; it is not a newly collected image dataset. Image rights remain with the original rights holders. The Kaggle license category Other is retained from the source dataset. Oxford/Paris benchmark images, model weights and checkpoints are not included. No training result or mAP is claimed.''')
 folder=ROOT/'publish-files'
 (folder/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 allowed={'prepared_subset100.zip','dataset-metadata.json'}
 assert set(p.name for p in folder.iterdir())==allowed,'Unexpected file in upload folder'
 print(f'Uploading verified archive: {(folder / "prepared_subset100.zip").stat().st_size/1e9:.2f} GB',flush=True)
 result=api_client().dataset_create_new(str(folder),public=True,quiet=True,convert_to_csv=False,dir_mode='skip')
 out={k:getattr(result,k,None) for k in ['url','status','error','error_code','invalid_tags']}
 out['dataset']=DATASET;out['verified_images']=report['verified_images']
 (ROOT/'publish-result.json').write_text(json.dumps(out,default=str,indent=2))
 print(json.dumps(out,default=str),flush=True)
 if out.get('error') or not out.get('url'):
  raise RuntimeError('Dataset creation did not report success; inspect publish-result.json.')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download','validate','publish'])
 action=parser.parse_args().action
 globals()[action]()
