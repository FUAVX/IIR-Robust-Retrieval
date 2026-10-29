import json,os,pickle,time,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
started=time.monotonic()
print('Probe process started',flush=True)
from PIL import Image
result=dict(python=sys.version,pillow_import_seconds=time.monotonic()-started)
pending=[(Path('/kaggle/input'),0)];visited=0;found=[]
while pending and visited<200:
 path,depth=pending.pop(0);visited+=1
 candidate=path/'retrieval-SfM-120k.pkl'
 if candidate.is_file():found.append(candidate)
 if found:break
 if depth>=7:continue
 with os.scandir(path) as entries:
  for entry in entries:
   if entry.name.lower() in {'ims','jpg','images','image','jpeg','png','.git','__pycache__'}:continue
   if entry.is_dir(follow_symlinks=False):pending.append((Path(entry.path),depth+1))
result['discovery_seconds']=time.monotonic()-started
assert len(found)==1,found
metadata=found[0]
print('Found metadata:',metadata,flush=True)
with metadata.open('rb') as f:db=pickle.load(f)
result['metadata_seconds']=time.monotonic()-started
images=metadata.parent/'ims'
cids=db['train']['cids']
selected=list(cids[:16])+list(cids[-16:])
def decode(cid):
 path=images/cid[-2:]/cid[-4:-2]/cid[-6:-4]/cid
 begin=time.monotonic();size=path.stat().st_size;stat_time=time.monotonic()-begin
 with Image.open(path) as image:
  dims=image.size;image.convert('RGB').load()
 return dict(bytes=size,dimensions=dims,stat_seconds=stat_time,total_seconds=time.monotonic()-begin)
begin=time.monotonic()
serial=[decode(cid) for cid in selected[:16]]
result['serial_16_seconds']=time.monotonic()-begin
print('Serial 16 images:',result['serial_16_seconds'],flush=True)
begin=time.monotonic()
with ThreadPoolExecutor(max_workers=4) as executor:parallel=list(executor.map(decode,selected[16:]))
result['parallel_16_seconds']=time.monotonic()-begin
result['serial_samples']=serial;result['parallel_samples']=parallel
result['total_seconds']=time.monotonic()-started
Path('/kaggle/working/input_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
