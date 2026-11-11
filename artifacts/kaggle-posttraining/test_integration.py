"""CPU miniature of staged outputs, shard merging and strict coverage."""
import tempfile,json,contextlib
from pathlib import Path
import numpy as np
from PIL import Image
import runtime as R
import protocol as P

def test():
    old=P.DATASETS;P.DATASETS=['tiny']
    try:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);imgs=root/'jpg';imgs.mkdir();rng=np.random.default_rng(17)
            ids=['g'+str(i) for i in range(6)];qids=ids[:2]
            for gid in ids:Image.fromarray(rng.integers(0,256,(64,80,3),dtype=np.uint8)).save(imgs/(gid+'.jpg'))
            gt=dict(imlist=ids,qimlist=qids,gnd=[dict(ok=[2,3],junk=[0],bbx=[1,1,70,60]),dict(ok=[4],junk=[1],bbx=[1,1,70,60])]);bench={'tiny':dict(gt=gt,images=imgs)}
            manifest=dict(models=[dict(id=m,weights_sha256=m,checkpoint_sha256=m) for m in P.MODEL_IDS],protocol_hash='test')
            R.js(root/'models_manifest.json',manifest)
            for model in P.MODEL_IDS:
                np.savez_compressed(root/f'desc_{model}_tiny.npz',gallery=R.unit(rng.normal(size=(512,6))),queries=R.unit(rng.normal(size=(512,62))),gallery_ids=np.array(ids),query_ids=np.array(qids),conditions=np.array([c['name'] for c in P.conditions()]),weights_sha256=np.array(model),protocol_hash=np.array('test'))
            pixels=[]
            for c in P.conditions():
                for j,qid in enumerate(qids):
                    image=R.corrupt(R.prepare_image(imgs/(qid+'.jpg'),gt['gnd'][j]['bbx']),c,qid)
                    pixels.append(dict(query_id=qid,condition=c['name'],pixel_hash=R.pixel_hash(image)))
            R.js(root/f'pixels_{P.MODEL_IDS[0]}_tiny.json',pixels)
            for stage,fn in [('qf',R.qf_stage),('diffusion',R.diffusion_stage),('local',R.local_stage)]:
                out=root/stage;out.mkdir();fn(dict(input=str(root),job_id=stage),bench,manifest,out)
            for shard in [0,1]:
                out=root/f'geo{shard}';out.mkdir();R.geometry_stage(dict(input=str(root),job_id=f'geo{shard}',dataset='tiny',shard_index=shard,shard_count=2),bench,manifest,out)
            out=root/'report';out.mkdir();R.report_stage(dict(input=str(root)),bench,manifest,out)
            coverage=json.loads((out/'coverage.json').read_text());assert coverage['status']=='COMPLETE',coverage
            assert json.loads((out/'report_status.json').read_text())['status']=='INCOMPLETE' # direct hardware latency intentionally absent
            assert (out/'summary.csv').exists() and (out/'failure_examples.png').exists()
            print('PASS staged CPU fixture: all main cells, sensitivity, geometry shards, report; absent direct latency flagged')
    finally:P.DATASETS=old

if __name__=='__main__':test()
