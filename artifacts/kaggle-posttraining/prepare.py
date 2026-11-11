"""Export only epoch-100 weights from the two user-owned archives."""
import io,json,hashlib,zipfile,shutil,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from runtime import digest,js,load_model,encode
from PIL import Image
import protocol as P

def prepare(project):
    output=ROOT/'bundle';output.mkdir(exist_ok=True)
    sources=[next(project.glob('run1*.zip')),next(project.glob('run2*.zip'))];models=[]
    image=Image.fromarray(np.random.default_rng(42).integers(0,256,(64,80,3),dtype=np.uint8))
    for model_id,archive in zip(P.MODEL_IDS,sources):
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if n.endswith('/latest.pt')]
            assert len(names)==1
            blob=z.read(names[0]);sha=hashlib.sha256(blob).hexdigest()
            if names[0]+'.sha256' in z.namelist():assert z.read(names[0]+'.sha256').decode().strip()==sha
            state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
        assert state['epoch']==100 and state['identity']['config']['attention']==(model_id==P.MODEL_IDS[1])
        path=output/(model_id+'.pth');torch.save(state['model'],path)
        info=dict(id=model_id,epoch=100,selection='final_epoch_100',checkpoint_sha256=sha,archive=archive.name,archive_sha256=digest(archive),weights_file=path.name,weights_sha256=digest(path),config=state['identity']['config'],training_identity=state['identity'])
        model=load_model(dict(info,path=str(path)),'cpu');a=encode(model,image,'cpu')
        model.load_state_dict(state['model'],strict=True);b=encode(model,image,'cpu');assert np.array_equal(a,b)
        js(output/(model_id+'_history.json'),state['history']);models.append(info)
        print('EXPORTED',model_id,path.stat().st_size,'bytes',flush=True)
        del state,blob,model
    code=['runtime.py','protocol.py','encoder_runtime.py','legacy_evaluate.py','artifacts_io.py']
    for name in code:shutil.copy2(ROOT/name,output/name)
    js(output/'models_manifest.json',dict(protocol=P.VERSION,protocol_hash=digest(ROOT/'protocol.py'),models=models,code_hashes={n:digest(ROOT/n) for n in code},weights_roundtrip='exact_float32_output',exploratory=True))
    return output

if __name__=='__main__':prepare(ROOT.parent.parent)
