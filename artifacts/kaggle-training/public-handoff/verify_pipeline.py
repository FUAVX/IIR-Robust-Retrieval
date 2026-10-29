"""Numerical CPU checks; requires torch/numpy/Pillow, no GPU or Kaggle data."""
import ast, base64, copy, hashlib, importlib.util, json, pathlib, sys, tempfile, time, zlib
from types import SimpleNamespace, ModuleType
sys.dont_write_bytecode = True
import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
base = json.loads((ROOT/'notebookc025fac755_optimized.ipynb').read_text())
source_cell = next(''.join(c['source']) for c in base['cells'] if 'SOURCE_PAYLOAD = ' in ''.join(c['source']))
payload = next(ast.literal_eval(n.value) for n in ast.parse(source_cell).body if isinstance(n, ast.Assign) and any(getattr(t,'id',None)=='SOURCE_PAYLOAD' for t in n.targets))
snapshot = json.loads(zlib.decompress(base64.b64decode(payload)))
TMP = tempfile.TemporaryDirectory()
for name, value in snapshot.items():
    p = pathlib.Path(TMP.name)/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value)
sys.path.insert(0,TMP.name)
# Load only pinned layer/dataset modules, without importing the torchvision CLI.
for name in ['cirtorch','cirtorch.layers','cirtorch.datasets']:
    package=ModuleType(name);package.__path__=[str(pathlib.Path(TMP.name)/name.replace('.', '/'))]
    sys.modules[name]=package
from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.layers.pooling import GeM
from cirtorch.datasets.genericdataset import ImagesFromList as _ImagesFromList
from cirtorch.datasets.datahelpers import pil_loader
# Notebook explicitly selects Pillow; inject that same backend without torchvision.
class ImagesFromList(_ImagesFromList):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('loader',pil_loader)
        super().__init__(*args,**kwargs)

spec = importlib.util.spec_from_file_location('runtime',ROOT/'pipeline_runtime.py')
r = importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.torch, r.time = torch, time
r.effective_workers = lambda n:0
r.traindataset = SimpleNamespace(ImagesFromList=ImagesFromList)
r.original_train = SimpleNamespace(set_batchnorm_eval=lambda m:None, args=SimpleNamespace(update_every=1, print_freq=10))

class TinyRetrieval(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1),torch.nn.ReLU(),torch.nn.MaxPool2d(2),torch.nn.Conv2d(8,8,3,padding=1),torch.nn.ReLU())
        self.pool = GeM()
        self.meta = {'outputdim':8}
        self.forward_calls=0
        self.forward_images=0
    def forward(self,x):
        self.forward_calls+=1;self.forward_images+=x.shape[0]
        x=self.pool(self.features(x)).flatten(1)
        return torch.nn.functional.normalize(x,dim=1).t()

torch.set_num_threads(1)
torch.manual_seed(42)
shapes=[(16,24),(24,16),(16,24),(16,24),(24,16),(20,20),(20,20)]
images=[torch.randn(1,3,h,w) for h,w in shapes]
a=TinyRetrieval();b=copy.deepcopy(a)
criterion=ContrastiveLoss()
target=torch.tensor([-1,1,0,0,0,0,0.])
x=torch.cat([a(im) for im in images],dim=1)
y=r.pipeline_tuple_vectors(b,images)
torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
torch.testing.assert_close(criterion(x,target),r.PipelineContrastiveLoss()(y,target),rtol=1e-5,atol=1e-6)
criterion(x,target).backward();r.PipelineContrastiveLoss()(y,target).backward()
for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)
assert b.forward_calls==4 and a.forward_calls==7
print('PASS: mixed-shape microbatch descriptors, vectorized contrastive loss and all parameter gradients match scalar reference (7 -> 4 forward calls).')

for tuples in (1, 5):
    x=torch.randn(512,7*tuples,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    labels=target.repeat(tuples)
    loss_a=criterion(x,labels)
    loss_b=r.PipelineContrastiveLoss()(y,labels)
    torch.testing.assert_close(loss_a,loss_b,rtol=1e-6,atol=1e-6)
    loss_a.backward();loss_b.backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-6)
print('PASS: vectorized loss and descriptor gradients match original for one/five tuples with 512-D descriptors.')

# Verify two tuples/update over multiple updates, including Adam state.
class Loader:
    batch_size=2;drop_last=True
    def __init__(self,ds,values):self.dataset=ds;self.values=values
    def __iter__(self):return iter(self.values)
    def __len__(self):return len(self.values)
a=TinyRetrieval();b=copy.deepcopy(a)
oa=torch.optim.Adam(a.parameters(),lr=1e-4);ob=torch.optim.Adam(b.parameters(),lr=1e-4)
batches=[([images,[im*.7 for im in images]],[target,target]) for _ in range(3)]
reference=[]
for inputs,labels in batches:
    for ims,label in zip(inputs,labels):
        loss=criterion(torch.cat([a(im) for im in ims],dim=1),label)
        reference.append(loss.item());loss.backward()
    oa.step();oa.zero_grad()
loader=Loader(SimpleNamespace(create_epoch_tuples=lambda model:None),batches)
value=r.pipeline_train(loader,b,r.PipelineContrastiveLoss(),ob,0)
assert abs(value-np.mean(reference))<1e-6
for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=2e-4,atol=2e-6)
print('PASS: optimizer update count, summed tuple gradients, Adam updates and reported train loss match reference.')

# Real images exercise Pillow thumbnail preprocessing, variable shapes and reordering.
folder=pathlib.Path(TMP.name)/'images';folder.mkdir()
paths=[]
rng=np.random.default_rng(2)
for i in range(12):
    path=folder/f'{i}.png'
    w,h=[(24,16),(16,24),(20,20)][i%3]
    Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8)).save(path)
    paths.append(str(path))
def transform(image):return torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/255
model=TinyRetrieval().eval()
dataset=ImagesFromList('',paths,imsize=20,transform=transform)
with torch.no_grad():
    ref=torch.cat([model(dataset[i].unsqueeze(0)) for i in range(len(dataset))],dim=1)
    out=r.pipeline_extract_gpu(model,paths,20,transform)
torch.testing.assert_close(out,ref,rtol=1e-5,atol=1e-6)
boxes=[(1,1,10,12)]*len(paths)
dataset=ImagesFromList('',paths,imsize=20,bbxs=boxes,transform=transform)
with torch.no_grad():
    ref=torch.cat([model(dataset[i].unsqueeze(0)) for i in range(len(dataset))],dim=1)
    out=r.pipeline_extract_vectors(model,paths,20,transform,bbxs=boxes)
torch.testing.assert_close(out,ref,rtol=1e-5,atol=1e-6)
print('PASS: windowed extraction keeps gallery order, variable shapes and original query crop/resize protocol.')

class MiningDataset:
    def __init__(self,mode='train',poolsize=12):
        self.mode=mode;self.images=paths;self.imsize=20;self.transform=transform
        self.clusters=[i//2 for i in range(12)]
        self.qpool=[0,2,4,6,8,10];self.ppool=[1,3,5,7,9,11]
        self.qsize=5;self.poolsize=poolsize;self.nnum=5
    def create_epoch_tuples(self,net):return r.pipeline_mine(self,net)
    def __len__(self):return self.qsize

def reference_mining(ds,model):
    pairs=torch.randperm(len(ds.qpool))[:ds.qsize].tolist()
    queries=[ds.qpool[i] for i in pairs]
    positives=[ds.ppool[i] for i in pairs]
    pool=torch.randperm(len(ds.images))[:ds.poolsize].tolist()
    dataset=ImagesFromList('',paths,imsize=20,transform=transform)
    with torch.no_grad():
        q=torch.cat([model(dataset[i].unsqueeze(0)) for i in queries],dim=1)
        p=torch.cat([model(dataset[i].unsqueeze(0)) for i in pool],dim=1)
        _,ranks=torch.sort(p.t()@q,dim=0,descending=True)
    selected=[]
    for column,query in enumerate(queries):
        seen=[ds.clusters[query]];negatives=[];rank=0
        while len(negatives)<ds.nnum:
            candidate=pool[int(ranks[rank,column])]
            if ds.clusters[candidate] not in seen:
                negatives.append(candidate);seen.append(ds.clusters[candidate])
            rank+=1
        selected.append(negatives)
    for _ in range(2):torch.empty((),dtype=torch.int64).random_()
    return queries,positives,pool,selected,torch.get_rng_state().clone()
for poolsize in (12,11):
    ds=MiningDataset(poolsize=poolsize)
    torch.manual_seed(73)
    qs,ps,pool,ns,state=reference_mining(ds,model)
    torch.manual_seed(73)
    before=model.forward_images
    distance=r.pipeline_mine(ds,model)
    assert ds.qidxs==qs and ds.pidxs==ps and ds.nidxs==ns
    assert torch.equal(state,torch.get_rng_state())
    assert model.forward_images-before==len(set(pool+qs))
    assert np.isfinite(distance)
print('PASS: full/partial-pool mining preserves sampled query/positive/negative indices and CPU RNG; forwards only unique images.')

# Validation uses exact drop_last and criterion summation without a second forward.
ds=MiningDataset(mode='val')
loader=Loader(ds,[])
torch.manual_seed(81)
value=r.pipeline_validate(loader,model,r.PipelineContrastiveLoss(),0)
assert not hasattr(ds,'_pipeline_validation_vectors')
dataset=ImagesFromList('',paths,imsize=20,transform=transform)
ref=0.;count=0
with torch.no_grad():
    for start in range(0,4,2):
        ids=[]
        for q in range(start,start+2):ids.extend([ds.qidxs[q],ds.pidxs[q]]+ds.nidxs[q])
        vectors=torch.cat([model(dataset[i].unsqueeze(0)) for i in ids],dim=1)
        ref+=criterion(vectors,target.repeat(2)).item();count+=2
assert abs(value-ref/count)<1e-6
print('PASS: validation descriptor reuse preserves loss, batch layout, drop_last and releases its cache.')

for name in ['01_prepare_sfm100_cpu.ipynb','02_train_sfm100_pipeline.ipynb']:
    notebook=json.loads((ROOT/name).read_text())
    for i,c in enumerate(notebook['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'{name}:{i}','exec')
    assert not any('make_subset' in ''.join(c['source']) for c in notebook['cells']) if name.startswith('02') else True
print('PASS: both notebooks compile; training contains no subset generator.')

# Preparation: execute the actual subset/cleanup cells on synthetic full metadata
# matching the historical counts, then verify exported bytes and collision refusal.
import pickle, random, shutil
train_sizes=[160]*53+[159]*47
val_sizes=[37]*19+[36]
def partition(sizes,cluster_offset,cid_offset,pairs,self_pairs=0):
    cids=[];clusters=[];valid=[];index=0
    for c,size in enumerate(sizes):
        for j in range(size):
            cids.append(f'{index+cid_offset:016x}');clusters.append(c+cluster_offset)
            valid.append((index, index-j+(j+1)%size));index+=1
    qp=(valid*((pairs//len(valid))+1))[:pairs-self_pairs]+[(i,i) for i in range(self_pairs)]
    return dict(cids=cids,cluster=clusters,qidxs=[q for q,p in qp],pidxs=[p for q,p in qp])
full=dict(train=partition(train_sizes,0,0,34339,2),val=partition(val_sizes,100,100000,181))
full_file=pathlib.Path(TMP.name)/'full.pkl';full_file.write_bytes(pickle.dumps(full))
ns=dict(pickle=pickle,random=random,Path=pathlib.Path,SOURCE_SFM_PKL=full_file,SFM_DB=full)
code_cells=[''.join(c['source']) for c in base['cells'] if c['cell_type']=='code']
for marker in ['SEED = 42','for part in ["train", "val"]:','# Remove self-pairs','# Derive provenance']:
    source=next(s for s in code_cells if marker in s)
    source=source.replace('/kaggle/working/b0_sfm100_notebook/generated_subset100',str(pathlib.Path(TMP.name)/'generated'))
    exec(source,ns)
assert len(ns['SFM_DB']['train']['qidxs'])==34337
assert len(ns['SFM_DB']['val']['qidxs'])==181
assert all(q!=p for part in ns['SFM_DB'].values() for q,p in zip(part['qidxs'],part['pidxs']))
assert ns['PROVENANCE_PATH'].read_text().startswith('num_clusters=100 seed=42\ntrain: 100 clusters, 15,953 images, 34,337 pairs')
prep=json.loads((ROOT/'01_prepare_sfm100_cpu.ipynb').read_text())
export=''.join(prep['cells'][-1]['source'])
fn=next(n for n in ast.parse(export).body if isinstance(n,ast.FunctionDef) and n.name=='copy_verified_image')
export_dir=pathlib.Path(TMP.name)/'export';export_dir.mkdir()
copy_ns=dict(EXPORT=export_dir,SFM_IMAGES=folder,hashlib=hashlib)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<export>','exec'),copy_ns)
relative,info=copy_ns['copy_verified_image'](pathlib.Path(paths[0]))
assert (export_dir/'ims'/relative).read_bytes()==pathlib.Path(paths[0]).read_bytes()
assert info['sha256']==hashlib.sha256(pathlib.Path(paths[0]).read_bytes()).hexdigest()
try:
    copy_ns['copy_verified_image'](pathlib.Path(paths[0]))
except FileExistsError:
    pass
else:raise AssertionError('Export must refuse overwriting files.')
print('PASS: actual subset preparation retains historical counts/removes exactly two self-pairs; exported original image bytes and hashes match; overwrite refused.')
