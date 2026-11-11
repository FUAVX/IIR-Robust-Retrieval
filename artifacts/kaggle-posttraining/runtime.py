"""Post-training evaluation. Read-only inputs; immutable outputs indexed by config/hash."""
import csv
import hashlib
import io
import json
import math
import os
import pickle
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, __version__ as PIL_VERSION
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import protocol as P
from legacy_evaluate import compute_map


def digest(value):
    if isinstance(value, Path):
        h = hashlib.sha256()
        with value.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
        return h.hexdigest()
    return hashlib.sha256(value if isinstance(value, bytes) else json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def js(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def unit(x):
    return np.asarray(x,dtype=np.float32)/np.maximum(np.linalg.norm(x,axis=0,keepdims=True),1e-12)


def ranking(scores, ids):
    # Descending score, then ascending actual image ID, including exact ties.
    order=np.argsort(np.asarray(ids),kind='stable')
    return order[np.argsort(-np.asarray(scores)[order],axis=0,kind='stable')]


def qe(q, gallery, ids, k=10, alpha=3, lam=.5, tau=None, minimum=3, eligible=None):
    q=np.asarray(q,np.float32); s=gallery.T@q
    r=ranking(s[:,None],ids)[:,0]
    candidates=r[:k] if eligible is None else np.asarray(eligible,dtype=int)[:k]
    candidates=candidates[s[candidates]>0]
    if tau is not None:candidates=candidates[s[candidates]>=tau]
    if lam==0 or not len(candidates) or (tau is not None and len(candidates)<minimum):
        return q.copy(),dict(open=False,feedback=0)
    w=np.maximum(s[candidates],0)**alpha; total=float(w.sum())
    if total<=1e-12:return q.copy(),dict(open=False,feedback=0)
    v=q+lam*(gallery[:,candidates]@(w/total))
    if np.linalg.norm(v)<=1e-12:return q.copy(),dict(open=False,feedback=0)
    return unit(v[:,None])[:,0],dict(open=True,feedback=len(candidates))


def fuse(score_list, ids, method='rrf', c=60):
    if method=='average':return np.mean(score_list,axis=0)
    out=np.zeros_like(score_list[0],dtype=np.float64)
    for scores in score_list:
        r=ranking(scores,ids); positions=np.empty_like(r)
        np.put_along_axis(positions,r,np.broadcast_to(np.arange(1,len(ids)+1)[:,None],r.shape),axis=0)
        out+=1/(c+positions)
    return out


def build_graph(gallery, ids, k=20, mutual=False):
    n=gallery.shape[1]; rows=[]; cols=[]; vals=[]; started=time.perf_counter()
    for start in range(0,n,128):
        scores=gallery.T@gallery[:,start:start+128]
        scores[np.arange(start,min(start+128,n)),np.arange(scores.shape[1])]=-np.inf
        r=ranking(scores,ids)[:min(k,n-1)]
        for j in range(scores.shape[1]):
            v=np.maximum(scores[r[:,j],j],0)**3; keep=v>0
            rows.extend([start+j]*int(keep.sum()));cols.extend(r[keep,j].tolist());vals.extend(v[keep].tolist())
    w=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=np.float64)
    w=w.minimum(w.T) if mutual else w.maximum(w.T)
    w.setdiag(0);w.eliminate_zeros()
    degrees=np.asarray(w.sum(axis=1)).ravel(); inv=np.zeros(n);np.divide(1.,np.sqrt(degrees),out=inv,where=degrees>0)
    s=sparse.diags(inv)@w@sparse.diags(inv)
    stats=dict(edges=int(w.nnz),isolated_fraction=float(np.mean(degrees==0)),components=int(connected_components(w,directed=False)[0]),build_seconds=time.perf_counter()-started)
    return s.tocsr(),stats


def diffuse(scores, graph, beta=.9, max_iterations=100, tolerance=1e-6):
    y=np.maximum(scores,0).astype(np.float64)**3; mass=y.sum(axis=0)
    fallback=(mass<=1e-20);y/=np.maximum(mass,1e-20)
    f=y.copy(); active=~fallback; iterations=np.zeros(scores.shape[1],dtype=int)
    converged=fallback.copy()
    for iteration in range(1,max_iterations+1):
        if not active.any():break
        columns=np.flatnonzero(active);v=(1-beta)*y[:,columns]+beta*(graph@f[:,columns])
        delta=np.linalg.norm(v-f[:,columns],axis=0)/np.maximum(np.linalg.norm(f[:,columns],axis=0),1e-20)
        finite=np.isfinite(v).all(axis=0); bad=columns[~finite];fallback[bad]=True;active[bad]=False
        good=columns[finite];f[:,good]=v[:,finite];iterations[good]=iteration
        done=good[delta[finite]<tolerance];converged[done]=True;active[done]=False
    f[:,fallback]=scores[:,fallback]
    return f,dict(iterations=iterations.tolist(),converged=converged.tolist(),fallback=fallback.tolist())


def prepare_image(path, bbox=None):
    with Image.open(path) as f:image=f.convert('RGB')
    original=max(image.size);target=P.IMAGE_SIZE
    if bbox is not None:
        image=image.crop(tuple(bbox));target=max(1,int(P.IMAGE_SIZE*max(image.size)/original))
    image.thumbnail((target,target),Image.Resampling.LANCZOS)
    return image


def corrupt(image, condition, query_id):
    family=condition['family']
    if family=='clean':return image.copy()
    seed=int(digest(dict(seed=P.SEED,query_id=query_id,condition=condition))[:16],16)
    rng=np.random.default_rng(seed);value=condition['value']
    x=np.asarray(image,dtype=np.float32)/255
    if family=='darkening':x=x*value
    elif family=='contrast':
        mean=x.mean((0,1),keepdims=True);x=mean+value*(x-mean)
    elif family=='blur':
        import cv2
        sigma=max(.1,value*max(image.size));kernel=2*math.ceil(3*sigma)+1
        x=cv2.GaussianBlur(x,(kernel,kernel),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101)
    elif family=='noise':x=x+rng.normal(0,value,x.shape).astype(np.float32)
    elif family=='jpeg':
        b=io.BytesIO();image.save(b,format='JPEG',quality=int(value),subsampling=2)
        b.seek(0)
        with Image.open(b) as f:return f.convert('RGB').copy()
    elif family=='occlusion':
        h,w=x.shape[:2];area=value*h*w;ratio=np.exp(rng.uniform(np.log(.5),np.log(2)))
        rh=max(1,min(h,round(math.sqrt(area/ratio))));rw=max(1,min(w,round(math.sqrt(area*ratio))))
        top=int(rng.integers(0,h-rh+1));left=int(rng.integers(0,w-rw+1));x=x.copy();x[top:top+rh,left:left+rw]=x.mean((0,1))
    else:raise ValueError(family)
    return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8))


def pixel_hash(image):
    return digest(str(image.size).encode()+np.asarray(image,dtype=np.uint8).tobytes())


def local_features(image):
    import cv2
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    detector=cv2.SIFT_create(nfeatures=P.GEO['features'])
    points,desc=detector.detectAndCompute(gray,None)
    if desc is None:return np.zeros((0,2),np.float32),np.zeros((0,128),np.float32)
    desc=np.sqrt(np.maximum(desc,0)/np.maximum(np.abs(desc).sum(1,keepdims=True),1e-12))
    desc/=np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-12)
    return np.asarray([p.pt for p in points],np.float32),desc.astype(np.float32)


def geometry(qfeat, dfeat, query_size, verify=True):
    import cv2
    qp,qd=qfeat;dp,dd=dfeat
    result=dict(matches=0,inliers=0,inlier_ratio=0.,coverage=0.,valid=False)
    if len(qd)<2 or len(dd)<2:return result
    bf=cv2.BFMatcher(cv2.NORM_L2)
    forward=bf.knnMatch(qd,dd,k=2);reverse=bf.match(dd,qd)
    back={m.queryIdx:m.trainIdx for m in reverse}
    matches=[a for pair in forward if len(pair)==2 for a,b in [pair] if a.distance<P.GEO['ratio']*b.distance and back.get(a.trainIdx)==a.queryIdx]
    result['matches']=len(matches)
    if len(matches)<4 or not verify:return result
    src=np.float32([qp[m.queryIdx] for m in matches]);dst=np.float32([dp[m.trainIdx] for m in matches])
    cv2.setRNGSeed(42)
    h,mask=cv2.findHomography(src,dst,cv2.RANSAC,P.GEO['threshold'],maxIters=P.GEO['max_iterations'],confidence=P.GEO['confidence'])
    if h is None or mask is None or not np.isfinite(h).all():return result
    valid=mask.ravel().astype(bool);count=int(valid.sum());result.update(inliers=count,inlier_ratio=count/len(matches))
    if count>=3:result['coverage']=float(cv2.contourArea(cv2.convexHull(src[valid]))/max(1,query_size[0]*query_size[1]))
    result['valid']=bool(count>=P.GEO['minimum_inliers'] and np.linalg.matrix_rank(h)==3 and np.linalg.cond(h)<1e12)
    return result


def rerank(scores, ids, evidence, k=50, eta=.5, geometric=True, query_valid=True):
    r=ranking(scores[:,None],ids)[:,0];top=r[:k]
    if not query_valid or (geometric and not any(evidence.get(int(i),{}).get('valid',False) for i in top)):
        return r,True
    extra=[]
    for i in top:
        e=evidence.get(int(i),{});m=e.get('matches',0);n=e.get('inliers',0)
        extra.append((n/max(m,1))*min(n/20,1) if geometric and e.get('valid',False) else (0 if geometric else min(m/50,1)))
    combined=(1-eta)*(scores[top]+1)/2+eta*np.asarray(extra)
    # Stable ties preserve original global ranking, then image ID.
    order=np.argsort(-combined,kind='stable');return np.concatenate([top[order],r[k:]]),False


def measure(ranks, gt):
    score,aps,_,_=compute_map(ranks,gt['gnd']);rows=[]
    for j,g in enumerate(gt['gnd']):
        usable=ranks[:,j][~np.isin(ranks[:,j],np.asarray(g.get('junk',[]),int))]
        ok=np.asarray(g['ok'],int);positive=np.isin(usable,ok)
        rows.append(dict(ap=float(aps[j]),p10=float(positive[:10].sum()/min(10,len(usable))),
                         recall20=float(positive[:20].sum()/max(1,len(ok))),recall50=float(positive[:50].sum()/max(1,len(ok))),recall100=float(positive[:100].sum()/max(1,len(ok)))))
    return float(score),rows


def find_one(root, name):
    hits=list(Path(root).rglob(name))
    if len(hits)!=1:raise RuntimeError(f'Expected one {name}; found {hits}')
    return hits[0]


def inputs(root):
    # Loading our own trusted benchmark pickle only, after the frozen hash check.
    benchmarks={}
    for dataset in P.DATASETS:
        path=find_one(root,f'gnd_{dataset}.pkl')
        if digest(path)!=P.GND_HASHES[dataset]:raise ValueError('Ground truth hash mismatch: '+dataset)
        gt=pickle.loads(path.read_bytes());images=path.parent/'jpg'
        if not images.is_dir():raise FileNotFoundError(images)
        for image in set(gt['imlist']+gt['qimlist']):
            if not (images/(image+'.jpg')).is_file():raise FileNotFoundError(image)
        if len(set(gt['imlist']))!=len(gt['imlist']):raise ValueError('Duplicate gallery IDs')
        benchmarks[dataset]=dict(gt=gt,images=images)
    return benchmarks


def bundle(root):
    manifest=find_one(root,'models_manifest.json');m=json.loads(manifest.read_text())
    if m['protocol_hash']!=digest(Path(P.__file__)):raise ValueError('Protocol differs from prepared inputs')
    for model in m['models']:
        path=manifest.parent/model['weights_file']
        if digest(path)!=model['weights_sha256']:raise ValueError('Weights checksum mismatch')
        model['path']=str(path)
    return m


def load_model(info, device):
    import torch
    from encoder_runtime import RetrievalModel
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=RetrievalModel(attention=info['config']['attention'])
    model.load_state_dict(torch.load(info['path'],map_location='cpu',weights_only=True),strict=True)
    model.precision='fp32';return model.eval().to(device)


def encode(model,image,device):
    import torch
    x=np.asarray(image,dtype=np.float32).copy()/255
    tensor=torch.from_numpy(x).permute(2,0,1)
    tensor=(tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]
    with torch.inference_mode():v=model(tensor.unsqueeze(0).to(device)).float().cpu().numpy()
    if not np.isfinite(v).all() or v.shape!=(512,1):raise ValueError('Invalid descriptor')
    return unit(v)[:,0]


def desc(root,model,dataset):
    path=find_one(root,f'desc_{model}_{dataset}.npz')
    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    manifest=json.loads(find_one(root,'models_manifest.json').read_text())
    info=next(m for m in manifest['models'] if m['id']==model)
    if str(d['weights_sha256'])!=info['weights_sha256'] or str(d['protocol_hash'])!=manifest['protocol_hash']:raise ValueError('Descriptor provenance mismatch')
    if list(d['conditions'])!=[c['name'] for c in P.conditions()]:raise ValueError('Condition order mismatch')
    if d['queries'].shape!=(512,len(d['query_ids'])*31) or d['gallery'].shape!=(512,len(d['gallery_ids'])):raise ValueError('Descriptor shape mismatch')
    if not np.allclose(np.linalg.norm(d['queries'],axis=0),1,atol=2e-5):raise ValueError('Nonunit query')
    if not np.isfinite(d['gallery']).all() or not np.isfinite(d['queries']).all():raise ValueError('Nonfinite cache')
    if not np.allclose(np.linalg.norm(d['gallery'],axis=0),1,atol=2e-5):raise ValueError('Nonunit gallery')
    return d


class Writer:
    def __init__(self,out,manifest,job):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.models={m['id']:m for m in manifest['models']};self.job=job
    def emit(self,dataset,gt,model,condition,method,config,ids,scores=None,ranks=None,latencies=None,diagnostics=None,phase='main'):
        key=digest(dict(model=model,dataset=dataset,condition=condition,method=method,config=config,protocol=P.VERSION))[:20]
        target=self.out/'results'/key
        if target.with_suffix('.json').exists():return
        r=ranking(scores,ids) if ranks is None else np.asarray(ranks)
        n=len(ids)
        if r.shape!=(n,len(gt['qimlist'])) or not np.all(np.sort(r,axis=0)==np.arange(n)[:,None]):raise ValueError('Ranking not a permutation')
        map_value,metrics=measure(r,gt)
        target.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(target.with_suffix('.npz'),ranks=r.astype(np.uint16 if n<65536 else np.uint32),gallery_ids=ids,query_ids=np.asarray(gt['qimlist']))
        rows=[]
        for j,(qid,metric) in enumerate(zip(gt['qimlist'],metrics)):
            info=self.models.get(model,{});row=dict(protocol=P.VERSION,model=model,checkpoint_hash=info.get('checkpoint_sha256','fusion'),weights_hash=info.get('weights_sha256','fusion'),train_seed=42,dataset=dataset,query_id=qid,landmark_group=qid.rsplit('_',1)[0],method=method,config_hash=digest(config),phase=phase,**condition,**metric,retrieval_ms=float(latencies[j]) if latencies is not None else None,error_status='OK',job=self.job)
            if diagnostics is not None:row.update(diagnostics[j])
            rows.append(row)
        js(target.with_suffix('.json'),dict(key=key,model=model,dataset=dataset,method=method,config=config,phase=phase,condition=condition,map=map_value,rows=rows))
        print(json.dumps(dict(event='RESULT',model=model,dataset=dataset,condition=condition['name'],method=method,map=map_value)),flush=True)


def audit(cfg,benchmarks,manifest,out):
    import torch,cv2,scipy
    facts=dict(protocol=P.VERSION,precision='FP32',exploratory=True,environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,opencv=cv2.__version__,pillow=PIL_VERSION,scipy=scipy.__version__,platform=platform.platform()),models=[{k:v for k,v in x.items() if k!='path'} for x in manifest['models']])
    if len(P.conditions())!=31:raise ValueError('Wrong corruption schedule')
    example=next(iter(benchmarks.values()));gt=example['gt'];p=example['images']/(gt['qimlist'][0]+'.jpg');image=prepare_image(p,gt['gnd'][0]['bbx'])
    corruption_checks=[]
    for c in P.conditions():
        a=corrupt(image,c,gt['qimlist'][0]);b=corrupt(image,c,gt['qimlist'][0]);assert pixel_hash(a)==pixel_hash(b)
        corruption_checks.append(dict(**c,pixel_hash=pixel_hash(a)))
    for info in manifest['models']:
        model=load_model(info,'cpu');v=encode(model,image.resize((64,64)),'cpu');assert np.isclose(np.linalg.norm(v),1,atol=1e-5)
    js(out/'audit.json',dict(**facts,corruption_checks=corruption_checks,status='PASS'))


def extract_stage(cfg,benchmarks,manifest,out):
    import torch,gc
    if not torch.cuda.is_available():raise RuntimeError('GPU extraction stage needs CUDA')
    device='cuda:0';torch.set_num_threads(2)
    log=[];encoder_latency=[]
    for info in manifest['models']:
        model=load_model(info,device)
        for dataset,bench in benchmarks.items():
            gt=bench['gt'];folder=bench['images'];target=out/f"desc_{info['id']}_{dataset}.npz"
            if target.exists():continue
            gallery=[];started=time.perf_counter()
            for j,gid in enumerate(gt['imlist']):
                gallery.append(encode(model,prepare_image(folder/(gid+'.jpg')),device))
                if (j+1)%250==0:print('GALLERY',info['id'],dataset,j+1,len(gt['imlist']),time.perf_counter()-started,flush=True)
            gallery=np.stack(gallery,axis=1);queries=[];pixel_manifest=[]
            for c in P.conditions():
                for j,qid in enumerate(gt['qimlist']):
                    # Timed single uncached query including decode/crop/corruption/encoder.
                    torch.cuda.synchronize();t=time.perf_counter()
                    image=corrupt(prepare_image(folder/(qid+'.jpg'),gt['gnd'][j]['bbx']),c,qid)
                    v=encode(model,image,device);torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                    queries.append(v);pixel_manifest.append(dict(dataset=dataset,query_id=qid,condition=c['name'],pixel_hash=pixel_hash(image),size=list(image.size)))
                    encoder_latency.append(dict(model=info['id'],dataset=dataset,condition=c['name'],query_id=qid,encoder_uncached_ms=elapsed))
                    if info['id']==P.MODEL_IDS[0] and c['name']=='clean' and j<3:
                        image.save(out/f'example_{dataset}_{j}.png')
                print('QUERIES',info['id'],dataset,c['name'],flush=True)
            np.savez_compressed(target,gallery=gallery,queries=np.stack(queries,axis=1),gallery_ids=np.asarray(gt['imlist']),query_ids=np.asarray(gt['qimlist']),conditions=np.asarray([c['name'] for c in P.conditions()]),weights_sha256=np.asarray(info['weights_sha256']),protocol_hash=np.asarray(manifest['protocol_hash']))
            js(out/f"pixels_{info['id']}_{dataset}.json",pixel_manifest)
            log.append(dict(model=info['id'],dataset=dataset,seconds=time.perf_counter()-started,gallery_images=len(gallery.T),query_images=len(queries)))
            js(out/'extraction_progress.json',log);js(out/'encoder_latency.json',encoder_latency)
        del model;gc.collect();torch.cuda.empty_cache()
    # Pixel checksums must be identical across encoders.
    for dataset in P.DATASETS:
        a=json.loads((out/f'pixels_{P.MODEL_IDS[0]}_{dataset}.json').read_text())
        b=json.loads((out/f'pixels_{P.MODEL_IDS[1]}_{dataset}.json').read_text());assert a==b
    js(out/'extraction_environment.json',dict(torch=torch.__version__,pillow=PIL_VERSION,numpy=np.__version__,gpu=torch.cuda.get_device_name(),precision='FP32',tf32=False))


def qe_configs(clean):
    base=dict(k=10,alpha=3,lam=.5)
    yield 'AQE',dict(base,alpha=0),None,'main'
    yield 'alpha_QE',base,None,'main'
    yield 'gated_QE',base,.5,'main'
    if clean:
        variants=[dict(base,k=x) for x in [5,20]]+[dict(base,alpha=x) for x in [1,5]]+[dict(base,lam=x) for x in [.25,1.]]
        for v in variants:yield 'alpha_QE',v,None,'sensitivity'
        for tau in [.4,.6]:yield 'gated_QE',base,tau,'sensitivity'


def qf_stage(cfg,benchmarks,manifest,out):
    root=cfg['input'];writer=Writer(out,manifest,cfg['job_id'])
    for dataset,bench in benchmarks.items():
        caches=[desc(root,m,dataset) for m in P.MODEL_IDS];gt=bench['gt'];nq=len(gt['qimlist']);ids=caches[0]['gallery_ids']
        for d in caches:
            assert list(d['gallery_ids'])==gt['imlist'] and list(d['query_ids'])==gt['qimlist']
        for ci,c in enumerate(P.conditions()):
            all_scores=[]
            for model,d in zip(P.MODEL_IDS,caches):
                q=d['queries'][:,ci*nq:(ci+1)*nq];g=d['gallery'];scores=[];times=[]
                for j in range(nq):
                    t=time.perf_counter();scores.append(g.T@q[:,j]);ranking(scores[-1][:,None],ids);times.append((time.perf_counter()-t)*1000)
                scores=np.stack(scores,axis=1);all_scores.append(scores)
                writer.emit(dataset,gt,model,c,'cosine',{},ids,scores=scores,latencies=times)
                for method,pars,tau,phase in qe_configs(ci==0):
                    values=[];lat=[];diags=[]
                    for j in range(nq):
                        t=time.perf_counter();v,di=qe(q[:,j],g,ids,**pars,tau=tau);values.append(g.T@v);ranking(values[-1][:,None],ids);lat.append((time.perf_counter()-t)*1000);diags.append(dict(qe_gate_open=di['open'],feedback_images=di['feedback']))
                    writer.emit(dataset,gt,model,c,method,dict(pars,tau=tau,minimum=3),ids,scores=np.stack(values,axis=1),latencies=lat,diagnostics=diags,phase=phase)
            for method,cs,phase in [('score_fusion',0,'main'),('RRF',60,'main')]+([('RRF',10,'sensitivity'),('RRF',100,'sensitivity')] if ci==0 else []):
                fused=[];lat=[]
                for j in range(nq):
                    t=time.perf_counter();component=[d['gallery'].T@d['queries'][:,ci*nq+j:ci*nq+j+1] for d in caches]
                    value=fuse(component,ids,'average' if method=='score_fusion' else 'rrf',cs);ranking(value,ids);fused.append(value[:,0]);lat.append((time.perf_counter()-t)*1000)
                writer.emit(dataset,gt,'fusion_run1_run2',c,method,dict(c=cs,models=P.MODEL_IDS),ids,scores=np.stack(fused,axis=1),latencies=lat,phase=phase)


def diffusion_stage(cfg,benchmarks,manifest,out):
    writer=Writer(out,manifest,cfg['job_id'])
    for dataset,bench in benchmarks.items():
        for model in P.MODEL_IDS:
            d=desc(cfg['input'],model,dataset);g=d['gallery'];ids=d['gallery_ids'];nq=len(d['query_ids'])
            graphs={}
            for k in [10,20,40]:
                for mutual in [False,True]:
                    graph,stats=build_graph(g,ids,k,mutual);graphs[k,mutual]=graph
                    sparse.save_npz(out/f'graph_{model}_{dataset}_k{k}_{mutual}.npz',graph)
                    js(out/f'graph_{model}_{dataset}_k{k}_{mutual}.json',stats)
            for ci,c in enumerate(P.conditions()):
                q=d['queries'][:,ci*nq:(ci+1)*nq]
                variants=[(20,.9,False,'main'),(20,.9,True,'main')]
                if ci==0:variants += [(k,.9,m,'sensitivity') for k in [10,40] for m in [False,True]]+[(20,b,m,'sensitivity') for b in [.5,.99] for m in [False,True]]
                for k,beta,mutual,phase in variants:
                    values=[];diags=[];lat=[]
                    for j in range(nq):
                        t=time.perf_counter();s=g.T@q[:,j:j+1];value,di=diffuse(s,graphs[k,mutual],beta);ranking(value,ids)
                        values.append(value[:,0]);lat.append((time.perf_counter()-t)*1000)
                        diags.append(dict(iterations=di['iterations'][0],converged=di['converged'][0],fallback=di['fallback'][0]))
                    writer.emit(dataset,bench['gt'],model,c,'diffusion_mutual' if mutual else 'diffusion_union',dict(k=k,beta=beta,max_iterations=100,tolerance=1e-6),ids,scores=np.stack(values,axis=1),latencies=lat,diagnostics=diags,phase=phase)


def feature_pack(image):
    import cv2
    gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    kp,raw=cv2.SIFT_create(nfeatures=P.GEO['features']).detectAndCompute(gray,None)
    if raw is None:raw=np.zeros((0,128),np.uint8)
    if not np.all(raw==np.rint(raw)):raise ValueError('Noninteger native SIFT; cannot losslessly pack')
    raw=raw.astype(np.uint8 if raw.max(initial=0)<=255 else np.uint16)
    b=io.BytesIO();np.savez_compressed(b,points=np.asarray([k.pt for k in kp],np.float32).reshape(-1,2),raw=raw,size=np.asarray(image.size),pixel_hash=np.asarray(pixel_hash(image)))
    return b.getvalue()


def feature_unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        points=z['points'];raw=z['raw'].astype(np.float32);size=tuple(z['size']);ph=str(z['pixel_hash'])
    v=np.sqrt(raw/np.maximum(raw.sum(1,keepdims=True),1e-12));v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
    return (points,v),size,ph


def feature_key(dataset,image_id,condition='gallery'):
    return dataset+'/'+condition+'/'+image_id


def local_stage(cfg,benchmarks,manifest,out):
    import sqlite3,cv2
    cv2.setNumThreads(1)
    db=sqlite3.connect(str(out/'local_features.sqlite'));db.execute('CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, blob BLOB NOT NULL)')
    pilot=[]
    for dataset,bench in benchmarks.items():
        gt=bench['gt'];folder=bench['images'];started=time.perf_counter();count=0
        expected=json.loads(find_one(cfg['input'],f'pixels_{P.MODEL_IDS[0]}_{dataset}.json').read_text())
        hashes={(x['query_id'],x['condition']):x['pixel_hash'] for x in expected}
        for gid in gt['imlist']:
            key=feature_key(dataset,gid)
            if not db.execute('SELECT 1 FROM features WHERE key=?',(key,)).fetchone():
                db.execute('INSERT INTO features VALUES (?,?)',(key,feature_pack(prepare_image(folder/(gid+'.jpg')))))
            count+=1
            if count%250==0:db.commit();print('LOCAL_GALLERY',dataset,count,time.perf_counter()-started,flush=True)
        for c in P.conditions():
            for j,qid in enumerate(gt['qimlist']):
                key=feature_key(dataset,qid,c['name'])
                if db.execute('SELECT 1 FROM features WHERE key=?',(key,)).fetchone():continue
                image=corrupt(prepare_image(folder/(qid+'.jpg'),gt['gnd'][j]['bbx']),c,qid)
                if pixel_hash(image)!=hashes[qid,c['name']]:raise ValueError('Extraction/local corruption pixels differ; check dependency versions')
                db.execute('INSERT INTO features VALUES (?,?)',(key,feature_pack(image)))
            db.commit();print('LOCAL_QUERY',dataset,c['name'],flush=True)
        pilot.append(dict(dataset=dataset,feature_build_seconds=time.perf_counter()-started,images=len(gt['imlist'])+31*len(gt['qimlist'])))
    db.commit();db.close()
    js(out/'local_cache_environment.json',dict(opencv=cv2.__version__,pillow=PIL_VERSION,lossless_native_sift=True,detector=P.GEO,storage_bytes=(out/'local_features.sqlite').stat().st_size,offline=pilot))
    # Use actual top-ranked candidates to project shard work, including difficult queries.
    store=FeatureStore(out/'local_features.sqlite')
    for dataset,bench in benchmarks.items():
        times=[];gt=bench['gt']
        for model in P.MODEL_IDS:
            d=desc(cfg['input'],model,dataset);g=d['gallery'];nq=len(gt['qimlist'])
            for ci in [0,15,30]:
                for j in [0,min(27,nq-1)]:
                    r=ranking((g.T@d['queries'][:,ci*nq+j])[:,None],d['gallery_ids'])[:10,0]
                    qfeat,size,_=store.get(feature_key(dataset,gt['qimlist'][j],P.conditions()[ci]['name']))
                    for i in r:
                        t=time.perf_counter();dfeat,_,_=store.get(feature_key(dataset,str(d['gallery_ids'][i])));geometry(qfeat,dfeat,size);times.append(time.perf_counter()-t)
        mean=float(np.mean(times));estimate=(2*len(gt['qimlist'])*31*50+2*len(gt['qimlist'])*50)*mean
        # Safety factor 1.5; at most one original query per shard.
        shards=max(1,min(nq,math.ceil(estimate*1.5/(3*3600))))
        js(out/f'geometry_pilot_{dataset}.json',dict(samples=len(times),pair_mean_seconds=mean,pair_p95_seconds=float(np.percentile(times,95)),projected_pair_count=2*nq*31*50+2*nq*50,projected_seconds=estimate,margin=1.5,shards=shards,note='Estimate from sampled real candidates; not a runtime guarantee'))
    store.close()


class FeatureStore:
    def __init__(self,path):
        import sqlite3
        from functools import lru_cache
        self.db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
        self.get=lru_cache(maxsize=128)(self._get)
    def _get(self,key):
        row=self.db.execute('SELECT blob FROM features WHERE key=?',(key,)).fetchone()
        if row is None:raise KeyError(key)
        return feature_unpack(row[0])
    def close(self):self.get.cache_clear();self.db.close()


def geometry_stage(cfg,benchmarks,manifest,out):
    import cv2,sqlite3
    cv2.setNumThreads(1)
    dataset=cfg['dataset'];bench=benchmarks[dataset];gt=bench['gt'];nq=len(gt['qimlist'])
    selected=list(range(cfg['shard_index'],nq,cfg['shard_count']))
    subset=dict(gt);subset['qimlist']=[gt['qimlist'][j] for j in selected];subset['gnd']=[gt['gnd'][j] for j in selected]
    writer=Writer(out,manifest,cfg['job_id']);store=FeatureStore(find_one(cfg['input'],'local_features.sqlite'))
    db=sqlite3.connect(str(out/'geometry_evidence.sqlite'));db.execute('CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    for ci,c in enumerate(P.conditions()):
        for model in P.MODEL_IDS:
            d=desc(cfg['input'],model,dataset);g=d['gallery'];ids=d['gallery_ids'];evidence=[];pairlat=[];qvalid=[];queries=[]
            for j in selected:
                q=d['queries'][:,ci*nq+j];queries.append(q);s=g.T@q;r=ranking(s[:,None],ids)[:,0]
                key=feature_key(dataset,gt['qimlist'][j],c['name']);qfeat,size,_=store.get(key);qvalid.append(len(qfeat[0])>=2)
                record={};timings={}
                for i in r[:100 if ci==0 else 50]:
                    pk=key+'/'+str(ids[i]);cached=db.execute('SELECT value FROM pairs WHERE key=?',(pk,)).fetchone()
                    if cached:entry=json.loads(cached[0])
                    else:
                        t=time.perf_counter();dfeat,_,_=store.get(feature_key(dataset,str(ids[i])));entry=geometry(qfeat,dfeat,size);entry['cached_matching_ms']=(time.perf_counter()-t)*1000
                        db.execute('INSERT INTO pairs VALUES (?,?)',(pk,json.dumps(entry)))
                    record[int(i)]=entry;timings[int(i)]=entry['cached_matching_ms']
                evidence.append(record);pairlat.append(timings);db.commit()
            variants=[(50,.5,False,'main'),(50,.5,True,'main')]
            if ci==0:variants +=[(k,.5,geo,'sensitivity') for k in [20,100] for geo in [False,True]]+[(50,eta,geo,'sensitivity') for eta in [.25,.75] for geo in [False,True]]
            for k,eta,geo,phase in variants:
                columns=[];lat=[];diags=[]
                for jj,j in enumerate(selected):
                    t=time.perf_counter();s=g.T@queries[jj];r=ranking(s[:,None],ids)[:,0];new,fallback=rerank(s,ids,evidence[jj],k,eta,geo,qvalid[jj]);columns.append(new)
                    online_matching=sum(pairlat[jj][int(i)] for i in r[:k]);lat.append((time.perf_counter()-t)*1000+online_matching)
                    diags.append(dict(fallback=fallback,valid_geometry_candidates=sum(evidence[jj][int(i)]['valid'] for i in r[:k]),candidate_relevant_recall_k=float(np.isin(r[:k],gt['gnd'][j]['ok']).sum()/max(1,len(gt['gnd'][j]['ok']))),max_inliers=max([evidence[jj][int(i)]['inliers'] for i in r[:k]] or [0]),matching_ms=online_matching,latency_policy='replayed_measured_matching_not_full_end_to_end'))
                writer.emit(dataset,subset,model,c,'geometric_rerank' if geo else 'local_rerank',dict(k=k,eta=eta,**{a:b for a,b in P.GEO.items() if a not in ['k','eta']}),ids,ranks=np.stack(columns,axis=1),latencies=lat,diagnostics=diags,phase=phase)
            values=[];diags=[];lat=[]
            for jj,j in enumerate(selected):
                t=time.perf_counter();q=queries[jj];s=g.T@q;r=ranking(s[:,None],ids)[:50,0]
                eligible=[int(i) for i in r if s[i]>0 and evidence[jj][int(i)]['valid']][:10]
                if len(eligible)<3:v=q;di=dict(open=False,feedback=0)
                else:v,di=qe(q,g,ids,eligible=eligible,k=10,alpha=3,lam=.5)
                values.append(g.T@v);ranking(values[-1][:,None],ids)
                matching=sum(pairlat[jj][int(i)] for i in r);lat.append((time.perf_counter()-t)*1000+matching)
                diags.append(dict(qe_gate_open=di['open'],feedback_images=di['feedback'],matching_ms=matching,latency_policy='replayed_measured_matching_not_full_end_to_end'))
            writer.emit(dataset,subset,model,c,'verified_QE',dict(k=10,alpha=3,lam=.5,minimum=3,geometry_k=50),ids,scores=np.stack(values,axis=1),latencies=lat,diagnostics=diags)
            print('GEOMETRY',dataset,ci,model,'queries',selected,flush=True)
    store.close();db.close()



def latency_stage(cfg,benchmarks,manifest,out):
    """Direct single-query pipeline benchmark, with no cached query images/features."""
    import torch,cv2,gc
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('Direct encoder latency needs CUDA')
    device='cuda:0';store=FeatureStore(find_one(cfg['input'],'local_features.sqlite'));rows=[]
    methods=['cosine','AQE','alpha_QE','gated_QE','local_rerank','geometric_rerank','verified_QE','diffusion_union','diffusion_mutual']
    conditions=[c for c in P.conditions() if c['name']=='clean' or (c['severity']==3 and c['realization']==0)]
    for dataset,bench in benchmarks.items():
        gt=bench['gt'];folder=bench['images'];caches={m:desc(cfg['input'],m,dataset) for m in P.MODEL_IDS}
        graphcache={(m,mutual):sparse.load_npz(find_one(cfg['input'],f'graph_{m}_{dataset}_k20_{mutual}.npz')) for m in P.MODEL_IDS for mutual in [False,True]}
        models={i['id']:load_model(i,device) for i in manifest['models']}
        # Warm each encoder on a query excluded from the timed sample; one warm geometry call.
        warm=prepare_image(folder/(gt['qimlist'][1]+'.jpg'),gt['gnd'][1]['bbx'])
        for model in models.values():encode(model,warm,device)
        local_features(warm);torch.cuda.synchronize()
        for c in conditions:
            for j in [0,min(27,len(gt['qimlist'])-1)]:
                qid=gt['qimlist'][j]
                for model_id in P.MODEL_IDS:
                    d=caches[model_id];g=d['gallery'];ids=d['gallery_ids']
                    for method in methods:
                        for repeat in range(2):
                            torch.cuda.synchronize();t=time.perf_counter()
                            image=corrupt(prepare_image(folder/(qid+'.jpg'),gt['gnd'][j]['bbx']),c,qid)
                            q=encode(models[model_id],image,device);scores=g.T@q;rank=ranking(scores[:,None],ids)[:,0]
                            if method in ['AQE','alpha_QE','gated_QE']:
                                v,_=qe(q,g,ids,k=10,alpha=0 if method=='AQE' else 3,lam=.5,tau=.5 if method=='gated_QE' else None);rank=ranking((g.T@v)[:,None],ids)[:,0]
                            elif method.startswith('diffusion_'):
                                value,_=diffuse(scores[:,None],graphcache[model_id,method=='diffusion_mutual']);rank=ranking(value,ids)[:,0]
                            elif method in ['local_rerank','geometric_rerank','verified_QE']:
                                qfeat=local_features(image);ev={}
                                for i in rank[:50]:
                                    dfeat,_,_=store.get(feature_key(dataset,str(ids[i])))
                                    ev[int(i)]=geometry(qfeat,dfeat,image.size,verify=method!='local_rerank')
                                if method=='verified_QE':
                                    eligible=[int(i) for i in rank[:50] if scores[i]>0 and ev[int(i)]['valid']][:10]
                                    v=q if len(eligible)<3 else qe(q,g,ids,k=10,alpha=3,lam=.5,eligible=eligible)[0]
                                    rank=ranking((g.T@v)[:,None],ids)[:,0]
                                else:rank,_=rerank(scores,ids,ev,geometric=method=='geometric_rerank',query_valid=len(qfeat[0])>=2)
                            torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                            rows.append(dict(model=model_id,dataset=dataset,query_id=qid,condition=c['name'],method=method,repeat=repeat,end_to_end_ms=elapsed,query_cached=False,gallery_features_cached=True,gpu=torch.cuda.get_device_name()))
                for method in ['score_fusion','RRF']:
                    for repeat in range(2):
                        torch.cuda.synchronize();t=time.perf_counter();components=[]
                        for model_id in P.MODEL_IDS:
                            image=corrupt(prepare_image(folder/(qid+'.jpg'),gt['gnd'][j]['bbx']),c,qid)
                            q=encode(models[model_id],image,device);components.append(caches[model_id]['gallery'].T@q[:,None])
                        value=fuse(components,caches[P.MODEL_IDS[0]]['gallery_ids'],'average' if method=='score_fusion' else 'rrf');ranking(value,caches[P.MODEL_IDS[0]]['gallery_ids']);torch.cuda.synchronize()
                        rows.append(dict(model='fusion_run1_run2',dataset=dataset,query_id=qid,condition=c['name'],method=method,repeat=repeat,end_to_end_ms=(time.perf_counter()-t)*1000,query_cached=False,gallery_features_cached=True,gpu=torch.cuda.get_device_name()))
                write_csv(out/'direct_end_to_end_latency.csv',rows)
                print('LATENCY',dataset,c['name'],qid,flush=True)
        del models;gc.collect();torch.cuda.empty_cache()
    store.close()


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def result_paths(root):
    return list(Path(root).rglob('results/*.json'))+list(Path(root).rglob('result_*.json'))


def report_stage(cfg,benchmarks,manifest,out):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for file in result_paths(cfg['input']):
        value=json.loads(file.read_text());records.extend(value['rows'])
    if not records:raise ValueError('No retrieval results attached')
    df=pd.DataFrame(records)
    unique=['model','dataset','query_id','name','method','config_hash','phase']
    if df.duplicated(unique).any():raise ValueError('Duplicated results; do not attach repeated kernel versions')
    write_csv(out/'per_query.csv',records)
    main=df[df.phase=='main'].copy();metrics=['ap','p10','recall20','recall50','recall100']
    keys=['model','dataset','method','config_hash']
    # Average realizations first, then queries and finally equally weighted 18 conditions.
    averaged=main.groupby(keys+['family','severity','query_id'],dropna=False)[metrics].mean().reset_index()
    conditions=averaged.groupby(keys+['family','severity'],dropna=False)[metrics].mean().reset_index();conditions.to_csv(out/'conditions.csv',index=False)
    summaries=[]
    for key,group in conditions.groupby(keys):
        clean=group[group.family=='clean'];cor=group[group.family!='clean'];rows=main
        for name,value in zip(keys,key):rows=rows[rows[name]==value]
        latency=rows.retrieval_ms.dropna()
        summary=dict(zip(keys,key));summary.update(clean_map=float(clean.ap.mean()),corrupted_map=float(cor.ap.mean()),worst_map=float(cor.ap.min()),drop_pp=float((clean.ap.mean()-cor.ap.mean())*100),conditions=int(len(cor)),p10=float(group.p10.mean()),recall20=float(group.recall20.mean()),recall50=float(group.recall50.mean()),recall100=float(group.recall100.mean()),retrieval_median_ms=float(latency.median()),retrieval_p95_ms=float(latency.quantile(.95)),exploratory=True,training_seeds=1)
        summaries.append(summary)
    write_csv(out/'summary.csv',summaries)
    sensitivity=df[df.phase=='sensitivity'];sensitivity.to_csv(out/'sensitivity_clean.csv',index=False)
    # Coverage is exact at query/condition/method granularity, not just aggregate row counts.
    expected=set()
    main_methods=['cosine','AQE','alpha_QE','gated_QE','local_rerank','geometric_rerank','verified_QE','diffusion_union','diffusion_mutual']
    for dataset,bench in benchmarks.items():
        for q in bench['gt']['qimlist']:
            for c in P.conditions():
                for model in P.MODEL_IDS:
                    for method in main_methods:expected.add((model,dataset,q,c['name'],method))
                for method in ['score_fusion','RRF']:expected.add(('fusion_run1_run2',dataset,q,c['name'],method))
    actual=set(tuple(x) for x in main[['model','dataset','query_id','name','method']].itertuples(index=False,name=None))
    missing=sorted(expected-actual);unexpected=sorted(actual-expected)
    js(out/'coverage.json',dict(status='COMPLETE' if not missing and not unexpected else 'INCOMPLETE',expected=len(expected),actual=len(actual),missing=missing,unexpected=unexpected))
    # Paired query deltas. Fusion is compared to each constituent separately.
    base=averaged[averaged.method=='cosine'];deltas=[]
    for _,row in averaged.iterrows():
        for reference in P.MODEL_IDS if row['model']=='fusion_run1_run2' else [row['model']]:
            match=base[(base.model==reference)&(base.dataset==row.dataset)&(base.family==row.family)&(base.severity==row.severity)&(base.query_id==row.query_id)]
            if len(match)==1:deltas.append(dict(model=row['model'],reference=reference,dataset=row.dataset,method=row.method,query_id=row.query_id,family=row.family,severity=int(row.severity),ap_delta=float(row.ap-match.iloc[0].ap)))
    write_csv(out/'paired_deltas.csv',deltas)
    if deltas:
        dd=pd.DataFrame(deltas);dd.sort_values('ap_delta').head(30).to_csv(out/'largest_regressions.csv',index=False);dd.sort_values('ap_delta',ascending=False).head(30).to_csv(out/'largest_improvements.csv',index=False)
        fractions=[]
        for key,group in dd.groupby(['model','reference','dataset','method']):fractions.append(dict(zip(['model','reference','dataset','method'],key),increased=float((group.ap_delta>1e-6).mean()),decreased=float((group.ap_delta< -1e-6).mean()),unchanged=float((group.ap_delta.abs()<=1e-6).mean())))
        write_csv(out/'query_improvement_rates.csv',fractions)
    for dataset in P.DATASETS:
        fig,axes=plt.subplots(2,3,figsize=(16,8))
        for family,ax in zip(P.CORRUPTIONS,axes.ravel()):
            subset=conditions[(conditions.dataset==dataset)&(conditions.family==family)]
            for (model,method),group in subset.groupby(['model','method']):ax.plot(group.severity,group.ap*100,label=model+'/'+method,marker='.')
            ax.set_title(family);ax.set_xlabel('Severity');ax.set_ylabel('mAP (%)');ax.set_xticks([1,2,3])
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=6);fig.tight_layout(rect=[0,.15,1,1]);fig.savefig(out/f'robustness_{dataset}.png',dpi=140);plt.close(fig)
    summaries_frame=pd.DataFrame(summaries);fig,ax=plt.subplots(figsize=(10,7))
    for _,r in summaries_frame.iterrows():ax.scatter(r.retrieval_median_ms,r.corrupted_map*100);ax.annotate(r['model']+'/'+r['method']+'/'+r.dataset,(r.retrieval_median_ms,r.corrupted_map*100),fontsize=5)
    ax.set_xlabel('Retrieval-only median ms (geometry: replayed matching timings)');ax.set_ylabel('Mean corrupted mAP (%)');fig.tight_layout();fig.savefig(out/'quality_latency.png',dpi=140);plt.close(fig)
    # Compose end-to-end for descriptor-only methods from uncached encoder measurements.
    encoder_paths=list(Path(cfg['input']).rglob('encoder_latency.json'))
    e2e=[]
    if len(encoder_paths)==1:
        lat=pd.DataFrame(json.loads(encoder_paths[0].read_text()))
        for _,r in main.iterrows():
            if r['method'] in ['local_rerank','geometric_rerank','verified_QE']:continue
            models=P.MODEL_IDS if r['model']=='fusion_run1_run2' else [r['model']]
            selected=lat[(lat.model.isin(models))&(lat.dataset==r.dataset)&(lat.condition==r['name'])&(lat.query_id==r.query_id)]
            if len(selected)==len(models):e2e.append(dict(model=r['model'],dataset=r.dataset,method=r['method'],condition=r['name'],query_id=r.query_id,composed_uncached_ms=float(selected.encoder_uncached_ms.sum()+r.retrieval_ms),policy='sum of independently measured single-query encoder and retrieval; not a directly timed combined call'))
    write_csv(out/'composed_latency.csv',e2e)
    direct_paths=list(Path(cfg['input']).rglob('direct_end_to_end_latency.csv'))
    direct_complete=False
    if len(direct_paths)==1:
        direct=pd.read_csv(direct_paths[0]);direct.to_csv(out/'direct_end_to_end_latency.csv',index=False)
        direct.groupby(['model','dataset','method']).end_to_end_ms.agg(median='median',p95=lambda x:x.quantile(.95),samples='count').reset_index().to_csv(out/'end_to_end_summary.csv',index=False)
        direct_complete=(len(direct)==1120 and not direct.duplicated(['model','dataset','query_id','condition','method','repeat']).any())
    files=[]
    for path in Path(cfg['input']).rglob('*'):
        if path.is_file() and (path.name.startswith(('desc_','graph_','local_features','geometry_evidence'))):files.append(dict(file=str(path.relative_to(cfg['input'])),bytes=path.stat().st_size))
    write_csv(out/'cache_storage.csv',files)
    diagnostic_columns=[c for c in ['qe_gate_open','feedback_images','fallback','converged','iterations','valid_geometry_candidates'] if c in main.columns]
    if diagnostic_columns:main.groupby(['model','dataset','method'])[diagnostic_columns].mean().reset_index().to_csv(out/'diagnostics.csv',index=False)
    # Failure figures: fixed rule = largest clean gains, largest clean losses and seeded random queries.
    if deltas:
        clean_delta=pd.DataFrame(deltas);clean_delta=clean_delta[(clean_delta.family=='clean')&(clean_delta.method!='cosine')]
        if len(clean_delta):
            selected=pd.concat([clean_delta.nsmallest(3,'ap_delta'),clean_delta.nlargest(3,'ap_delta'),clean_delta.sample(min(3,len(clean_delta)),random_state=42)]).drop_duplicates(['model','dataset','query_id','method'])
            fig,axes=plt.subplots(len(selected),4,figsize=(12,3*len(selected)),squeeze=False)
            result_files=result_paths(cfg['input'])
            result_index=[(p,json.loads(p.read_text())) for p in result_files]
            for row_index,(_,r) in enumerate(selected.iterrows()):
                bench=benchmarks[r.dataset];gt=bench['gt'];j=gt['qimlist'].index(r.query_id)
                images=[prepare_image(bench['images']/(r.query_id+'.jpg'),gt['gnd'][j]['bbx'])]
                match=next(((p,x) for p,x in result_index if x['model']==r['model'] and x['dataset']==r.dataset and x['method']==r.method and x['phase']=='main' and x['condition']['name']=='clean' and any(v['query_id']==r.query_id for v in x['rows'])),None)
                if match:
                    with np.load(match[0].with_suffix('.npz'),allow_pickle=False) as z:
                        qindex=list(z['query_ids']).index(r.query_id);top=z['ranks'][:3,qindex];gids=z['gallery_ids'][top]
                    images.extend(prepare_image(bench['images']/(str(gid)+'.jpg')) for gid in gids)
                for ax,image in zip(axes[row_index],images):ax.imshow(image);ax.axis('off')
                axes[row_index,0].set_title(str(r['model'])+'/'+r.method+' delta='+format(r.ap_delta,'.3f'),fontsize=7)
            fig.tight_layout();fig.savefig(out/'failure_examples.png',dpi=100);plt.close(fig)
    status='COMPLETE' if not missing and not unexpected and direct_complete else 'INCOMPLETE'
    (out/'REPORT.md').write_text(f'''# Post-training epoch 100 — {status}

Exploratory evaluation. Two models, training seed 42 only; no claim of three-seed reproducibility.
Both encoders use final epoch 100 and FP32 inference with TF32 disabled, not Colab best/BF16 caches.
No configuration is selected by test labels. Sensitivity on clean is descriptive only.

Coverage: {len(actual)}/{len(expected)} main query-method-condition cells. See coverage.json.
Graph construction uses unlabeled test gallery (transductive). No labels used in ranking.
Geometry latency replays measured matching costs; cache I/O is included there, local-query extraction is offline.
Composed latency adds independently measured uncached global extraction and retrieval; it is not a direct end-to-end benchmark.
Direct end-to-end benchmark coverage: {direct_complete}; sampled clean + severe corruption, two queries per dataset/condition, two repeats. See direct_end_to_end_latency.csv.
Negative gains and capped diffusion iterations remain in per_query.csv.
See summary.csv, conditions.csv, sensitivity_clean.csv and paired_deltas.csv.
''')
    js(out/'report_status.json',dict(status=status,main_coverage_complete=not missing and not unexpected,direct_geometry_end_to_end=direct_complete))


def run(cfg):
    import cv2
    cv2.setNumThreads(1)
    out=Path(cfg.get('output','/kaggle/working'))/'posttraining';out.mkdir(parents=True,exist_ok=True)
    started=time.time();js(out/'job_config.json',cfg)
    try:
        benchmarks=inputs(cfg['input']);manifest=bundle(cfg['input'])
        stages=dict(audit=audit,extract=extract_stage,qf=qf_stage,local=local_stage,geometry=geometry_stage,diffusion=diffusion_stage,report=report_stage,latency=latency_stage)
        stages[cfg['stage']](cfg,benchmarks,manifest,out)
        from artifacts_io import pack
        pack(out)
        js(out/'job_status.json',dict(status='COMPLETE',stage=cfg['stage'],job_id=cfg['job_id'],elapsed_seconds=time.time()-started))
    except Exception as error:
        js(out/'job_status.json',dict(status='FAILED',stage=cfg['stage'],job_id=cfg['job_id'],error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started));raise
