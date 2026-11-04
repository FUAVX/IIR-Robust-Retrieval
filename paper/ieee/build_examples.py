"""Illustrative best/worst verified-QE cases from saved AP/rankings, no inference."""
from pathlib import Path
import sys,csv,json,pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'artifacts/kaggle-posttraining'))
import runtime as rt
import protocol as P
OUT=ROOT/'artifacts/kaggle-posttraining/outputs'
with (OUT/'iir-post100-report/unpacked/per_query.csv').open() as f:
 rows=[r for r in csv.DictReader(f) if r['model']=='run1_aug_epoch100' and r['dataset']=='oxford5k' and r['name']=='blur_s3_r0' and r['phase']=='main' and r['method'] in ['cosine','verified_QE']]
paired={}
for r in rows:paired.setdefault(r['query_id'],{})[r['method']]=float(r['ap'])
assert len(paired)==55 and all(len(d)==2 for d in paired.values())
ordered=sorted(paired,key=lambda q:(paired[q]['verified_QE']-paired[q]['cosine'],q))
selected=[ordered[-1],ordered[0]]
with (ROOT/'data/test/oxford5k/gnd_oxford5k.pkl').open('rb') as f:gt=pickle.load(f)
desc=np.load(OUT/'iir-post100-extract/unpacked/desc_run1_aug_epoch100_oxford5k.npz',allow_pickle=False)
assert list(desc['gallery_ids'])==list(gt['imlist'])
condition=next(c for c in P.conditions() if c['name']=='blur_s3_r0')
ci=list(desc['conditions']).index(condition['name'])
qranks={}
for p in OUT.glob('iir-post100-geometry-oxford5k-*/unpacked/results/*.json'):
 j=json.loads(p.read_text())
 if j['model']=='run1_aug_epoch100' and j['method']=='verified_QE' and j['phase']=='main' and j['condition']['name']=='blur_s3_r0':
  with np.load(p.with_suffix('.npz'),allow_pickle=False) as d:
   for n,qid in enumerate(d['query_ids']):
    if qid in selected:qranks[str(qid)]=d['ranks'][:,n].astype(int)
assert set(qranks)==set(selected)
fig,axes=plt.subplots(2,8,figsize=(7.1,3.0))
log=[]
for row,qid in enumerate(selected):
 qi=list(gt['qimlist']).index(qid);g=gt['gnd'][qi]
 clean=rt.prepare_image(ROOT/f'data/test/oxford5k/jpg/{qid}.jpg',g['bbx'])
 corrupt=rt.corrupt(clean,condition,qid)
 scores=desc['gallery'].T@desc['queries'][:,ci*55+qi]
 ranks=rt.ranking(scores[:,None],desc['gallery_ids'])[:,0]
 def nonjunk(r):return [int(i) for i in r if i not in g['junk']][:3]
 cos=nonjunk(ranks);ver=nonjunk(qranks[qid])
 images=[clean,corrupt]+[rt.prepare_image(ROOT/f"data/test/oxford5k/jpg/{gt['imlist'][i]}.jpg") for i in cos+ver]
 for col,(a,im) in enumerate(zip(axes[row],images)):
  a.imshow(im);a.set_xticks([]);a.set_yticks([])
  if col<2:color='gray';label='Clean query' if col==0 else 'Blur severity 3'
  else:
   idx=(cos+ver)[col-2];color='#16813a' if idx in g['ok'] else '#777777' if idx in g['junk'] else '#c92a2a'
   label=('Cosine ' if col<5 else 'Verified ')+str((col-2)%3+1)
  a.set_title(label,fontsize=5.5)
  for spine in a.spines.values():spine.set_visible(True);spine.set_color(color);spine.set_linewidth(1.5)
 delta=paired[qid]['verified_QE']-paired[qid]['cosine']
 axes[row,0].set_ylabel(('Largest gain' if row==0 else 'No improvement')+'\nAP '+f"{paired[qid]['cosine']:.3f} → {paired[qid]['verified_QE']:.3f}",fontsize=6)
 log.append(dict(query_id=qid,condition=condition,pixel_hash=rt.pixel_hash(corrupt),**paired[qid],delta=delta,cosine_top3=[gt['imlist'][i] for i in cos],verified_top3=[gt['imlist'][i] for i in ver]))
fig.tight_layout(pad=.7,w_pad=.4,h_pad=.7);fig.savefig(HERE/'figures/qualitative_cases.pdf',bbox_inches='tight');plt.close(fig)
(HERE/'data/qualitative_cases.json').write_text(json.dumps(log,indent=2))
print(json.dumps(log,indent=2))
