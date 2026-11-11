"""Generate paper tables/figures from saved results; does not train or infer."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
R=ROOT/'artifacts/results-consolidated'
P=ROOT/'artifacts/kaggle-posttraining/outputs/iir-post100-report/unpacked'
if not (R/'all_posttraining_results.csv').is_file() or not (P/'conditions.csv').is_file():
 R=HERE/'data';P=R
CONFIG=R/'report/training_data_config_audit.json'
if not CONFIG.exists(): CONFIG=R/'training_data_config_audit.json'
F=HERE/'figures';T=HERE/'tables';D=HERE/'data'
for p in [F,T,D]:p.mkdir(exist_ok=True)
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def fmt(v):return f'{float(v)*100:.2f}'
def save(name,lines):(T/name).write_text('\n'.join(lines)+'\n')
post=read(R/'all_posttraining_results.csv'); clean=read(R/'clean_model_comparison.csv')
assert len(post)==40 and len(clean)==12
names={'cosine':'Cosine','AQE':'AQE','alpha_QE':r'$\alpha$-QE','gated_QE':'Gated QE','local_rerank':'Local re-rank','geometric_rerank':'Geometric re-rank','verified_QE':'Verified QE','diffusion_mutual':'Mutual diffusion','diffusion_union':'Union diffusion'}
methods=list(names)
lookup={(r['model_short'],r['dataset'],r['method']):r for r in post}
# Full main table, generated to prevent hand transcription errors.
lines=[]
for m in methods:
 vals=[fmt(lookup[model,ds,m][metric]) for model in ['run1','run2'] for ds in ['oxford5k','paris6k'] for metric in ['clean_map','corrupted_map']]
 lines.append(names[m]+' & '+' & '.join(vals)+r' \\')
save('posttraining_rows.tex',lines)
lines=[]
for m,label in [('score_fusion','Score fusion'),('RRF','RRF')]:
 vals=[fmt(lookup['fusion',ds,m][metric]) for ds in ['oxford5k','paris6k'] for metric in ['clean_map','corrupted_map']]
 lines.append(label+' & '+' & '.join(vals)+r' \\')
save('fusion_rows.tex',lines)
selections=[('B0','best_validation','B0 best'),('B0','last_completed_in_snapshot','B0 last available'),('run1','best_validation','R1 best'),('run2','best_validation','R2 best'),('run1','final_epoch_100','R1 final'),('run2','final_epoch_100','R2 final')]
# Selection strings differ in consolidated data; inspect allowed values before resolving.
print('Clean selections:',sorted(set((r['model'],r['selection']) for r in clean)))
lines=[]
for model,sel,label in selections:
 rr=[r for r in clean if r['model']==model and r['selection']==sel]
 if not rr and sel=='final_epoch_100':rr=[r for r in clean if r['model']==model and int(r['epoch'])==100]
 assert len(rr)==2,(model,sel)
 dd={r['dataset']:r for r in rr}
 lines.append(f"{label} & {rr[0]['epoch']} & {fmt(dd['oxford5k']['map'])} & {fmt(dd['paris6k']['map'])}"+r' \\')
save('checkpoint_rows.tex',lines)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
colors={'cosine':'#2166ac','alpha_QE':'#d95f02','verified_QE':'#1b7837'}
hist=read(R/'all_training_history.csv')
fig,ax=plt.subplots(1,2,figsize=(7.0,2.25),sharey=True)
for a,model in zip(ax,['run1','run2']):
 rr=[r for r in hist if r['model']==model];ep=[int(r['epoch']) for r in rr]
 a.plot(ep,[float(r['train_loss']) for r in rr],label='Train',color='#2166ac')
 a.plot(ep,[float(r['val_loss']) for r in rr],label='Validation',color='#b2182b')
 best=min(rr,key=lambda r:float(r['val_loss']));a.axvline(int(best['epoch']),ls=':',color='gray',lw=1)
 a.set(title='R1: augmentation' if model=='run1' else 'R2: augmentation + gate',xlabel='Epoch',xlim=(1,100));a.legend(frameon=False,fontsize=7)
ax[0].set_ylabel('Contrastive loss');fig.tight_layout();fig.savefig(F/'training.pdf');plt.close(fig)
conditions=read(P/'conditions.csv')
fig,axes=plt.subplots(2,3,figsize=(7.0,3.8),sharey=True)
family_names={'darkening':'Darkening','contrast':'Low contrast','blur':'Blur','noise':'Noise','jpeg':'JPEG','occlusion':'Occlusion'}
for a,(family,label) in zip(axes.flat,family_names.items()):
 for ds,style in [('oxford5k','-'),('paris6k','--')]:
  for m in ['cosine','alpha_QE','verified_QE']:
   rr=sorted([r for r in conditions if r['model']=='run1_aug_epoch100' and r['dataset']==ds and r['method']==m and r['family']==family],key=lambda r:int(r['severity']))
   assert len(rr)==3
   ys=[float(lookup['run1',ds,m]['clean_map'])*100]+[float(r['ap'])*100 for r in rr]
   a.plot(range(4),ys,style,marker='o',ms=2.5,lw=1.2,color=colors[m])
 a.set_title(label);a.set_xticks(range(4),['Clean','1','2','3']);a.set_ylim(40,85)
for a in axes[:,0]:a.set_ylabel('mAP (%)')
handles=[plt.Line2D([],[],color=colors[m],label=names[m].replace(r'$\alpha$','Alpha')) for m in colors]+[plt.Line2D([],[],color='gray',ls=sty,label=ds) for ds,sty in [('Oxford','-'),('Paris','--')]]
fig.legend(handles=handles,ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.045),fontsize=7)
fig.tight_layout();fig.savefig(F/'robustness.pdf');plt.close(fig)
lat=read(P/'end_to_end_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(7.0,2.5),sharey=True)
selected=['cosine','alpha_QE','verified_QE','local_rerank','diffusion_mutual','diffusion_union']
short={'cosine':'Cosine','alpha_QE':'Alpha-QE','verified_QE':'Verified QE','local_rerank':'Local','diffusion_mutual':'Mutual','diffusion_union':'Union'}
for a,ds in zip(axes,['oxford5k','paris6k']):
 for i,m in enumerate(selected):
  r=next(r for r in lat if r['model']=='run1_aug_epoch100' and r['dataset']==ds and r['method']==m)
  x=float(r['median']);y=float(lookup['run1',ds,m]['corrupted_map'])*100
  a.scatter(x,y,s=24,color=plt.get_cmap('tab10')(i)); offset={'cosine':(4,-12),'alpha_QE':(4,5),'verified_QE':(-54,5),'local_rerank':(-30,-12),'diffusion_mutual':(4,4),'diffusion_union':(4,4)}[m]
  if ds=='paris6k': offset={'cosine':(10,-2),'alpha_QE':(-20,12),'diffusion_mutual':(8,3),'diffusion_union':(8,-10)}.get(m,offset)
  a.annotate(short[m],(x,y),xytext=offset,textcoords='offset points',fontsize=6.5)
 a.set(xscale='log',xlabel='Direct end-to-end median (ms)',title='Oxford5k' if ds=='oxford5k' else 'Paris6k',xlim=(45,18000),ylim=(15,82))
axes[0].set_ylabel('Corrupted mAP (%)');fig.tight_layout();fig.savefig(F/'quality_latency.pdf');plt.close(fig)
# Per-query examples use mean AP across the same 18 conditions, never independent training replicas.
# Per-query rankings are not needed to rebuild aggregate figures.
# Evidence copies used by the portable source package.
for path in [R/'clean_model_comparison.csv',R/'all_posttraining_results.csv',R/'all_clean_sensitivity.csv',R/'all_training_history.csv',R/'training_summary.csv',CONFIG,P/'conditions.csv',P/'end_to_end_summary.csv',P/'diagnostics.csv',P/'paired_deltas.csv',P/'query_improvement_rates.csv']:
 if path.resolve()!=(D/path.name).resolve(): shutil.copy2(path,D/path.name)
evidence_inputs=[
 ('artifacts/results-consolidated/clean_model_comparison.csv',R/'clean_model_comparison.csv'),
 ('artifacts/results-consolidated/all_posttraining_results.csv',R/'all_posttraining_results.csv'),
 ('artifacts/results-consolidated/all_clean_sensitivity.csv',R/'all_clean_sensitivity.csv'),
 ('artifacts/results-consolidated/all_training_history.csv',R/'all_training_history.csv'),
 ('artifacts/results-consolidated/report/training_data_config_audit.json',CONFIG),
 ('artifacts/kaggle-posttraining/outputs/iir-post100-report/unpacked/conditions.csv',P/'conditions.csv'),
 ('artifacts/kaggle-posttraining/outputs/iir-post100-report/unpacked/end_to_end_summary.csv',P/'end_to_end_summary.csv'),
]
manifest={label:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                 'bytes':p.stat().st_size,
                 'snapshot':'data/'+p.name} for label,p in evidence_inputs}
(HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Supplemental tables retain complete sensitivity and secondary metrics.
sens=read(R/'all_clean_sensitivity.csv');assert len(sens)==100
sl=[]
for r in sens:
 cfg=json.loads(r['config_json']);tokens=[]
 for k in ['k','alpha','lam','tau','eta','beta','c']:
  if k in cfg and cfg[k] is not None:tokens.append(f'{k}={cfg[k]:g}')
 model='R1+R2' if r['model'].startswith('fusion') else ('R1' if r['model'].startswith('run1') else 'R2')
 method=names.get(r['method'],r['method']).replace('_',r'\_')
 sl.append(' & '.join([model,'Ox' if r['dataset']=='oxford5k' else 'Pa',method,', '.join(tokens),fmt(r['clean_map'])])+r' \\')
save('sensitivity_rows.tex',sl)
sl=[]
for r in post:
 model=r['model_short'].replace('run','R').replace('fusion','R1+R2')
 sl.append(' & '.join([model,'Ox' if r['dataset']=='oxford5k' else 'Pa',names.get(r['method'],r['method']).replace('_',r'\_')]+[fmt(r[k]) for k in ['p10','recall20','recall50','recall100','worst_map']])+r' \\')
save('secondary_rows.tex',sl)
print('Generated assets, tables, evidence manifest.')

macros={'checkpoint_rows':'checkpointrows','posttraining_rows':'posttrainingrows','fusion_rows':'fusionrows','sensitivity_rows':'sensitivityrows','secondary_rows':'secondaryrows'}
(T/'generated_rows.tex').write_text('\n'.join('\\newcommand{\\'+macro+'}{%\n'+(T/(file+'.tex')).read_text().rstrip()+'%\n}' for file,macro in macros.items())+'\n')
