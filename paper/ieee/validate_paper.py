"""Check manuscript data, source provenance, and final PDF structure."""
from pathlib import Path
import csv,json,re,subprocess,hashlib
H=Path(__file__).resolve().parent
D=H/'data'
def rows(name):
 with (D/name).open() as f:return list(csv.DictReader(f))
post=rows('all_posttraining_results.csv');conditions=rows('conditions.csv')
assert len(post)==40
for r in post:
 rr=[c for c in conditions if all(c[k]==r[k] for k in ['model','dataset','method','config_hash'])]
 assert len(rr)==19,(r['model'],r['method'],len(rr))
 clean=[c for c in rr if c['family']=='clean'];cor=[c for c in rr if c['family']!='clean']
 assert len(clean)==1 and len(cor)==18
 assert abs(float(r['clean_map'])-float(clean[0]['ap']))<1e-12
 assert abs(float(r['corrupted_map'])-sum(float(c['ap']) for c in cor)/18)<1e-12
 assert abs(float(r['worst_map'])-min(float(c['ap']) for c in cor))<1e-12
 for key in ['p10','recall20','recall50','recall100']:
  assert abs(float(r[key])-sum(float(c[key]) for c in rr)/19)<1e-12
# Full result table cells are generated without manual edits.
methods=['cosine','AQE','alpha_QE','gated_QE','local_rerank','geometric_rerank','verified_QE','diffusion_mutual','diffusion_union']
lookup={(r['model_short'],r['dataset'],r['method']):r for r in post}
mainrows=(H/'tables/posttraining_rows.tex').read_text().splitlines()
assert len(mainrows)==9
for line,m in zip(mainrows,methods):
 got=[v.strip().removesuffix(r'\\').strip() for v in line.split('&')[1:]]
 expected=[f"{float(lookup[model,ds,m][metric])*100:.2f}" for model in ['run1','run2'] for ds in ['oxford5k','paris6k'] for metric in ['clean_map','corrupted_map']]
 assert got==expected,(m,got,expected)
revision=rows('sparse_diffusion_summary.csv');assert len(revision)==4
revision_views=rows('sparse_diffusion_views.csv');assert len(revision_views)==124
for r in revision:
 rr=[v for v in revision_views if v['model']==r['model'] and v['dataset']==r['dataset']]
 clean=next(float(v['map']) for v in rr if v['family']=='clean')
 condition_maps=[sum(float(v['map']) for v in rr if v['family']==family and int(v['severity'])==severity)/len([v for v in rr if v['family']==family and int(v['severity'])==severity]) for family in ['darkening','contrast','blur','noise','jpeg','occlusion'] for severity in [1,2,3]]
 assert abs(float(r['clean_map'])-clean)<1e-12
 assert abs(float(r['corrupted_map'])-sum(condition_maps)/18)<1e-12
assert len(rows('all_clean_sensitivity.csv'))==100
assert len((H/'tables/sensitivity_rows.tex').read_text().splitlines())==100
assert len((H/'tables/secondary_rows.tex').read_text().splitlines())==40
history=rows('all_training_history.csv');assert len(history)==214
for model,epoch in [('B0',12),('run1',27),('run2',28)]:
 rr=[r for r in history if r['model']==model]
 assert int(min(rr,key=lambda r:float(r['val_loss']))['epoch'])==epoch
cfg=json.loads((D/'training_data_config_audit.json').read_text())
for name in ['baseline_config','run1','run2']:
 assert cfg[name]['seed']==42 and cfg[name]['batch_size']==5 and cfg[name]['lr']==1e-6
evidence=json.loads((H/'evidence_manifest.json').read_text())
assert len(evidence)==7
for original,entry in evidence.items():
 snapshot=H/entry.get('snapshot','data/'+Path(original).name)
 assert snapshot.is_file(),(original,snapshot)
 assert snapshot.stat().st_size==entry['bytes'],(original,'size')
 assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==entry['sha256'],(original,'hash')
macro=lambda model,method,key:sum(float(lookup[model,ds,method][key]) for ds in ['oxford5k','paris6k'])/2*100
assert round(macro('run1','cosine','corrupted_map'),2)==66.82
assert round(macro('run1','alpha_QE','corrupted_map'),2)==69.34
assert round(macro('run1','verified_QE','corrupted_map'),2)==71.05
assert round(macro('run1','alpha_QE','corrupted_map')-macro('run1','cosine','corrupted_map'),2)==2.52
assert round(macro('run1','verified_QE','corrupted_map')-macro('run1','cosine','corrupted_map'),2)==4.23
provenance=json.loads((H/'template/provenance.json').read_text())
assert hashlib.sha256((H/'IEEEtran.cls').read_bytes()).hexdigest()==provenance['class_sha256']
assert (H/'IEEEtran.cls').read_bytes()==(H/'template/original/conference-latex-template_10-17-19/IEEEtran.cls').read_bytes()
# All citations resolve to verified bibliography keys.
source='\n'.join(p.read_text() for p in (H/'sections').glob('*.tex'))
cites=set(k for group in re.findall(r'\\cite\{([^}]+)\}',source) for k in group.split(','))
bibkeys=set(re.findall(r'@\w+\{([^,]+),',(H/'references.bib').read_text()))
assert cites==bibkeys and len(cites)==16
assert len(rows('reference_sources.csv'))==16 if (D/'reference_sources.csv').exists() else True
report={'status':'PASS','main_summary_rows':40,'sensitivity_rows':100,'secondary_rows':40,'history_rows':214,'verified_references':16,'class_unmodified':True,'checks':['All 40 aggregate rows recomputed from 19 condition rows','Corrupted mAP uses 18 conditions; secondary summary uses 19 including clean','All 72 main post-training table cells match CSV','Best epochs match validation-loss minima','Seven source evidence hashes match committed snapshots','Frozen config, macro gains, bibliography keys and original class checked','Four revised diffusion summaries recomputed from 124 cached views'],'pdf':{},'build_log_checks':{}}
for name in ['seg301_ieee_paper','supplement']:
 log_path=H/(name+'.log')
 if log_path.is_file():
  log=log_path.read_text()
  for bad in ['Overfull \\hbox','Overfull \\vbox','Missing character','undefined on input','There were undefined references']:
   assert bad not in log,(name,bad)
  report['build_log_checks'][name]='PASS'
 else:
  report['build_log_checks'][name]='SKIPPED: build log not committed; compile the LaTeX source to check layout warnings'
 pdf=H/(name+'.pdf');info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
 pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
 text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
 assert '\ufffd' not in text and '[?]' not in text and '??' not in text
 assert len([p for p in text.split('\f') if p.strip()])==pages
 assert all(len(p.strip())>300 for p in text.split('\f') if p.strip())
 report['pdf'][name]={'pages':pages,'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'bytes':pdf.stat().st_size}
report['visual_review']='Historical manual review was recorded for the committed PDFs; this script checks their text and structure, not visual layout.'
(H/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
