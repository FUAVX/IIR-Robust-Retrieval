"""Check manuscript data, source provenance, and final PDF structure."""
from pathlib import Path
import csv,json,re,subprocess,hashlib,collections,math
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
b0=rows('b0_complete_summary.csv');b0_query=rows('b0_complete_per_query.csv');b0_views=rows('b0_complete_views.csv')
assert len(b0)==10 and len(b0_query)==17050 and len(b0_views)==310
status=json.loads((D/'b0_complete_status.json').read_text())
assert status['status']=='COMPLETE' and status['ap_records']==17050
manifest=json.loads((D/'b0_complete_run_manifest.json').read_text())
checks=json.loads((D/'b0_complete_validation.json').read_text())
assert checks['status']=='PASS' and checks['script_sha_verified'] and checks['input_manifest_verified']
assert checks['query_pixel_checksums_matching_R1']==3410
assert checks['rankings_verified']==310 and checks['ap_values_independently_rescored']==17050
assert manifest['input_manifest']['epoch']==12 and manifest['input_manifest']['selection']=='validation_loss'
methods_b0=set(manifest['methods']);condition_names={c['name'] for c in manifest['query_views']}
expected={(ds,m,c) for ds in ['oxford5k','paris6k'] for m in methods_b0 for c in condition_names}
assert len(expected)==310
vl={(r['dataset'],r['method'],r['condition']):r for r in b0_views};assert set(vl)==expected
groups=collections.defaultdict(list)
for r in b0_query:
 assert math.isfinite(float(r['ap'])) and -1e-12<=float(r['ap'])<=1+1e-12
 groups[r['dataset'],r['method'],r['condition']].append(r)
assert set(groups)==expected
for key,rr in groups.items():
 assert len(rr)==55 and len({r['query_id'] for r in rr})==55
 assert abs(sum(float(r['ap']) for r in rr)/55-float(vl[key]['map']))<1e-12
for r in b0:
 rr=[v for v in b0_views if v['dataset']==r['dataset'] and v['method']==r['method']]
 clean=next(float(v['map']) for v in rr if v['family']=='clean')
 maps=[sum(float(v['map']) for v in rr if v['family']==family and int(v['severity'])==severity)/len([v for v in rr if v['family']==family and int(v['severity'])==severity]) for family in ['darkening','contrast','blur','noise','jpeg','occlusion'] for severity in [1,2,3]]
 assert abs(float(r['clean_map'])-clean)<1e-12
 assert abs(float(r['corrupted_map'])-sum(maps)/18)<1e-12
 assert abs(float(r['worst_map'])-min(maps))<1e-12
baseline_macro=re.search(r'\\newcommand\{\\revisionbaselinerows\}\{%\n(.*?)%\n\}', (H/'tables/revision_rows.tex').read_text(), re.S)
assert baseline_macro
baseline_lines=baseline_macro.group(1).splitlines();assert len(baseline_lines)==5
bl={(r['dataset'],r['method']):r for r in b0}
for line,method in zip(baseline_lines,['cosine','AQE','alpha_QE','gated_QE','diffusion_mutual_top20']):
 got=[v.strip().removesuffix(r'\\').strip() for v in line.split('&')[1:]]
 expected=[f"{100*float(bl[ds,method][metric]):.2f}" for ds in ['oxford5k','paris6k'] for metric in ['clean_map','corrupted_map']]
 assert got==expected,(method,got,expected)
b0_evidence=json.loads((H/'b0_complete_evidence.json').read_text())
for name,entry in b0_evidence['files'].items():
 assert (D/name).stat().st_size==entry['bytes']
 assert hashlib.sha256((D/name).read_bytes()).hexdigest()==entry['sha256']
comparison=rows('b0_complete_comparison.csv');assert len(comparison)==30
for r in comparison:
 source=b0 if r['model']=='b0_epoch12' else (revision if r['method']=='diffusion_mutual_top20' else post)
 source_row=next(x for x in source if all(x[k]==r[k] for k in ['model','dataset','method']))
 for metric in ['clean_map','corrupted_map','worst_map']:assert abs(float(r[metric])-float(source_row[metric]))<1e-12
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
report={'status':'PASS','main_summary_rows':40,'sensitivity_rows':100,'secondary_rows':40,'history_rows':214,'verified_references':16,'b0_summary_rows':10,'b0_ap_records':17050,'class_unmodified':True,'checks':['All 40 aggregate rows recomputed from 19 condition rows','Corrupted mAP uses 18 conditions; secondary summary uses 19 including clean','All 72 main post-training table cells match CSV','Best epochs match validation-loss minima','Seven source evidence hashes match committed snapshots','Frozen config, macro gains, bibliography keys and original class checked','Four revised diffusion summaries recomputed from 124 cached views','Ten B0 groups and 310 views recomputed from 17050 AP records; both datasets complete','Complete B0 snapshot hashes and 30-row comparison verified','All 20 complete B0 table cells match the CSV'],'pdf':{},'build_log_checks':{}}
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
report['visual_review']='This script checks text and structure, not visual layout. The revision manual-review record is under artifacts/revision-validation/results/validation.json.'
(H/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
