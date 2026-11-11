"""Flatten immutable downloaded outputs when a report has too many source kernels."""
import json,shutil,hashlib
from pathlib import Path

def consolidate(alias,slugs):
    from orchestrate import ROOT,OUTPUTS,OWNER,call
    folder=ROOT/'merged_results';folder.mkdir(exist_ok=True)
    for slug in slugs:
        source=OUTPUTS/slug
        for path in source.rglob('*'):
            if not path.is_file():continue
            if path.parent.name=='results' and path.suffix in ['.json','.npz']:
                name='result_'+hashlib.sha256((slug+'/'+path.stem).encode()).hexdigest()[:24]+path.suffix
            elif path.name.startswith(('desc_','graph_','local_features','geometry_evidence')) or path.name in ['encoder_latency.json','direct_end_to_end_latency.csv']:
                name=path.name
                if name=='geometry_evidence.sqlite':name=slug+'_'+name
            else:continue
            target=folder/name
            if not target.exists():shutil.copy2(path,target)
    ref=OWNER+'/iir-post100-results-v1'
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(id=ref,title='IIR Post100 Results v1',licenses=[dict(name='other')]),indent=2))
    code,message=call(alias,['datasets','create','-p',str(folder)])
    if code:raise RuntimeError('Consolidation upload uncertain. Inspect remote before retry: '+message)
    return ref
