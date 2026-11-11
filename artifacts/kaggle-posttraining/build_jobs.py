"""Build reviewable private Kaggle notebook sources; never sends credentials."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OWNER='ndhoang2310'
DATASET=OWNER+'/iir-post100-inputs-v1'
BENCHMARK=OWNER+'/iir-b0-benchmarks'

def build(stage,dependencies=(),dataset=None,shard_index=0,shard_count=1,result_dataset=None):
    suffix='-'+dataset+'-'+str(shard_index) if stage=='geometry' else ''
    slug='iir-post100-'+stage+suffix
    folder=ROOT/'jobs'/slug;folder.mkdir(parents=True,exist_ok=True)
    config=dict(stage=stage,job_id=slug,input='/kaggle/working/input_view',output='/kaggle/working')
    if stage=='geometry':config.update(dataset=dataset,shard_index=shard_index,shard_count=shard_count)
    bootstrap='''import os, sys, json, hashlib
from pathlib import Path
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
root=Path('/kaggle/input')
hits=list(root.rglob('models_manifest.json'))
assert len(hits)==1, hits
code=hits[0].parent
manifest=json.loads(hits[0].read_text())
for name,expected in manifest['code_hashes'].items():
    assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected, name
sys.path.insert(0,str(code))
from artifacts_io import unpack
# Real directories + read-only file symlinks preserve benchmark paths; rglob does
# not follow directory symlinks consistently across Python versions.
view=Path('/kaggle/working/input_view');view.mkdir(exist_ok=True)
for source in root.rglob('*'):
    if source.is_file():
        dest=view/'mounted'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(source)
for index,artifact in enumerate(sorted(root.rglob('artifact_manifest.json'))):
    unpack(artifact,view/'dependencies'/str(index))
import runtime
'''
    source=bootstrap+'\nCONFIG = '+repr(config)+'\ntry:\n    runtime.run(CONFIG)\nfinally:\n    import shutil\n    shutil.rmtree(view, ignore_errors=True)\n'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=[dict(id='intro',cell_type='markdown',metadata={},source=[f'# Post-training epoch 100: {stage}\n','Private exploratory evaluation; fixed settings, no training.\n']),dict(id='run',cell_type='code',metadata={},source=source.splitlines(True),outputs=[],execution_count=None)])
    compile(source,slug,'exec')
    (folder/'notebook.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    metadata=dict(id=OWNER+'/'+slug,title='IIR Post100 '+stage+suffix,code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=stage in ['extract','latency'],enable_tpu=False,enable_internet=False,dataset_sources=[DATASET,BENCHMARK]+([result_dataset] if result_dataset else []),kernel_sources=[OWNER+'/'+x for x in dependencies],competition_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return folder

if __name__=='__main__':
    for stage,deps in [('audit',[]),('extract',['iir-post100-audit']),('qf',['iir-post100-extract']),('local',['iir-post100-extract']),('diffusion',['iir-post100-extract']),('latency',['iir-post100-extract','iir-post100-local','iir-post100-diffusion'])]:print(build(stage,deps))
