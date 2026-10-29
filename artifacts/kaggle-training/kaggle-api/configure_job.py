import json,pathlib
root=pathlib.Path(__file__).resolve().parent
n=json.loads((root.parent/'01_prepare_sfm100_cpu.ipynb').read_text())
export='''# API transport: stream verified original bytes into a single archive.
# The final archive has the same prepared-dataset layout as the folder exporter.
import shutil, zipfile
ARCHIVE = WORKING / 'prepared_subset100.zip'
if ARCHIVE.exists():
    raise RuntimeError('Refusing to overwrite an existing prepared archive.')
needed_bytes = sum(path.stat().st_size for path in referenced)
free_bytes = shutil.disk_usage(WORKING).free
if needed_bytes + 2 * 1024**3 > free_bytes:
    raise RuntimeError(f'Insufficient space for original bytes={needed_bytes}, free={free_bytes}')
exported = {}
stage(f'API export: {len(referenced)} original images, {needed_bytes/1e9:.2f} GB')
with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
    for index, source in enumerate(referenced, 1):
        relative = source.relative_to(SFM_IMAGES).as_posix()
        digest, size = hashlib.sha256(), 0
        with source.open('rb') as inp, archive.open('ims/' + relative, 'w', force_zip64=True) as dest:
            for chunk in iter(lambda: inp.read(1024*1024), b''):
                dest.write(chunk); digest.update(chunk); size += len(chunk)
        exported[relative] = dict(bytes=size, sha256=digest.hexdigest())
        if index % 500 == 0:
            print(f'Export: {index}/{len(referenced)} images', flush=True)
    for source in [SFM_PKL, FILTER_PATH, PROVENANCE_PATH]:
        archive.write(source, source.name)
    manifest = dict(format='b0_prepared_subset_v1', dataset_identity=DATASET_IDENTITY,
        metadata_files={p.name: sha256_file(p) for p in [SFM_PKL,FILTER_PATH,PROVENANCE_PATH]},
        sfm_images=exported, full_image_decode_preflight=True,
        note='Original image bytes; no resize/recompression. API-transport archive; same prepared layout.')
    archive.writestr('prepared_manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
summary = dict(status='COMPLETE', images=len(exported), original_image_bytes=needed_bytes,
               archive=ARCHIVE.name, archive_bytes=ARCHIVE.stat().st_size,
               archive_sha256=sha256_file(ARCHIVE), dataset_identity=DATASET_IDENTITY)
(WORKING / 'preparation_result.json').write_text(json.dumps(summary,indent=2))
print('PREPARATION COMPLETE:', json.dumps(summary,indent=2))
_session_log_stop()
'''
n['cells'][-1]['source']=export.splitlines(keepends=True)
n['metadata']['kaggle'].update(isGpuEnabled=False,isInternetEnabled=False)
n['metadata']['kaggle'].pop('accelerator',None)
n['metadata']['kaggle']['dataSources']=[s for s in n['metadata']['kaggle'].get('dataSources',[]) if s['sourceType']=='datasetVersion']
for i,c in enumerate(n['cells']):
 if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
folder=root/'prepare-kernel'
(folder/'prepare.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
metadata=dict(id='ndhoang2310/iir-b0-sfm100-cpu-preparation',title='IIR B0 SfM100 CPU Preparation',
 code_file='prepare.ipynb',language='python',kernel_type='notebook',is_private=True,
 enable_gpu=False,enable_tpu=False,enable_internet=False,
 dataset_sources=['ndhoang2310/iir-b0-sfm-full','ndhoang2310/iir-b0-benchmarks','ndhoang2310/iir-b0-vgg16-weights'],
 competition_sources=[],kernel_sources=[],model_sources=[])
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Prepared CPU kernel package:',folder)
