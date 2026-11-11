"""Focused follow-up to the real v3 log; no need to repeat cache/GPU-128 OOM sweep."""
import copy
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT/'build_profile_notebook.py'))
nb = json.loads((ROOT/'00_profile_a100_training.ipynb').read_text())
nb['cells'][0]['source'] = '''# A100 retest — sửa cache tồn dư và so sánh GPU với cùng DataLoader

Log v3 đã xác nhận peak 30–31,6 GiB và microbatch 128 OOM. Không chạy lại các test 128 hoặc preload cache 30 GiB.
Retest này giải phóng cache cũ trước resource plan, in rõ HOST_RAM/HOST_PLAN và giữ workers=4/prefetch=2/cache=off khi so kích thước GPU.

Thử fusion 1/2/5 × (microbatch/window 16/16, 32/32, 64/64), rồi fine-tune prefetch/window và đo lại finalist 3 lần.
Giữ FP32 numerical reference, loss sum, batch 5 tuple và 400 updates/epoch. Không tự chạy full training.
Giới hạn mềm 45 phút / 6 CU; kết quả trên Drive trong `profile_a100_retest_<timestamp>`.
'''.splitlines(keepends=True)
replacements = {
    'RUN_ID': '"profile_a100_retest_" + time.strftime("%Y%m%d_%H%M%S")',
    'PROFILE_MAX_MINUTES':'45', 'PROFILE_CU_BUDGET':'6.0',
    'PRECISION_CANDIDATES':'["BF16"]', 'WORKER_CANDIDATES':'[4]',
    'PREFETCH_CANDIDATES':'[2, 4]', 'MICROBATCH_CANDIDATES':'[8, 16, 32]',
    'WINDOW_CANDIDATES':'[8, 16, 32, 64]', 'CAPACITY_MICROBATCHES':'[]',
    'CAPACITY_FUSIONS':'[]', 'CACHE_CANDIDATES':'[("off", 0.0)]',
    'CACHE_JOINT_GIB':'[]', 'CACHE_JOINT_WORKERS':'[]',
}
for cell in nb['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type']=='code' and 'PROFILE_MAX_MINUTES = ' in source:
        for key,value in replacements.items():
            source = re.sub(r'^'+re.escape(key)+r' = .*$',key+' = '+value,source,flags=re.M)
    if cell['cell_type']=='code' and 'CONTROLS = dict(' in source:
        source += '''
CONTROLS['enabled_stages'] = ['precision', 'compute_joint', 'prefetch', 'fusion', 'window', 'repeat_finalists']
CONTROLS['compute_joint_fusions'] = [1, 2, 5]
CONTROLS['compute_joint_shapes'] = [(16, 16), (32, 32), (64, 64)]
'''
    cell['source'] = source.splitlines(keepends=True)
    if cell['cell_type']=='code':
        body = source.split('\n',1)[1] if source.startswith('%%writefile ') else source
        compile(body,'retest notebook','exec')
nb['metadata']['colab']['name'] = '00b_retest_a100_training.ipynb'
for index,cell in enumerate(nb['cells']):
    cell['id'] = f'a100-retest-{index:03d}'
target = ROOT/'00b_retest_a100_training.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print(target.name)
