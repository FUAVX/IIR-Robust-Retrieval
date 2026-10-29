"""CPU-only checks for logging and scoped loader adapters; no Kaggle/GPU claim."""
import ast
import io
import json
import pathlib
import re
import threading
import time
from types import SimpleNamespace

root = pathlib.Path(__file__).resolve().parent
nb = json.loads((root / 'notebookc025fac755_optimized.ipynb').read_text())
codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
for i, source in enumerate(codes):
    compile(source, f'cell_{i}', 'exec')
assert all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells'] if c['cell_type'] == 'code')

# Exercise the exact logger class from the notebook with fragmented print writes.
logger_source = next(s for s in codes if 'class SessionTee:' in s)
tree = ast.parse(logger_source)
logger_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SessionTee')
raw, console = io.StringIO(), io.StringIO()
clock = SimpleNamespace(monotonic=lambda: 0.0)
ns = dict(re=re, time=clock, _session_lock=threading.RLock(), _session_file=raw, CONSOLE_PROGRESS_SECONDS=60)
exec(compile(ast.Module(body=[logger_class], type_ignores=[]), '<logger>', 'exec'), ns)
tee = ns['SessionTee'](console)
pieces = ['Epoch', ' ', '8', '\n']
for i in range(10000):
    pieces.extend(['\r>>>> ', str(i+1), '/10000 done...'])
pieces.extend(['\n', 'Traceback: simulated failure\n', 'RuntimeError: test\n'])
for p in pieces:
    tee.write(p)
    tee.flush()  # Old Tee used to flush on each fragment.
tee.finish()
assert raw.getvalue() == ''.join(pieces)
assert 'Epoch 8\n' in console.getvalue()
assert 'RuntimeError: test\n' in console.getvalue()
assert len(console.getvalue().splitlines()) == 4
clock.monotonic = lambda: 61.0
tee.write('>>>> 10000/10000 done...\n')
assert len(console.getvalue().splitlines()) == 5
tee.write('partial final message')
tee.finish()
assert console.getvalue().endswith('partial final message\n')
print('PASS: full raw log preserved; 10,000 progress updates -> 1 visible update; errors visible; timer and final fragment work.')

# Exercise the exact scoped adapters against a stand-in torch module.
adapter_source = next(s for s in codes if 'class _ModuleView:' in s).split('def preview_mining_pool')[0]
calls = []
def loader(*args, **kwargs):
    calls.append(kwargs)
    return kwargs
shared_data = SimpleNamespace(DataLoader=loader)
torch = SimpleNamespace(utils=SimpleNamespace(data=shared_data))
train_module, eval_module = SimpleNamespace(torch=torch), SimpleNamespace(torch=torch)
ns = dict(torch=torch, traindataset=train_module, imageretrievalnet=eval_module,
          os=SimpleNamespace(cpu_count=lambda: 4))
exec(adapter_source, ns)
assert torch.utils.data.DataLoader is loader
assert train_module.torch.utils.data.DataLoader(num_workers=8)['num_workers'] == 4
assert eval_module.torch.utils.data.DataLoader(num_workers=2)['num_workers'] == 2
assert train_module.torch.utils.data.DataLoader(num_workers=0)['num_workers'] == 0
assert train_module.torch.utils.data.DataLoader(batch_size=1, shuffle=False)['batch_size'] == 1
print('PASS: worker cap applies to scoped loaders; shared torch DataLoader stays untouched; zero workers supported.')

# Verify source provenance and historical model/training settings remain intact.
config_source = next(s for s in codes if 'SOURCE_DEFAULTS = {' in s)
assert "'workers': 8" in config_source
assert "'test_freq': 1" in config_source
assert "RESUME_CHECKPOINT = \"/kaggle/working/latest_resume.pt\"" in config_source
assert 'class TwoT4Features' in config_source
controller = next(s for s in codes if 'def atomic_json' in s)
assert 'config_identity(state[\'config\']) == config_identity(cfg)' in controller
assert "torch.stack(flags).all().item()" in controller
assert "timing_epoch_" in controller
assert 'SOURCE_PAYLOAD' in next(s for s in codes if 'SOURCE_HASHES = ' in s)
print('PASS: all cells compile; outputs cleared; resume checks and baseline settings retained.')
