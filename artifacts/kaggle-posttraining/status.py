"""Read-only local view of persisted Kaggle stage states."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent/'state'
for path in sorted(root.glob('iir-post100-*.json')):
    d=json.loads(path.read_text());print(d['slug'],d['status'],d.get('remote',''),d.get('error',''))
p=root/'pipeline_status.json'
if p.exists():print(p.read_text())
