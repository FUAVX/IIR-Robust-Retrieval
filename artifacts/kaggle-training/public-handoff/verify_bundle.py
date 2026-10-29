"""Verify public handoff file integrity; standard-library Python only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'bundle-manifest.json').read_text())
for name, expected in manifest.items():
    assert Path(name).name == name, name
    data = (root / name).read_bytes()
    assert len(data) == expected['bytes'], name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
print(f'PASS: {len(manifest)} handoff files match the bundle manifest.')
