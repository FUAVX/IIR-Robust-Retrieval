"""Rebuild the portable LaTeX source ZIP from the current manuscript files."""
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
archive = HERE/'seg301_ieee_source.zip'
allowed = {'.tex', '.bib', '.cls', '.py', '.md', '.sh', '.json', '.csv'}
files = []
for path in HERE.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    relative = path.relative_to(HERE)
    if path.suffix in allowed or (path.suffix == '.pdf' and relative.parts[0] in ['figures', 'template']) or path == HERE/'template/ieee-conference-template.zip':
        files.append(path)
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(files):
        z.write(path, str(path.relative_to(HERE)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for path in files:
        assert z.read(str(path.relative_to(HERE))) == path.read_bytes()
print('PASS source ZIP:', len(files), 'files,', archive.stat().st_size, 'bytes')
