"""Generate revision tables from saved CPU diagnostics; never runs an encoder."""
import csv
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]/'artifacts/revision-validation/results'
DATA = HERE/'data'
if not SOURCE.is_dir():
    SOURCE = DATA


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


summary = read(SOURCE/'sparse_diffusion_summary.csv')
views = read(SOURCE/'sparse_diffusion_views.csv')
assert len(summary) == 4 and len(views) == 124
lines = []
for model in ['run1_aug_epoch100', 'run2_aug_attention_epoch100']:
    values = []
    for dataset in ['oxford5k', 'paris6k']:
        row = next(r for r in summary if r['model'] == model and r['dataset'] == dataset)
        group = [r for r in views if r['model'] == model and r['dataset'] == dataset]
        clean = next(float(r['map']) for r in group if r['family'] == 'clean')
        means = []
        for family in ['darkening', 'contrast', 'blur', 'noise', 'jpeg', 'occlusion']:
            for severity in [1, 2, 3]:
                selected = [float(r['map']) for r in group if r['family'] == family and int(r['severity']) == severity]
                means.append(sum(selected)/len(selected))
        assert abs(clean-float(row['clean_map'])) < 1e-12
        assert abs(sum(means)/18-float(row['corrupted_map'])) < 1e-12
        values += [f'{100*float(row[k]):.2f}' for k in ['clean_map', 'corrupted_map']]
    lines.append(('R1' if model.startswith('run1') else 'R2')+' & '+' & '.join(values)+r' \\')
macro = '\\newcommand{\\revisiondiffusionrows}{%\n'+'\n'.join(lines)+'%\n}\n'
(HERE/'tables/revision_rows.tex').write_text(macro)
for name in ['sparse_diffusion_summary.csv', 'sparse_diffusion_views.csv', 'sparse_diffusion_manifest.json',
             'diffusion_clean_audit.csv', 'diffusion_clean_audit.json']:
    if (SOURCE/name).resolve() != (DATA/name).resolve():
        shutil.copy2(SOURCE/name, DATA/name)
print('PASS revision table: all four summaries independently recomputed from 124 views')
