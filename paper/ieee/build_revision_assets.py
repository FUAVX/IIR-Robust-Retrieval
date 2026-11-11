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

# Only fully recovered Oxford is accepted from the budget-limited job.
b0_path = SOURCE/'b0_clean/summary.csv'
if not b0_path.is_file():
    b0_path = DATA/'b0_clean_summary.csv'
if b0_path.is_file():
    b0 = read(b0_path)
    assert len(b0) == 5 and {r['dataset'] for r in b0} == {'oxford5k'}
    labels = [('cosine', 'Cosine'), ('AQE', 'AQE'), ('alpha_QE', r'$\alpha$-QE'),
              ('gated_QE', 'Gated QE'), ('diffusion_mutual_top20', 'Sparse diffusion')]
    b0_lines = []
    for method, label in labels:
        values = []
        row = next(r for r in b0 if r['method'] == method)
        values.extend(f'{100*float(row[k]):.2f}' for k in ['map', 'p10'])
        b0_lines.append(label+' & '+' & '.join(values)+r' \\')
    with (HERE/'tables/revision_rows.tex').open('a') as f:
        f.write('\\newcommand{\\revisionbaselinerows}{%\n'+'\n'.join(b0_lines)+'%\n}\n')
    (HERE/'tables/b0_clean_table.tex').write_text(r'''\begin{table}[t]
\caption{Clean Oxford5k evaluation of the validation-selected B0 epoch-12 checkpoint. All values are percentages; fixed settings and FP32 inference are used.}
\label{tab:b0clean}
\centering\footnotesize
\begin{tabular}{lrr}
\toprule
Method & mAP & P@10 \\
\midrule
\revisionbaselinerows
\bottomrule
\end{tabular}
\end{table}
Table~\ref{tab:b0clean} reports all 55 Oxford queries for the B0 checkpoint selected by validation loss at epoch 12. AQE improves mAP from 81.06\% to 83.77\%, whereas sparse diffusion gives 78.35\%. Thus, correcting seed support does not guarantee a gain for every encoder. The same fixed inference, crop/resize and junk-handling rules apply to all five methods. A 30-minute extraction limit interrupted Paris; only the complete Oxford descriptor and ranking files are recovered, hash-checked and rescored locally. No new Paris result is reported.
''')
    if b0_path.resolve() != (DATA/'b0_clean_summary.csv').resolve():
        shutil.copy2(b0_path, DATA/'b0_clean_summary.csv')
        shutil.copy2(b0_path.parent/'per_query.csv', DATA/'b0_clean_per_query.csv')
        shutil.copy2(b0_path.parent/'recovery_manifest.json', DATA/'b0_clean_recovery_manifest.json')
    print('PASS B0 clean table: five complete Oxford groups; Paris is excluded')
