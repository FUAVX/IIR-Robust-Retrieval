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

# The complete v2 supersedes the historical Oxford-only recovery in the paper.
complete = SOURCE/'b0_complete_v2'
b0_path = complete/'summary.csv'
if not b0_path.is_file():
    complete = DATA
    b0_path = DATA/'b0_complete_summary.csv'
b0 = read(b0_path)
b0_views = read(complete/'views.csv' if complete != DATA else DATA/'b0_complete_views.csv')
assert len(b0) == 10 and len(b0_views) == 310
b0_lookup = {(r['dataset'], r['method']): r for r in b0}
labels = [('cosine', 'Cosine'), ('AQE', 'AQE'), ('alpha_QE', r'$\alpha$-QE'),
          ('gated_QE', 'Gated QE'), ('diffusion_mutual_top20', 'Sparse diffusion')]
b0_lines = []
for method, label in labels:
    values = [f"{100*float(b0_lookup[dataset, method][key]):.2f}"
              for dataset in ['oxford5k', 'paris6k'] for key in ['clean_map', 'corrupted_map']]
    b0_lines.append(label+' & '+' & '.join(values)+r' \\')
with (HERE/'tables/revision_rows.tex').open('a') as f:
    f.write('\\newcommand{\\revisionbaselinerows}{%\n'+'\n'.join(b0_lines)+'%\n}\n')
def macro(method, key):
    return sum(float(b0_lookup[ds, method][key]) for ds in ['oxford5k', 'paris6k'])/2*100
(HERE/'tables/b0_results_table.tex').write_text(r'''\begin{table}[t]
\caption{Complete B0 epoch-12 evaluation, mAP (\%). C: clean; R: mean over 18 corrupted conditions. The checkpoint is selected by validation loss; search settings are fixed.}
\label{tab:b0post}
\centering\footnotesize
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{2}{c}{Oxford} & \multicolumn{2}{c}{Paris} \\
Method & C & R & C & R \\
\midrule
\revisionbaselinerows
\bottomrule
\end{tabular}
\end{table}
Table~\ref{tab:b0post} applies all five fixed methods to the validation-selected B0 checkpoint on both benchmarks. AQE increases clean mAP from 81.06\% to 83.77\% on Oxford and from 77.13\% to 80.19\% on Paris. Its corrupted dataset-macro mAP rises from BASECOR\% to AQECOR\%. Weighted and cosine-gated QE give similar scores.

Sparse diffusion is encoder- and dataset-dependent: Oxford clean mAP falls to 78.35\%, whereas Paris rises to 90.50\%. Its corresponding corrupted scores are 77.46\% and 89.16\%, yielding DIFFCOR\% dataset-macro mAP. Correcting query seeding therefore resolves the earlier collapse without guaranteeing an improvement for every encoder and dataset.

The completed B0 grid contains 310 method/view groups and 17,050 per-query AP records; all 3,410 query pixel checksums match the R1 inputs.
'''.replace('BASECOR', f'{macro("cosine", "corrupted_map"):.2f}')
   .replace('AQECOR', f'{macro("AQE", "corrupted_map"):.2f}')
   .replace('DIFFCOR', f'{macro("diffusion_mutual_top20", "corrupted_map"):.2f}'))

files = {'summary.csv': 'b0_complete_summary.csv', 'views.csv': 'b0_complete_views.csv',
         'per_query.csv': 'b0_complete_per_query.csv', 'comparison_b0_r1_r2.csv': 'b0_complete_comparison.csv',
         'run_manifest.json': 'b0_complete_run_manifest.json', 'status.json': 'b0_complete_status.json',
         'validation_local.json': 'b0_complete_validation.json'}
if complete != DATA:
    for source, target in files.items():
        shutil.copy2(complete/source, DATA/target)
import hashlib
evidence = dict(kernel='ndhoang2310/iir-b0-epoch12-complete-clean-and-robustness-v2/1',
    files={target: dict(bytes=(DATA/target).stat().st_size,
                       sha256=hashlib.sha256((DATA/target).read_bytes()).hexdigest()) for target in files.values()})
(HERE/'b0_complete_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')

# New B0 curves; the original R1 figure remains available as an archived asset.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .18, 'savefig.bbox': 'tight'})
colors = {'cosine': '#2166ac', 'AQE': '#d95f02', 'diffusion_mutual_top20': '#1b7837'}
families = [('darkening', 'Darkening'), ('contrast', 'Low contrast'), ('blur', 'Blur'),
            ('noise', 'Noise'), ('jpeg', 'JPEG'), ('occlusion', 'Occlusion')]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.8), sharey=True)
for ax, (family, title) in zip(axes.flat, families):
    for dataset, style in [('oxford5k', '-'), ('paris6k', '--')]:
        for method, color in colors.items():
            values = [100*float(b0_lookup[dataset, method]['clean_map'])]
            for severity in [1, 2, 3]:
                rr = [r for r in b0_views if r['dataset'] == dataset and r['method'] == method
                      and r['family'] == family and int(r['severity']) == severity]
                values.append(100*sum(float(r['map']) for r in rr)/len(rr))
            ax.plot(range(4), values, style, marker='o', ms=2.5, lw=1.2, color=color)
    ax.set_title(title); ax.set_xticks(range(4), ['Clean', '1', '2', '3']); ax.set_ylim(55, 95)
for ax in axes[:, 0]:
    ax.set_ylabel('mAP (%)')
handles = [plt.Line2D([], [], color=colors[m], label=label) for m, label in
           [('cosine', 'Cosine'), ('AQE', 'AQE'), ('diffusion_mutual_top20', 'Sparse diffusion')]]
handles += [plt.Line2D([], [], color='gray', ls=style, label=name) for name, style in [('Oxford', '-'), ('Paris', '--')]]
fig.legend(handles=handles, ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.045), fontsize=7)
fig.tight_layout(); fig.savefig(HERE/'figures/b0_robustness.pdf'); plt.close(fig)
print('PASS completed B0: ten summary groups, 310 views, source hashes and severity figure')
