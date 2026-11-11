"""Frozen exploratory protocol. No tuning on test relevance labels."""
VERSION = 'post100_v1'
MODEL_IDS = ['run1_aug_epoch100', 'run2_aug_attention_epoch100']
DATASETS = ['oxford5k', 'paris6k']
GND_HASHES = {'oxford5k':'c84e5eb68524fd46aab5c0e80b90f970d7bf677507aeb8cddd6b12990d4bc19e',
              'paris6k':'def916cf4f197a182d9a7840cdb6294489fa57c2121f9742a1bcc6f2209bb345'}
SEED = 20261030
IMAGE_SIZE = 1024
QE = dict(k=10, alpha=3, lam=.5, tau=.5, minimum=3)
GEO = dict(k=50, eta=.5, features=2000, ratio=.75, threshold=3., confidence=.995, max_iterations=2000, minimum_inliers=8)
DIFF = dict(k=20, beta=.9, max_iterations=100, tolerance=1e-6)
RRF_C = 60
CORRUPTIONS = {'darkening':[.8,.6,.4], 'contrast':[.8,.6,.4], 'blur':[.001,.002,.004],
               'noise':[.02,.05,.08], 'jpeg':[80,50,25], 'occlusion':[.05,.1,.2]}

def conditions():
    result = [dict(name='clean', family='clean', severity=0, realization=0, value=None)]
    for family, values in CORRUPTIONS.items():
        for severity, value in enumerate(values, 1):
            for realization in range(3 if family in ['noise','occlusion'] else 1):
                result.append(dict(name=f'{family}_s{severity}_r{realization}',family=family,
                                   severity=severity,realization=realization,value=value))
    return result
