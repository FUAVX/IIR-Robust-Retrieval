"""CPU checks of resumable prefixes, real image stages and condition averaging."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'kaggle-posttraining'))
import protocol as P
import run_b0_complete as B


class ToyEncoder(torch.nn.Module):
    def forward(self, x):
        return torch.arange(1, 513, dtype=torch.float32, device=x.device)[:, None]+x.mean()


class B0CompletionTests(unittest.TestCase):
    def test_prefix_survives_uncommitted_tail_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            s = B.ArrayStore(root/'old', 'sample', (2, 4), 'same')
            s.append([1, 0]); s.append([0, 1]); s.save()
            s.append([.6, .8]); s.array.flush()  # Simulate a kill before JSON commit.
            restored = B.ArrayStore(root/'new', 'sample', (2, 4), 'same', root/'old')
            self.assertEqual(restored.completed, 2)
            np.testing.assert_array_equal(restored.array[:, :2], np.eye(2))
            restored.append([.6, .8]); restored.append([1, 0]); restored.save()
            restored.array[0, 0] = .5; restored.array.flush()
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                B.ArrayStore(root/'bad', 'sample', (2, 4), 'same', root/'new')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                B.ArrayStore(root/'other', 'sample', (2, 4), 'changed', root/'old')

    def test_real_image_extraction_and_all_search_stages_on_cpu(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root/'out'; output.mkdir()
            ids = np.array([f'g{i}' for i in range(6)]); qids = ids[:2]
            paths = []
            for i, gid in enumerate(ids):
                path = root/(gid+'.jpg')
                Image.fromarray(np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(path)
                paths.append(path)
            gt = dict(imlist=ids.tolist(), qimlist=qids.tolist(), gnd=[dict(ok=[2, 3], junk=[0]), dict(ok=[4, 5], junk=[1])])
            gallery = B.ArrayStore(root/'arrays', 'gallery', (512, 6), 'gallery')
            started = time.monotonic()
            B.fill(ToyEncoder(), gallery, B.Images(paths), started, soft_seconds=30, device='cpu', workers=0)
            conditions = P.conditions()
            pairs = [(paths[j], c, str(qid)) for c in conditions for j, qid in enumerate(qids)]
            queries = B.ArrayStore(root/'arrays', 'queries', (512, 62), 'queries')
            tasks = B.Images([p[0] for p in pairs], [[0, 0, 48, 48]]*62,
                             [p[1] for p in pairs], [p[2] for p in pairs])
            B.fill(ToyEncoder(), queries, tasks, started, soft_seconds=30, device='cpu', workers=0)
            self.assertEqual(len(queries.pixels), 62)
            views, metrics = B.evaluate_dataset('oxford5k', gt, gallery.array, queries.array,
                                               ids, conditions, output, started, soft_seconds=30)
            self.assertEqual(len(views), 155); self.assertEqual(len(metrics), 310)
            self.assertEqual(len(list(output.glob('ranks_*.npz'))), 155)
            self.assertEqual(len(list((output/'conditions').glob('*.json'))), 31)
            self.assertTrue(all(np.isfinite(row['map']) for row in views))

    def test_realization_weights_are_not_pooled(self):
        views = []
        for dataset in ['oxford5k', 'paris6k']:
            for method in B.METHODS:
                for c in P.conditions():
                    value = .8 if c['family'] == 'clean' else float(c['family'] == 'noise')
                    views.append(dict(dataset=dataset, method=method, family=c['family'],
                                      severity=c['severity'], map=value, p10=value))
        summary = B.summarize(views)
        self.assertEqual(len(summary), 10)
        for row in summary:
            self.assertAlmostEqual(row['clean_map'], .8)
            self.assertAlmostEqual(row['corrupted_map'], 1/6)  # Pooling views would give .3.
            self.assertEqual(row['worst_map'], 0)


if __name__ == '__main__':
    unittest.main()
