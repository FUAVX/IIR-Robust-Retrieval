"""Toy checks for the legacy Oxford/Paris AP and mAP implementation.

These tests use only NumPy and synthetic rankings.  They protect the B0
evaluation protocol without requiring a model, a GPU, or benchmark files.
"""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).parents[1] / 'cirtorch' / 'utils' / 'evaluate.py'
_SPEC = importlib.util.spec_from_file_location('b0_evaluate', _MODULE_PATH)
_EVALUATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVALUATE)
compute_ap = _EVALUATE.compute_ap
compute_map = _EVALUATE.compute_map


class ComputeAPTests(unittest.TestCase):
    def test_perfect_ranking_has_ap_one(self):
        self.assertAlmostEqual(compute_ap(np.array([0, 1]), nres=2), 1.0)

    def test_late_positive_has_expected_trapezoidal_ap(self):
        # The only positive is at zero-based rank 2.  The implementation uses
        # trapezoidal integration of the precision-recall curve: AP = 1 / 6.
        self.assertAlmostEqual(compute_ap(np.array([2]), nres=1), 1.0 / 6.0)


class ComputeMapTests(unittest.TestCase):
    def test_junk_before_positive_is_ignored(self):
        # Image IDs 0 and 1 are junk.  Image 3 is the sole positive and is
        # effectively at rank 1 after junk removal, giving AP = 1 / 4.
        ranks = np.array([[0], [1], [2], [3]])
        gnd = [{'ok': [3], 'junk': [0, 1]}]

        map_score, aps, _, _ = compute_map(ranks, gnd)

        self.assertAlmostEqual(map_score, 0.25)
        self.assertAlmostEqual(aps[0], 0.25)

    def test_empty_ground_truth_is_excluded_from_map(self):
        ranks = np.array([[0, 3], [1, 2], [2, 1], [3, 0]])
        gnd = [{'ok': [0], 'junk': []}, {'ok': [], 'junk': []}]

        map_score, aps, _, _ = compute_map(ranks, gnd)

        self.assertAlmostEqual(map_score, 1.0)
        self.assertAlmostEqual(aps[0], 1.0)
        self.assertTrue(np.isnan(aps[1]))

    def test_map_is_mean_of_query_average_precisions(self):
        # Query 0 has AP = 1.  Query 1 has one positive at rank 2, AP = 1/6.
        ranks = np.array([[0, 2], [1, 1], [2, 0], [3, 3]])
        gnd = [{'ok': [0], 'junk': []}, {'ok': [0], 'junk': []}]

        map_score, aps, _, _ = compute_map(ranks, gnd)

        self.assertAlmostEqual(aps[0], 1.0)
        self.assertAlmostEqual(aps[1], 1.0 / 6.0)
        self.assertAlmostEqual(map_score, 7.0 / 12.0)


if __name__ == '__main__':
    unittest.main()
