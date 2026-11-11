import unittest
import numpy as np
from scipy import sparse

from diffusion_checks import historical, normalize_affinity, propagate, seeds


class DiffusionReferenceTests(unittest.TestCase):
    def test_two_nodes_has_closed_form(self):
        # f1 + f2 = 1, f2 = beta*f1, so f = [1/(1+b), b/(1+b)].
        graph = sparse.csr_matrix([[0., 1.], [1., 0.]])
        y = np.array([[1.], [0.]])
        actual, diag = propagate(graph, y, beta=.5)
        np.testing.assert_allclose(actual[:, 0], [2/3, 1/3], atol=1e-10)
        self.assertLess(diag['max_relative_residual'], 1e-9)

    def test_components_isolated_node_and_direct_solve(self):
        w = np.array([[0, 2, 1, 0, 0, 0], [2, 0, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0],
                      [0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 0, 0]], float)
        graph = normalize_affinity(w)
        y = np.zeros((6, 2)); y[0, 0] = 1; y[5, 1] = 1
        actual, diag = propagate(sparse.csr_matrix(graph), y)
        expected = np.linalg.solve(np.eye(6)-.9*graph, .1*y)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(actual[3:5], 0)
        self.assertAlmostEqual(actual[5, 1], .1)
        self.assertLess(diag['max_relative_residual'], 1e-9)

    def test_historical_graph_against_dense_reference(self):
        gallery = historical.unit(np.random.default_rng(3).normal(size=(8, 12)))
        ids = np.array([f'image{i:02}' for i in range(12)])
        similarities = gallery.T @ gallery
        directed = np.zeros((12, 12))
        for j in range(12):
            s = similarities[:, j].copy(); s[j] = -np.inf
            neighbors = sorted(range(12), key=lambda i: (-s[i], ids[i]))[:3]
            directed[j, neighbors] = np.maximum(s[neighbors], 0)**3
        for mutual in [False, True]:
            w = np.minimum(directed, directed.T) if mutual else np.maximum(directed, directed.T)
            graph, _ = historical.build_graph(gallery, ids, k=3, mutual=mutual)
            np.testing.assert_allclose(graph.toarray(), normalize_affinity(w), atol=1e-7)
            y = seeds(similarities[:, :2], ids, k=3)
            actual, _ = propagate(graph, y)
            expected = np.linalg.solve(np.eye(12)-.9*graph.toarray(), .1*y)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)

    def test_topk_ties_and_gallery_permutation(self):
        ids = np.array(['c', 'a', 'b', 'd'])
        scores = np.array([[.8], [.8], [.8], [-.2]])
        y = seeds(scores, ids, 2)
        np.testing.assert_array_equal(y[:, 0], [0, .5, .5, 0])
        p = np.array([2, 0, 3, 1]); inverse = np.argsort(p)
        np.testing.assert_allclose(seeds(scores[p], ids[p], 2)[inverse], y)
        np.testing.assert_array_equal(seeds(-np.ones((4, 1)), ids, 2), 0)


if __name__ == '__main__':
    unittest.main()
