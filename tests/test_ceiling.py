"""Checks for the v3 ceiling decomposition (NumPy only, runs without torch)."""
import unittest

import numpy as np

from qword.ceiling import filter_predictions, port_probabilities, prior_shapes
from qword.mobius import observe_probability, start_shape, trajectories


class CeilingTests(unittest.TestCase):
    def test_vectorized_prior_matches_start_shape_distribution(self):
        rng = np.random.default_rng(3)
        loop = np.stack([start_shape(rng) for _ in range(4000)])
        vector = prior_shapes(np.random.default_rng(4), 4000)
        for k in (1, 2):
            a = np.abs(np.exp(1j*k*loop).mean(-1)).mean()
            b = np.abs(np.exp(1j*k*vector).mean(-1)).mean()
            self.assertLess(abs(a - b), .02)
        self.assertEqual(vector.shape, (4000, 16))
        self.assertTrue(np.all(np.abs(vector) <= np.pi))

    def test_port_probabilities_match_observation_law(self):
        theta = prior_shapes(np.random.default_rng(5), 7)
        for action in range(3):
            np.testing.assert_allclose(port_probabilities(theta, action),
                                       [observe_probability(t, action) for t in theta], atol=1e-12)

    def test_point_mass_at_the_truth_reproduces_the_oracle(self):
        for world in ('mobius', 'harmonic'):
            a, y, oracle, shapes = trajectories(world, 4, 12, 91, return_shape=True)
            out = filter_predictions(a, y, world, initial=shapes[:, None, :])
            np.testing.assert_allclose(out['bayes'], oracle, atol=1e-6)
            np.testing.assert_allclose(out['open_loop'], oracle, atol=1e-6)

    def test_open_loop_never_uses_outcomes(self):
        for world in ('mobius', 'harmonic', 'drift'):
            a, y, _ = trajectories(world, 3, 10, 17)
            first = filter_predictions(a, y, world, particles=64, seed=2)
            second = filter_predictions(a, 1 - y, world, particles=64, seed=2)
            np.testing.assert_array_equal(first['open_loop'], second['open_loop'])
            self.assertFalse(np.array_equal(first['bayes'], second['bayes']))

    def test_bayes_filter_is_causal(self):
        for world in ('mobius', 'harmonic', 'drift'):
            a, y, _ = trajectories(world, 2, 9, 23)
            changed = y.copy()
            changed[:, 5:] = 1 - changed[:, 5:]
            first = filter_predictions(a, y, world, particles=64, seed=8)['bayes']
            second = filter_predictions(a, changed, world, particles=64, seed=8)['bayes']
            np.testing.assert_array_equal(first[:, :6], second[:, :6])
            self.assertTrue(np.all((first > .03) & (first < .97)))

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            filter_predictions(np.zeros((2, 3), int), np.zeros((2, 4), int), 'mobius')
        with self.assertRaises(ValueError):
            filter_predictions(np.zeros((2, 3), int), np.zeros((2, 3), int), 'elsewhere')


if __name__ == '__main__':
    unittest.main()
