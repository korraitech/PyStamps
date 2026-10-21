"""Run from the repository root with python3 -m unittest experiments.handwired_topofit.test_cpu."""

import unittest

import numpy as np

from experiments.handwired_topofit import cpu


class TrialSearchTests(unittest.TestCase):
    def test_fixture_reproduces_step2_random_phase_layout(self):
        phase, baseline, wraps = cpu.fixture(7, n_ifg=9)
        expected = np.exp(1j * (2 * np.pi * np.random.RandomState(2005).rand(9, 7)).T)
        np.testing.assert_array_equal(phase, expected)
        self.assertEqual(phase.dtype, np.complex128)
        self.assertEqual(baseline.dtype, np.float64)
        self.assertEqual(baseline.shape, (9,))
        self.assertEqual(wraps, 2.0)

    def test_random_search_and_refinement_match_actual_source(self):
        phase, baseline, wraps = cpu.fixture(13, n_ifg=11, n_trial_wraps=1.3)
        phase *= np.random.RandomState(8).uniform(0.2, 2.0, phase.shape)
        sums, selected = cpu.source_search(phase, baseline, wraps)
        weights = cpu.trial_weights(baseline, wraps)
        self.assertEqual(sums.shape, (13, 23))
        batched, batched_selected = cpu.batched_search(phase, weights, chunk_size=4)
        np.testing.assert_allclose(batched, sums, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(batched_selected, selected)
        actual = cpu.refine(phase, baseline, wraps, selected)
        for row in range(len(phase)):
            expected = cpu.reference_topofit(phase[row], baseline, wraps)
            for key, value in zip(("K", "C", "coh", "phase_residual"), expected):
                np.testing.assert_allclose(actual[key][row], np.asarray(value).squeeze(), atol=1e-14)

    def test_coherent_inputs_recover_trial_slope(self):
        baseline = np.linspace(-100, 100, 17)
        slopes = np.array([-4, 0, 3]) * np.pi / 4 / np.ptp(baseline)
        phase = np.exp(1j * (slopes[:, None] * baseline + 0.37))
        _, selected = cpu.source_search(phase, baseline, 1)
        np.testing.assert_array_equal(selected, [4, 8, 11])
        result = cpu.refine(phase, baseline, 1, selected)
        np.testing.assert_allclose(result["K"], slopes, atol=1e-15)
        np.testing.assert_allclose(result["C"], 0.37, atol=1e-15)
        np.testing.assert_allclose(result["coh"], 1, atol=1e-15)

    def test_first_exact_tie_and_fp32_near_tie_are_distinct(self):
        phase = np.array([[1, 1]], dtype=np.complex128)
        weights = np.ones((2, 3), dtype=np.complex128)
        _, selected = cpu.batched_search(phase, weights)
        np.testing.assert_array_equal(selected, [0])
        weights[:, 1] += 1e-9
        _, selected64 = cpu.batched_search(phase, weights)
        sums32, selected32 = cpu.batched_search(phase.astype(np.complex64), weights.astype(np.complex64))
        self.assertEqual(sums32.dtype, np.complex64)
        np.testing.assert_array_equal(selected64, [1])
        np.testing.assert_array_equal(selected32, [0])

    def test_source_uses_first_maximum_on_tied_trials(self):
        phase = np.ones((1, 2), dtype=np.complex128)
        baseline = np.array([0.0, 1.0])
        sums, selected = cpu.source_search(phase, baseline, 1)
        self.assertEqual(selected[0], 0)
        self.assertEqual(abs(sums[0, 0]), abs(sums[0, 8]))
        result = cpu.refine(phase, baseline, 1, selected)
        np.testing.assert_allclose(result["K"][0], cpu.reference_topofit(phase[0], baseline, 1)[0][0])

    def test_zero_mask_is_explicitly_out_of_shared_baseline_domain(self):
        phase, baseline, wraps = cpu.fixture(2, n_ifg=5)
        phase[0, 0] = 0
        weights = cpu.trial_weights(baseline, wraps)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            cpu.source_search(phase, baseline, wraps)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            cpu.batched_search(phase, weights)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            cpu.refine(phase, baseline, wraps, np.zeros(2, dtype=int))
        reference = cpu.reference_topofit(phase[0], baseline, wraps)
        self.assertEqual(reference[3].shape, (4,))

    def test_invalid_domain_and_chunk_size(self):
        phase, baseline, wraps = cpu.fixture(2, n_ifg=5)
        with self.assertRaises(ValueError):
            cpu.trial_weights(np.ones(5), wraps)
        with self.assertRaises(ValueError):
            cpu.trial_weights(baseline, -1)
        with self.assertRaises(ValueError):
            cpu.batched_search(phase, cpu.trial_weights(baseline, wraps), 0)
        phase[0, 0] = np.nan
        with self.assertRaises(ValueError):
            cpu.source_search(phase, baseline, wraps)


if __name__ == "__main__":
    unittest.main()
