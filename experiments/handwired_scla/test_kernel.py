import unittest

import numpy as np

from benchmark import source_kernel, numpy_kernel


class SmoothingTests(unittest.TestCase):
    def test_fractional_and_fp64_values_match_source(self):
        edges = np.array([[0, 1], [0, 2], [1, 2], [2, 3]], dtype=np.uint32)
        for dtype in (np.float32, np.float64):
            k = np.array([-1000.37, 0.3, 0.30000000001, 32.9], dtype=dtype)
            c = np.array([0.0, -0.0000001, 300.11, -22.13], dtype=dtype)
            expected = source_kernel(edges, k, c)
            actual = numpy_kernel(edges, k, c)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_nan_and_isolated_vertex_match_source(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
        k = np.array([1.1, np.nan, -1.5, 4.0])
        c = np.array([2.0, -0.5, 4.0, 1.0])
        for a, b in zip(numpy_kernel(edges, k, c), source_kernel(edges, k, c)):
            np.testing.assert_array_equal(a, b)

    def test_original_does_not_quantize_unclipped_values(self):
        k = np.array([-1.0, 0.30000000001, 1.0])
        edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.uint32)
        actual, _ = numpy_kernel(edges, k, k)
        self.assertEqual(actual[1], k[1])
        self.assertNotEqual(actual[1], float(np.float32(k[1])))


if __name__ == '__main__':
    unittest.main()
