import os
import numpy as np
from unittest import TestCase
from numpy.testing import assert_equal, assert_almost_equal

import entropy

TEST_DATA_FILE_1D = os.path.join(os.path.dirname(__file__), 'data_1d.txt')


class TestEntropy(TestCase):

    def setUp(self):
        self.data_1d = np.loadtxt(TEST_DATA_FILE_1D)

    def test_sampen_1d(self):
        e = entropy.sampen(self.data_1d, 2, 0.1)
        assert_almost_equal(e, 0.70589052)

    def test_sampen_naive(self):
        x = np.array([0, 0.1, 0.2, 0.3, 1])
        truth = np.log(3/1)
        en = entropy.sampen_naive(x, 2, 0.5)
        assert_almost_equal(en, truth)

        en = entropy.sampen(x, 2, 0.5, scale=False)
        assert_almost_equal(en, truth)

    def test_fuzzyen_1d(self):
        e = entropy.fuzzyen(self.data_1d, 2, 0.1, 2)
        assert_almost_equal(e, 0.20953354)

    def test_pattern_mat(self):
        x = list(range(8))

        targ3 = [[0, 1, 2, 3, 4, 5],
                 [1, 2, 3, 4, 5, 6],
                 [2, 3, 4, 5, 6, 7]]

        targ4 = [[0, 1, 2, 3, 4],
                 [1, 2, 3, 4, 5],
                 [2, 3, 4, 5, 6],
                 [3, 4, 5, 6, 7]]

        assert_equal(entropy.pattern_mat(x, 3), targ3)
        assert_equal(entropy.pattern_mat(x, 4), targ4)

    def test_copies(self):
        x = np.random.uniform(low=0, high=1, size=(2, 100))
        x0 = np.copy(x[0])
        x1 = np.copy(x[1])

        entropy.sampen(x0, 2, 0.1)
        assert_equal(x0, x[0])

        entropy.fuzzyen(x0, 2, 0.1, 2)
        assert_equal(x0, x[0])

        entropy.cross_sampen(x0, x1, 2, 0.1)
        assert_equal(x0, x[0])
        assert_equal(x1, x[1])

        entropy.cross_fuzzyen(x0, x1, 2, 0.1, 2)
        assert_equal(x0, x[0])
        assert_equal(x1, x[1])
