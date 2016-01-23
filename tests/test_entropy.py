import os
import numpy as np
from unittest import TestCase
from numpy.testing import assert_almost_equal

import entropy

TEST_DATA_FILE_1D = os.path.join(os.path.dirname(__file__), 'data_1d.txt')


class TestEntropy(TestCase):

    def setUp(self):
        self.data_1d = np.loadtxt(TEST_DATA_FILE_1D)

    def test_sampen_1d(self):
        e = entropy.sampen(self.data_1d, 2, 0.1)
        assert_almost_equal(e, 0.70589052)

    def test_fuzzyen_1d(self):
        e = entropy.fuzzyen(self.data_1d, 2, 0.1, 2)
        assert_almost_equal(e, 0.20953354)
