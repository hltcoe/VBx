#!/usr/bin/env python

import unittest
import numpy as np
from VBx.vbhmm import remap_label_numbers

class TestLabelRemap(unittest.TestCase):
    def test1(self):
        t1 = [2, 3, 4, 1]
        expected = [2, 3, 4, 1]
        actual = remap_label_numbers(t1)
        self.assertTrue(np.array_equal(expected, actual))

        t2 = [3, 2, 2, 3, 4]
        expected = [2, 1, 1, 2, 3]
        actual = remap_label_numbers(t2)
        self.assertTrue(np.array_equal(expected, actual))


if __name__ == '__main__':
    unittest.main()