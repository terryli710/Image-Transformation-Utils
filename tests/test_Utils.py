

# unittests for Utils.py

import unittest
import numpy as np
import torch


class TestUtils(unittest.TestCase):
    def test_type_check(self):
        from imgtrans.utils.type_utils import is_array_like
        self.assertTrue(is_array_like([1, 2, 3]))
        self.assertTrue(is_array_like((1, 2, 3)))
        self.assertTrue(is_array_like(np.array([1, 2, 3])))
        self.assertTrue(is_array_like(torch.tensor([1, 2, 3])))
        self.assertFalse(is_array_like(1))
        self.assertFalse(is_array_like("test"))
        pass


