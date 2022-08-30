import unittest
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ["IMG_TRANS_BACKEND"] = "tensorflow"

class TestGridUtils(unittest.TestCase):

    DIRPATH = os.path.dirname(os.path.realpath(__file__))

    def test_flow_grid_to_dvf(self):
        import imgtrans as imt

        # test if flow_grid is correctly generated from dvf
        dvf = tf.zeros((5, 5, 2))
        flow_grid = imt.tf.utils.grid_utils.Warp2Flow_grid(dvf)
        synth_flow_grid = tf.stack(tf.meshgrid(tf.linspace(-1, 1, 5), 
                                      tf.linspace(-1, 1, 5)), axis=-1)

        self.assertTrue(tf.experimental.numpy.allclose(flow_grid, synth_flow_grid))
        pass


    def test_dvf_to_flow_grid(self):
        import imgtrans as imt

        # test if dvf is correctly generated from flow_grid
        flow_grid = tf.stack(tf.meshgrid(tf.linspace(-1, 1, 5), 
                                               tf.linspace(-1, 1, 5)), axis=-1)
        dvf = imt.tf.utils.grid_utils.flow_grid2dvf(flow_grid)
        synth_dvf = tf.zeros((5, 5, 2))

        self.assertTrue(tf.experimental.numpy.allclose(dvf, synth_dvf))
        pass


if __name__ == '__main__':
    unittest.main()
    # test = TestGridUtils()
    # test.test_dvf_to_flow_grid()
    pass