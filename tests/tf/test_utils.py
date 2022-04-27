import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ["IMG_TRANS_BACKEND"] = "tensorflow"


class Test_volshape_to_meshgrid:

    DIRPATH = os.path.dirname(os.path.realpath(__file__))

    def test_volshape_to_meshgrid(self, ndarrays_regression):
        import imgtrans as imt
        volshape = (10, 10, 10)
        meshgrid = imt.tf.utils.spatial.volshape_to_meshgrid(volshape)
        ndarrays_regression.check({"meshgrid": meshgrid})
        pass