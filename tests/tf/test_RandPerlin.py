import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ["IMG_TRANS_BACKEND"] = "tensorflow"


class TestTransforms:
    
    DIRPATH = os.path.dirname(os.path.realpath(__file__))

    def gen_grid_img(self, length, cells_per_row):
        '''Generates a binary grid in a numpy array. cells_per_row must be a power of 2'''
        aux = np.full(length, False)
        n = 2
        aux[:length // n] = True
        while n < cells_per_row:
            n *= 2
            aux = aux != np.roll(aux, length // n)

        a, b = np.meshgrid(aux, np.roll(aux, length // n), indexing="ij")
        return (a != b).astype(np.bool_).astype(np.int32)

    def test_RandPerlin(self, ndarrays_regression):
        import imgtrans as imt
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        exp_img = tf.convert_to_tensor(exp_img)
        perlin = imt.perlin.RandPerlin(max_std=10)
        ret, params = perlin(exp_img[None, ...], seed=42, dtype=tf.float32)
        print("ret.shape", ret.shape)
        ndarrays_regression.check({"ret": ret, "dvf": params['dvf']})
        save_path = osp.join(self.DIRPATH, "random_perlin.jpeg")
        plt.imsave(save_path, arr=ret[0,...], cmap="gray")
        print(f"saved to {save_path}")
        pass

#     def test_RandPerlin_debug(self):
#         import imgtrans as imt
#         exp_img = self.gen_grid_img(length=100, cells_per_row=10)
#         exp_img = tf.convert_to_tensor(exp_img)
#         perlin = imt.perlin.RandPerlin(max_std=10)
#         ret, params = perlin(exp_img[None, ...], seed=42, dtype=tf.float32)
#         print("ret.shape", ret.shape)
#         # ndarrays_regression.check({"ret": ret, "flow_grid": params['flow_grid']})
#         save_path = osp.join(self.DIRPATH, "random_perlin.jpeg")
#         plt.imsave(save_path, arr=ret[0,...], cmap="gray")
#         print(f"saved to {save_path}")
#         pass

# if __name__ == "__main__":
#     test = TestTransforms()
#     test.test_RandPerlin_debug()
#     pass