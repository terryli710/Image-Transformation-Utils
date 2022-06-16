import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
os.environ["IMG_TRANS_BACKEND"] = "pytorch"



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

    def test_RandElastic(self, ndarrays_regression):
        import imgtrans as imt
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        exp_img = torch.from_numpy(exp_img).type(torch.float)
        elastic = imt.perlin.RandPerlin(scales=(32, 64), max_std=5)
        ret, params = elastic(exp_img[None, ...], seed=42)
        ndarrays_regression.check({"ret": ret, "flow_grid": params['flow_grid']})
        save_path = osp.join(self.DIRPATH, "random_perlin.jpeg")
        plt.imsave(save_path, arr=ret[0,...], cmap="gray")
        print(f"saved to {save_path}")
        pass