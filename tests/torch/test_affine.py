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

    def test_Affine(self, ndarrays_regression):
        import imgtrans as imt
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        exp_img = torch.from_numpy(exp_img)
        affine = imt.affine.Affine(padding_mode="reflection")
        ret, params = affine(exp_img[None, ...],
                             rotate=0.1,
                             shear=(0.1, 0.1),
                             translate=(0.1, 0.1),
                             scale=(0.9, 1.1),)
        ndarrays_regression.check({"ret": ret, "aff_mtx": params['aff_mtx']})
        save_path = osp.join(self.DIRPATH, "affine_np.jpeg")
        plt.imsave(save_path, arr=ret[0,...], cmap="gray")
        print(f"saved to {save_path}")
        pass

    def test_RandAffine(self, ndarrays_regression):
        import imgtrans as imt
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        exp_img = torch.from_numpy(exp_img)
        rand_affine = imt.affine.RandAffine(rotate_range=(0.1, 0.2),
                                            shear_range=(0.1, 0.2),
                                            translate_range=(0.1, 0.1),
                                            scale_range=(0.9, 1.1),
                                            padding_mode="reflection")
        ret, params = rand_affine(exp_img[None, ...], seed=42) # input must be [num_channels, H, W (,D)]
        ndarrays_regression.check({
            "ret": ret,
            "aff_mtx": params['aff_mtx'],
            "params": np.array(params['params']['shear'])
        })
        save_path = osp.join(self.DIRPATH, "random_affine_np.jpeg")
        plt.imsave(osp.join(self.DIRPATH, "random_affine_np.jpeg"),
                   arr=ret[0,...],
                   cmap="gray")
        print(f"saved to {save_path}")
        pass

