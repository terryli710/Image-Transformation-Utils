import imgtrans as imt
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


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
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        affine = imt.affine.Affine()
        ret, params = affine(exp_img,
                             rotate_params=0.1,
                             shear_params=(0.1, 0.1),
                             translate_params=(0.1, 0.1),
                             scale_params=(0.9, 1.1),
                             padding_mode="reflect")
        ndarrays_regression.check({"ret": ret, "aff_mtx": params['aff_mtx']})
        save_path = osp.join(self.DIRPATH, "affine_np.jpeg")
        plt.imsave(save_path, arr=ret, cmap="gray")
        print(f"saved to {save_path}")
        pass

    def test_RandAffine(self, ndarrays_regression):
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        rand_affine = imt.affine.RandAffine(rotate_range=(0.1, 0.2),
                                            shear_range=(0.1, 0.2),
                                            translate_range=(0.1, 0.1),
                                            scale_range=(0.9, 1.1),
                                            padding_mode="reflect")
        ret, params = rand_affine(exp_img, seed=42)
        ndarrays_regression.check({
            "ret": ret,
            "aff_mtx": params['aff_mtx'],
            "params": np.array(params['params']['shear'])
        })

        plt.imsave(osp.join(self.DIRPATH, "random_affine_np.jpeg"),
                   arr=ret,
                   cmap="gray")
        pass
