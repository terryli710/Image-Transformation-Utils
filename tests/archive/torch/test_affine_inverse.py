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

    def test_InverseAffine(self, ndarrays_regression):
        import imgtrans as imt
        exp_img = self.gen_grid_img(length=100, cells_per_row=10)
        exp_img = torch.from_numpy(exp_img)
        affine = imt.affine.Affine(padding_mode="reflection")
        img_trans, params = affine(exp_img[None, ...],
                             rotate=0.1,
                             shear=(0.1, 0.1),
                             translate=(0.1, 0.1),
                             scale=(0.9, 1.1),)
        inv_img, inv_params = imt.affine.inverse_affine(img_trans, params['aff_mtx'])
        ndarrays_regression.check({"ret": inv_img, "aff_mtx": inv_params['aff_mtx']})
        save_path = osp.join(self.DIRPATH, "inverse_affine_np.jpeg")
        fig, ax = plt.subplots(1, 3, figsize=(5, 10))
        ax[0].imshow(exp_img)
        ax[1].imshow(img_trans[0])
        ax[2].imshow(inv_img[0])
        plt.savefig(save_path)
        print(f"saved to {save_path}")
        pass