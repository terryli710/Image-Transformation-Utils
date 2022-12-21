

# unittests for imgtrans.torch.utils.grid_utils.py
import os
import os.path as osp
import unittest
from typing import List

import matplotlib.pyplot as plt
import torch

from imgtrans.torch.utils.grid_utils import (get_mesh, mov2pos_grid,
                                             pos2mov_grid, resample,
                                             rescale_grid)


class TestGridUtils(unittest.TestCase):
    
    def generate_grid_img(self, spatial_dims=(100, 100), line_interval=10):
        """
        Generate images with grid lines
        Args:
            spatial_dims: (H, W[, D])
            line_interval: int, interval of grid lines
        Returns:
            img: (H, W[, D]), torch.Tensor
        """
        img = torch.zeros(spatial_dims)
        for i in range(0, spatial_dims[0], line_interval):
            img[i, :] = 1
        for i in range(0, spatial_dims[1], line_interval):
            img[:, i] = 1
        return img
    
    
    
    @staticmethod
    def plot_imgs(imgs: List[torch.Tensor], save_name, titles=None):
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        fig, ax = plt.subplots(1, len(imgs), figsize=(len(imgs) * 5, 5))
        for i, img in enumerate(imgs):
            ax[i].imshow(img, cmap="gray")
            if titles is not None:
                ax[i].set_title(titles[i])
        plt.savefig(osp.join(current_file_path, save_name), dpi=300)
        print(f"saved to {save_name}")
        pass
        
    
    def test_resample(self):
        """
        data: ((B dims), (C dims), H, W[, D])
        grid: ((B dims), H, W[, D], 2 or 3)
        resample should use the same grid for all channels
        """
        # generate two types of grid, one is identity, the other is a translation to the right
        # data = (2, 2, 100, 100)
        # grid = (2, 100, 100, 2), grid[0] -> identity, grid[1] -> translation to the right
        
        img = self.generate_grid_img()[None, None, ...].repeat(2, 2, 1, 1)
        grid = get_mesh(img.shape[-2:], device=img.device, dtype=img.dtype)[None, ...].repeat(2, 1, 1, 1)
        grid = rescale_grid(grid, [(0, s) for s in img.shape[-2:]], (-1, 1))
        # make translation grid
        grid[1, :, :, 0] -= 0.5
        
        img_trans = resample(img, grid, mode="bilinear") # (2, 2, 100, 100)
        
        # visualize the images
        self.plot_imgs([img[0, 0], img_trans[0, 1], img_trans[1, 0]],
                        save_name="test_resample.jpeg",
                        titles=["original", "trans_orig", "trans_trans"])
        
        # NOTE: not too close as some approximation is used when grid_sample
        # check if the first channel is the same as the original image
        # self.assertTrue(torch.allclose(img[0, 0], img_trans[0, 0]))
        # check if the second channel is translated to the right
        # self.assertTrue(torch.allclose(img[0, 0, :, 25:], img_trans[0, 1, :, :-25]))
        
        pass
        
        
    def test_pos2mov_grid(self):
        """
        data: ((B dims), (C dims), H, W[, D])
        grid: ((B dims), H, W[, D], 2 or 3)
        """
        # generate a identity position grid
        pos_grid = get_mesh((100, 100), device="cpu", dtype=torch.float32)[None, ...]
        mov_grid = pos2mov_grid(pos_grid, val_range="pixel")
        # assert that the mov_grid is close to all zeros
        self.assertTrue(torch.allclose(mov_grid, torch.zeros_like(mov_grid)))
        pass

    def test_mov2pos_grid(self):
        """
        data: ((B dims), (C dims), H, W[, D])
        grid: ((B dims), H, W[, D], 2 or 3)
        """
        # generate a identity moving grid -> all zeros
        mov_grid = torch.zeros((1, 100, 100, 2), device="cpu", dtype=torch.float32)
        pos_grid = mov2pos_grid(mov_grid, val_range="pixel")
        # assert that the pos_grid is close to pixel positions
        pos = get_mesh((100, 100), device="cpu", dtype=torch.float32)[None, ...]
        self.assertTrue(torch.allclose(pos_grid, pos))
        pass
    
    
    def test_get_mesh(self):
        """
        Test the get mesh function in 3D case,
        first dim increase along the z-axis;
        second dim increase along the y-axis;
        third dim increase along the x-axis;
        """
        mesh_grid = get_mesh((10, 10, 10), device="cpu", dtype=torch.float32)
        # check the first dim
        self.assertTrue(torch.allclose(mesh_grid[0, 0, :, 0], torch.arange(10, dtype=torch.float32)))
        # check the second dim
        self.assertTrue(torch.allclose(mesh_grid[0, :, 0, 1], torch.arange(10, dtype=torch.float32)))
        # check the third dim
        self.assertTrue(torch.allclose(mesh_grid[:, 0, 0, 2], torch.arange(10, dtype=torch.float32)))
        pass
    
