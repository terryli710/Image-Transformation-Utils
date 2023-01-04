

# unittests for imgtrans.torch.deformable.py
import os
import os.path as osp
import unittest
from typing import List

import matplotlib.pyplot as plt
import torch

from imgtrans.torch.deformable import DeformableTransform, ElasticGrid, PerlinGrid



class TestDeformableTransform(unittest.TestCase):
    
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
        
    
    def test_elastic_transform(self):
        img = self.generate_grid_img()
        elastic_grid = ElasticGrid(ndim=2, alpha=5, sigma=5 ,return_params=True)
        transformer = DeformableTransform()
        grid, params = elastic_grid(spatial_size=(100, 100))
        trans_img = transformer(img, grid)[0, 0]
        # plot the images
        self.plot_imgs([img, trans_img, params['mov_grid'][0, ..., 0]], 
                       "elastic_transform.jpeg", 
                       ["original", "transformed", "mov_grid"])
        pass
        
    def test_perlin_transform(self):
        img = self.generate_grid_img()
        perlin_grid = PerlinGrid(ndim=2, scales=[5, 10], return_params=True)
        transformer = DeformableTransform()
        grid, params = perlin_grid(spatial_size=(100, 100))
        trans_img = transformer(img, grid)[0, 0]
        # plot the images
        self.plot_imgs([img, trans_img, params['mov_grid'][0, ..., 0]], 
                       "perlin_transform.jpeg", 
                       ["original", "transformed", "mov_grid"])
        pass