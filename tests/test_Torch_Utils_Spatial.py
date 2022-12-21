

# unittests for imgtrans.torch.utils.grid_utils.py
import os
import os.path as osp
import unittest
from typing import List

import matplotlib.pyplot as plt
import torch

from imgtrans.torch.utils.randomize import RandomFromIntervalTorch
from imgtrans.torch.utils.spatial import GuassianFilter
from imgtrans.utils.randomize import Interval

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
    
    def test_GaussianFilter(self):
        """
        Test the conv function gaussian filter with specified kernel
        maybe just visually testing
        """
        test_img = self.generate_grid_img()
        gaussian_filter = GuassianFilter(ndim=2, kernel_size=3, sigma=1)
        blurred_img = gaussian_filter(test_img)
        
        gaussian_filter = GuassianFilter(ndim=2, kernel_size=5, sigma=2)
        more_blurred_img = gaussian_filter(test_img)
        
        self.plot_imgs([test_img, blurred_img, more_blurred_img],
                       "gaussian_filter.jpeg",
                        titles=["original", "blurred", "more blurred"])
        pass
        