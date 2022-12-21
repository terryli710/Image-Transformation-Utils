

# unittests for imgtrans.torch.utils.grid_utils.py
import os
import os.path as osp
import unittest
from typing import List

import matplotlib.pyplot as plt
import torch

from imgtrans.torch.utils.randomize import RandomFromIntervalTorch
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
    
    def test_Interval(self):
        interval = Interval(0, 1)
        self.assertEqual(str(interval), "Interval(lower=0, upper=1)")
        self.assertTrue(0 in interval)
        self.assertFalse(2 in interval)
        pass
    
    def test_RandomFromIntervalTorch(self):
        """
        Test that given some intervals, the randomly sampled number 
        are correctly lying within the range
        """
        intervals = [Interval(0, 1), Interval(0, 2), Interval(0, 3)]
        rand_sampler = RandomFromIntervalTorch(include_negative=False)
        samples = rand_sampler.sample(intervals, nbatch=100) # (100, 3)
        # samples should be within the ranges, with shape (100, 3)
        self.assertTrue(torch.all(samples >= 0))
        self.assertTrue(torch.all(samples[:, 0] <= 1))
        self.assertTrue(torch.all(samples[:, 1] <= 2))
        self.assertTrue(torch.all(samples[:, 2] <= 3))
        self.assertTrue(samples.shape == torch.Size([100, 3]))
        
        # include nagetive
        rand_sampler = RandomFromIntervalTorch(include_negative=True)
        samples = rand_sampler.sample(intervals, nbatch=100) # (100, 3)
        # samples should be within the ranges (-1, 1), ..., with shape (100, 3)
        self.assertTrue(torch.all(torch.abs(samples[:, 0]) <= 1))
        self.assertTrue(torch.all(torch.abs(samples[:, 1]) <= 2))
        self.assertTrue(torch.all(torch.abs(samples[:, 2]) <= 3))
        self.assertTrue(samples.shape == torch.Size([100, 3]))
        pass
    
        
        
        
        
        
