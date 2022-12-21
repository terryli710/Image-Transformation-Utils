# test how to generate a identity grid for grid_smaple
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DebugGridSample:

    def load_labelmap_torch(self):
        label_map = np.load("/home/yiheng/processed_data/lspine_labelmap/imgs_320-320-16/406.npy", allow_pickle=True)
        label_map = torch.from_numpy(label_map[None, ...].astype("float32")) # for perlin, the image is (B or C, H, W, D)
        return label_map

    def test_grid_sample_2d(self):
        label_map = self.load_labelmap_torch() # (B or C, H, W, D)
        slice_idx = 8
        # genrate an identity grid
        image_shape = label_map.shape[1:-1] # 2d
        ls = [torch.linspace(-1, 1, x) for x in image_shape]
        # indexing = "xy"
        grid = torch.stack(torch.meshgrid(*ls, indexing="xy"), dim=-1)
        # grid = torch.flip(grid, [-1])
        
        # grid_sample
        img_trans = F.grid_sample(label_map[:, None, ..., slice_idx], grid[None, ...])[0,...]
        
        # visualize the image
        plt.imsave("/home/yiheng/temp/test_grid_sample_2d.jpeg", img_trans[0,...])
        pass

    def test_grid_sample_3d(self):
        label_map = self.load_labelmap_torch() # (B or C, H, W, D)
        # genrate an identity grid
        image_shape = label_map.shape[1:] # 3d
        ls = [torch.linspace(-1, 1, x) for x in image_shape]
        grid = torch.stack(torch.meshgrid(*ls, indexing="ij"), dim=-1)
        grid = torch.flip(grid, [-1])
        
        # grid_sample
        img_trans = F.grid_sample(label_map[:, None, ...], grid[None, ...])[0,...]
        
        # visualize the image
        plt.imsave("/home/yiheng/temp/test_grid_sample_3d.jpeg", img_trans[0,:,:,0])
        pass


if __name__ == "__main__":
    a = DebugGridSample()
    a.test_grid_sample_2d()
    a.test_grid_sample_3d()
    pass

