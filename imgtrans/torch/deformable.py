


# deformable transformations
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from imgtrans.img_utils.torch.methods import resize
from imgtrans.torch.utils.grid_utils import mov2pos_grid, rescale_grid
from imgtrans.torch.utils.spatial import GuassianFilter

class DeformableTransform(nn.Module):
    """
    Use grid_sample to deform the image.
    """
    
    def __init__(self, mode='bilinear', padding_mode='zeros'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        pass
    
    def forward(self, img, grid):
        """
        Args:
            img: ((*batch_dims), (*channel_dims), *spatial_size)
            grid: (*batch_dims), *spatial_size, ndim)
        """
        # reshape img to (B, C, *spatial_size)
        # reshape grid to (B, *spatial_size, ndim)
        ndim = grid.shape[-1]
        spatial_size = img.shape[-ndim:]
        batch_dims = grid.shape[:-ndim-1]
        if len(img.shape) > len(spatial_size) + len(batch_dims):
            channel_dims = img.shape[len(batch_dims):-ndim]
        else:
            channel_dims = torch.Size([1])
        
        # reshape imge to (B, C, *spatial_size)
        batch_dim, channel_dim = torch.prod(torch.tensor(batch_dims)), torch.prod(torch.tensor(channel_dims))
        img = img.reshape(batch_dim, channel_dim, *spatial_size)
        # grid_sample
        img_trans = F.grid_sample(img, grid, mode=self.mode, padding_mode=self.padding_mode)
        # reshape img
        return img_trans.reshape(*batch_dims, *channel_dims, *spatial_size)
        
        
class ElasticGrid(nn.Module):
    def __init__(
        self,
        ndim: int,
        alpha: Union[float, Sequence[float]],
        sigma: Union[float, Sequence[float]],
        kernel_size: Union[int, Sequence[int]] = 11,
        output_params: bool = False,
        mode='bilinear',
        padding_mode='zeros',
        device='cpu',
    ):
        """
        Args:
            alpha (Union[float, Sequence[float]]): magnitude of the offset
            sigma (Union[float, Sequence[float]]): sigma of the gaussian kernel
            mode (str, optional): interpolation mode. Defaults to 'bilinear'.
            padding_mode (str, optional): padding mode. Defaults to 'zeros'.
            device (str, optional): device. Defaults to 'cpu'.
        """
        super().__init__()
        self.ndim = ndim
        self.alpha = alpha
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.output_params = output_params
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        
        # assert if alpha or sigma is a sequence, then the length should be ndim
        assert isinstance(alpha, (float, int)) or len(alpha) == ndim
        assert isinstance(sigma, (float, int)) or len(sigma) == ndim
        
        self.gaussian_filter = GuassianFilter(ndim=ndim, sigma=sigma, 
                                              kernel_size=kernel_size, device=device)
        self.deformable_transform = DeformableTransform(mode=mode, padding_mode=padding_mode)
        pass
    
    def forward(self, 
                batch: int = 1, spatial_size=None, 
                mov_grid: torch.Tensor = None):
        """
        Args:
            If mov_grid is given, return the smoothed grid.
            If mov_grid is None, return the smoothed random offset grid.
        """
        if mov_grid is None:
            # create random offset grid
            mov_grid = torch.rand(batch, self.ndim, *spatial_size) * 2 - 1
        # gaussian filter, shape = (B, ndim, *spatial_size)
        mov_grid = self.gaussian_filter(mov_grid) * torch.tensor(self.alpha)[None].to(self.device)
        
        # create grid_sample's grid, range from [-1, 1]
        pos_grid = mov2pos_grid(mov_grid, val_range="percentage")
        pos_grid = rescale_grid(pos_grid, orig_range="percentage", new_range=(-1, 1))
        if not self.output_params:
            return pos_grid
        else:
            return pos_grid, {"mov_grid": mov_grid, "pos_grid": pos_grid}
        

class PerlinGrid(nn.Module):
    def __init__(self, 
                 scales: Sequence[float], 
                 spatial_dims: Sequence[int],
                 min_std: float = 0, 
                 max_std: float = 1, 
                 mode='bilinear', 
                 padding_mode='zeros', 
                 device='cpu'):
        """
        Args:
            scales (Sequence[float]): scales of the perlin noise
            spatial_dims (Sequence[int]): spatial dimensions of the perlin noise
            min_std, max_std (float): min and max std of the gaussian filter
            mode (str, optional): interpolation mode. Defaults to 'bilinear'.
            padding_mode (str, optional): padding mode. Defaults to 'zeros'.
            device (str, optional): device. Defaults to 'cpu'.
        """
        super().__init__()
        self.scales = scales
        self.spatial_dims = spatial_dims
        self.ndim = len(spatial_dims)
        self.min_std = min_std
        self.max_std = max_std
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        self.resizer = resize
        pass

    def forward(self, 
                nbatch: int = 1,
                scales: Sequence[float] = None, 
                grids: List[torch.Tensor] = None,
                ):
        """
        Returns:
            perlin_grid (torch.Tensor): (nbatch, *spatial_size, ndim)
        """
        scales = scales if scales is not None else self.scales
        if grids is None:
            grids = []
            # generate perlin noises with diff scales
            for scale in scales:
                grid_size = [int(s * scale) for s in self.spatial_dims]
                gaussian_grid = torch.randn(nbatch, self.ndim, *grid_size).to(self.device)
                grids.append(gaussian_grid)
        
        perlin_grid = torch.zeros(nbatch, self.ndim, *self.spatial_dims).to(self.device)
        # upsample grids
        for grid in grids:
            std_size_grid = self.resizer(grid, self.spatial_dims, ndim=self.ndim)
            perlin_grid += std_size_grid
        
        # normalize perlin grid
        perlin_grid = perlin_grid / len(grids)
        # (nbatch, ndim, *spatial_size) -> (nbatch, *spatial_size, ndim)
        # if self.ndim == 2:
        #     return perlin_grid.permute(0, 2, 3, 1)
        # elif self.ndim == 3:
        #     return perlin_grid.permute(0, 2, 3, 4, 1)
        
        permute_dims = [0] + [i + 2 for i in range(self.ndim)] + [1]
        return perlin_grid.permute(*permute_dims)
