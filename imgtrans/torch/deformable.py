


# deformable transformations
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from imgtrans.img_utils.torch.methods import resize
from imgtrans.img_utils.torch.type_utils import process_shape
from imgtrans.img_utils.torch.utils import move_dim
from imgtrans.torch.utils.grid_utils import mov2pos_grid, rescale_grid
from imgtrans.torch.utils.spatial import GuassianFilter
from abc import ABC, abstractmethod

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


class GridGenerator(nn.Module, ABC):
    """
    Abstract class to generate grid for deformable transformations
    """
    def __init__(self, 
                 ndim: int = None,
                 spatial_size: Optional[Union[Sequence[int], torch.Size]] = None,
                 mode='bilinear', 
                 padding_mode='zeros', 
                 device='cpu',
                 return_params: bool = False):
        """
        Args:
            ndim (int, optional): number of spatial dimensions. Defaults to None.
            spatial_size (Optional[Union[Sequence[int], torch.Size]], optional): spatial size. Defaults to None.
            mode (str, optional): mode for grid_sample. Defaults to 'bilinear'.
            padding_mode (str, optional): padding_mode for grid_sample. Defaults to 'zeros'.
            device (str, optional): device. Defaults to 'cpu'.
            return_params (bool, optional): whether to return params. Defaults to False.
        """
        super().__init__()
        self.spatial_size = process_shape(spatial_size)
        self.ndim = ndim or len(spatial_size)
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        self.return_params = return_params
        pass
    

    def _check_grid_params(self, nbatch = 1, spatial_size = None, grid: torch.Tensor = None):
        spatial_size = process_shape(spatial_size) or self.spatial_size
        # check ndim
        ndim = self.ndim or len(spatial_size)
        # check shape of grid
        if grid is not None:
            assert grid.shape[-1] == ndim, f"grid should have shape (B, *spatial_size, ndim), where ndim = {ndim}, but got {grid.shape}"
            assert grid.shape[0] == nbatch, f"grid should have shape (B, *spatial_size, ndim), where B = {nbatch}, but got {grid.shape}"
        # return
        return nbatch, spatial_size, ndim
    
    def _convert_mov_to_pos_grid(self, mov_grid, val_range="percentage"):
        # create grid_sample's grid, range from [-1, 1]
        pos_grid = mov2pos_grid(mov_grid, val_range=val_range)
        pos_grid = rescale_grid(pos_grid, orig_range=val_range, new_range=(-1, 1))
        return pos_grid
    
    def _output(self, grid: torch.Tensor, params: dict = {}, return_params=None):
        return_params = return_params if return_params is not None else self.return_params
        if self.return_params:
            return grid, params
        else:
            return grid
    
    def _generate_raw_grid(self, nbatch: int, spatial_size, ndim, grid: torch.Tensor=None):
        # example implementation to generate zero mov_grid
        mov_grid = torch.zeros(nbatch, *spatial_size, ndim, device=self.device)
        return mov_grid
    
    def forward(self, 
                nbatch: int = 1, 
                spatial_size = None, 
                specified_grid: torch.Tensor = None,
                return_params: bool = None):
        """
        Args:
            nbatch (int, optional): number of batch. Defaults to 1.
            spatial_size (Optional[Sequence[int]], optional): spatial size. Defaults to None.
            specified_grid (torch.Tensor, optional): given grid. sizes = (B, *spatial_size, ndim). 
            NOTE: specified grid's relationship with the return grid is not defined here, it varies from different subclasses.
        Returns:
            pos_grid: (B, *spatial_size, ndim) range = [-1, 1], can directly be used in grid_sample
            params (optional): dict, parameters of the grid
        """
        # this abstract method should be implemented in the subclass
        nbatch, spatial_size, ndim = self._check_grid_params(nbatch, spatial_size, specified_grid)
        mov_grid = specified_grid or self._generate_raw_grid(nbatch, spatial_size, ndim, specified_grid)
        pos_grid = self._convert_mov_to_pos_grid(mov_grid)
        return self._output(pos_grid, return_params=return_params)

        
class ElasticGrid(GridGenerator):
    def __init__(
        self,
        ndim: int,
        alpha: Union[float, Sequence[float]],
        sigma: Union[float, Sequence[float]],
        kernel_size: Union[int, Sequence[int]] = 11,
        return_params: bool = False,
        spatial_size: Optional[Sequence[int]] = None,
        mode='bilinear',
        padding_mode='zeros',
        device='cpu',
    ):
        """
        Args:
            alpha (Union[float, Sequence[float]]): magnitude of the offset, larger alpha means more deformation
            sigma (Union[float, Sequence[float]]): sigma of the gaussian kernel, larger sigma means more smooth
            mode (str, optional): interpolation mode. Defaults to 'bilinear'.
            padding_mode (str, optional): padding mode. Defaults to 'zeros'.
            device (str, optional): device. Defaults to 'cpu'.
        """
        super(ElasticGrid, self).__init__(
            ndim=ndim, 
            spatial_size=spatial_size, 
            mode=mode, 
            padding_mode=padding_mode, 
            device=device, 
            return_params=return_params)
        self.alpha = alpha
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.return_params = return_params
        
        # assert if alpha or sigma is a sequence, then the length should be ndim
        assert isinstance(alpha, (float, int)) or len(alpha) == ndim
        assert isinstance(sigma, (float, int)) or len(sigma) == ndim
        
        self.gaussian_filter = GuassianFilter(ndim=ndim, nchannel=ndim, sigma=sigma, 
                                              kernel_size=kernel_size, device=device)
        self.deformable_transform = DeformableTransform(mode=mode, padding_mode=padding_mode)
        pass
    
    def _generate_raw_grid(self, nbatch, spatial_size, ndim):
        # create random offset grid
        offset_grid = torch.rand(nbatch, *spatial_size, ndim, device=self.device) * 2 - 1
        return offset_grid
    
    def forward(self, 
                nbatch: int = 1, 
                spatial_size = None, 
                specified_grid: torch.Tensor = None,
                return_params: bool = None):
        """
        Args:
            specified_grid (torch.Tensor, optional): given grid. sizes = (B, *spatial_size, ndim).
            If specified_grid is given, return the smoothed grid.
            If specified_grid is not given, generate random offset grid and return the smoothed grid.
        Returns:
            pos_grid: (*batch_dims, *spatial_size, ndim)
        """
        # sanity check
        nbatch, spatial_size, ndim = self._check_grid_params(nbatch, spatial_size, specified_grid)
        
        # randomly generate offset grid, if grid is not given
        if specified_grid is None:
            # create random offset grid
            grid = self._generate_raw_grid(nbatch, spatial_size, ndim)
        else:
            grid = specified_grid
        
        # gaussian filter, shape = (B, ndim, *spatial_size)
        grid = move_dim(self.gaussian_filter(move_dim(grid, -1, 1)), 1, -1)
        grid = grid * torch.tensor(self.alpha)[None].to(self.device)
        
        pos_grid = self._convert_mov_to_pos_grid(grid)
        params = {"pos_grid": grid, "mov_grid": grid}
        return self._output(pos_grid, params=params)
        

class PerlinGrid(GridGenerator):
    def __init__(self, 
                 scales: Sequence[float], 
                 spatial_size: Sequence[int] = None,
                 ndim: int = 2,
                 min_std: float = 0, 
                 max_std: float = 1, 
                 mode='bilinear', 
                 padding_mode='zeros', 
                 device='cpu',
                 return_params: bool = False):
        """
        Args:
            scales (Sequence[float]): scales of the perlin noise
            spatial_size (Sequence[int]): spatial sizes of the perlin noise
            min_std, max_std (float): min and max std of the gaussian filter
            mode (str, optional): interpolation mode. Defaults to 'bilinear'.
            padding_mode (str, optional): padding mode. Defaults to 'zeros'.
            device (str, optional): device. Defaults to 'cpu'.
        """
        super(PerlinGrid, self).__init__(
            ndim=ndim, 
            spatial_size=spatial_size, 
            mode=mode, 
            padding_mode=padding_mode, 
            device=device, 
            return_params=return_params)
        self.scales = scales
        self.min_std = min_std
        self.max_std = max_std
        self.resizer = resize
        pass
    
    def _check_grid_params(self, nbatch, spatial_size, specified_grid, scales):
        """
        specified_grid = list of torch.Tensor, each torch.Tensor has shape = (B, *spatial_size, ndim)
        """
        nbatch, spatial_size, ndim = super()._check_grid_params(nbatch, spatial_size)
        scales = scales if scales is not None else self.scales
        # check if specified_grid is given
        if specified_grid is not None:
            assert isinstance(specified_grid, list), "specified_grid should be a list of torch.Tensor"
            assert len(specified_grid) == len(self.scales), "number of specified_grid should be equal to number of scales"
            for grid in specified_grid:
                assert grid.shape[0] == nbatch, "number of batches should be the same"
                assert grid.shape[1:] == (ndim, *spatial_size), "spatial_size and ndim should be the same"
        return nbatch, spatial_size, ndim, scales
    
    def _generate_raw_grid(self, nbatch, spatial_size, scales):
        """
        generate perlin noise grid with different scales
        """
        # generate perlin noise grid with different scales
        grid = []
        # generate perlin noises with diff scales
        for scale in scales:
            grid_size = [int(s / scale) for s in spatial_size]
            gaussian_grid = torch.randn(nbatch, *grid_size, self.ndim).to(self.device)
            grid.append(gaussian_grid)
        return grid
    
    
    def forward(self, 
                nbatch: int = 1,
                spatial_size: Sequence[int] = None,
                scales: Sequence[float] = None, 
                specified_grid: List[torch.Tensor] = None,
                ):
        """
        Args:
            nbatch (int, optional): number of batches. Defaults to 1.
            spatial_size (Sequence[int], optional): spatial sizes of the perlin noise. Defaults to None.
            scales (Sequence[float], optional): scales of the perlin noise. Defaults to None.
            specified_grid (List[torch.Tensor], optional): specified_grid with different scales. If not given, generate grid.
        Returns:
            perlin_grid (torch.Tensor): (nbatch, *spatial_size, ndim)
        """
        # sanity check
        nbatch, spatial_size, ndim, scales = self._check_grid_params(nbatch, spatial_size, specified_grid, scales)
        
        if specified_grid is None:
            grid_list = self._generate_raw_grid(nbatch, spatial_size, scales)
        else:
            grid_list = specified_grid
        
        perlin_grid = torch.zeros(nbatch, *spatial_size, self.ndim).to(self.device)
        # upsample grid
        for grid in grid_list:
            # grid = (B, *spatial_size, ndim) -> (ndim, B, *spatial_size) to resize -> (B, *spatial_size, ndim)
            std_size_grid = self.resizer(move_dim(grid, -1, 1), spatial_size, ndim=self.ndim) # (B, *spatial_size, ndim) -> (B, ndim, *spatial_size)
            # (ndim, B, *spatial_size) ->  (B, *spatial_size, ndim)
            std_size_grid = move_dim(std_size_grid, 1, -1)
            perlin_grid += std_size_grid
        
        # normalize perlin grid
        perlin_grid = perlin_grid / len(grid)
        pos_grid = self._convert_mov_to_pos_grid(perlin_grid)
        params = {"mov_grid": perlin_grid}
        
        return self._output(pos_grid, params=params)
