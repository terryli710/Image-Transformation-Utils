""" perlin transform, inspired by neurite package """
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.grid_utils import DVF2Flow
from .utils.spatial import DrawPerlin


class RandPerlin(nn.Module):
    """
    random perlin transformations (use perlin noise as deformation field)
    """

    def __init__(self, scales=(32, 64), max_std=1, min_std=0, requires_grad=True):
        super().__init__()
        self.scales = scales
        self.min_std = min_std
        self.max_std = max_std
        self.requires_grad = requires_grad
        pass

    def forward(
        self,
        img: torch.Tensor,
        out_shape=None,
        mode="bilinear", # for small movement should not be nearest
        padding_mode="reflection",
        dtype=None,
        seed=None,
    ):
        """ 
        img = (C or B, H, W, (D))
        
        """
        if not out_shape:
            out_shape = img.shape[1:]
        if not dtype:
            dtype = img.dtype

        ndim = len(out_shape)

        perlin_dvf = DrawPerlin(requires_grad=self.requires_grad)(out_shape=(*out_shape, ndim),
                                  scales=self.scales,
                                  min_std=self.min_std,
                                  max_std=self.max_std,
                                  dtype=dtype,
                                  device=img.device,
                                  seed=seed)

        # convert range of warp from percentage of pixel moved (xvm.utils.transform)
        # to location of image from -1 to 1 (torch.nn.functional.grid_sample)
        # NOTE: perlin_dvf is a DVF that denotes the percentage of displacement
        # while flow_grid is a flow field that denotes the location of image
        # add batch dim to dvf
        flow_grid = DVF2Flow(requires_grad=self.requires_grad)(perlin_dvf[None, ...], out_shape) # dvf shape: (1, H, W, (D), 2 or 3), out_shape: (H, W, (D))
        # add batch dim to flow_grid
        flow_grid = flow_grid.repeat(img.shape[0], *[1] * (ndim + 1)).type_as(img)

        deformed_img = F.grid_sample(input=img[:, None, ...], # NOTE: adding Channels = 1 (B, C=1, H, W, (D))
                                     grid=flow_grid, # NOTE: (B, H, W, (D), ndim)
                                     mode=mode,
                                     padding_mode=padding_mode,
                                     align_corners=True)
        
        return deformed_img[:, 0, ...], {"dvf": perlin_dvf, "flow_grid": flow_grid}

