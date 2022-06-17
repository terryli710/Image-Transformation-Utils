## support image transform with all kinds of matrix for transformation (DVF, warp, flow_grid, grid or affine_mtx)
## support both 2D and 3D images
## support batch transformation

import torch
import torch.nn.functional as F
import torch.nn as nn
from imgtrans.utils.grid_utils import dvf2flow_grid


# some minor util functions to convert to flow grid
def convert_dvf(dvf, img_shape):
    """
    dvf = (H, W, (D), 2 or 3)
    convert dvf to flow grid
    """
    return dvf2flow_grid(dvf)


def warp2dvf(warp, img_shape):
    """
    convert warp to dvf
    rescale warp to percentage of pixel moved
    """
    # TODO: implement batch mode to dvf2flow_grid and test them
    # TODO: implement these functions and implement SpatialTransformer
    # TODO: test the spatial transformer
    ...

    


def convert_warp(warp, img_shape):
    ...


class SpatialTransofrmer(nn.Module):
    """
    perform spatial transformation for images, 2D or 3D, in a batched mode
    """

    def __init__(self, matrix_type="warp", **kwargs):
        """
        matrix_type: "warp", "grid", "affine_mtx", "dvf", "flow_grid"
        kwargs: for grid_sample: mode, padding_mode, align_corners
        """
        super().__init__()
        self.matrix_type = matrix_type
        self.convert_func = CONVERT_DICT[matrix_type]
        self.kwargs = kwargs
        pass

    def forward(self, img: torch.Tensor, matrix: torch.Tensor, **kwargs):
        """
        img = (C or B, H, W, (D))
        matrix = (B, H, W, (D), ndim)
        """
        # some assertions
        ...
        flow_grid = self.convert_func(matrix, img.shape[2:])
        deformed_img = F.grid_sample(input=img[:, None, ...], # NOTE: adding Channels = 1 (B, C=1, H, W, (D))
                                    grid=flow_grid, # NOTE: (B, H, W, (D), ndim)
                                    **self.kwargs)
        return deformed_img[:, 0, ...], {"flow_grid": flow_grid}