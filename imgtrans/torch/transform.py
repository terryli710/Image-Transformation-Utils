## support image transform with all kinds of matrix for transformation (DVF, warp, flow_grid, grid or affine_mtx)
## support both 2D and 3D images
## support batch transformation

import torch
import torch.nn.functional as F
import torch.nn as nn

import imgtrans as imt


# some minor util functions to convert to flow grid
def convert_dvf(dvf, img_shape):
    """
    dvf = (B, H, W, (D), 2 or 3)
    img_shape = tuple(H, W, (D))
    convert dvf to flow grid
    """
    return imt.utils.grid_utils.dvf2flow_grid(dvf)


def warp2dvf(warp, img_shape):
    """
    convert warp to dvf
    rescale warp to percentage of pixel moved
    - warp = (B, H, W, (D), 2 or 3) -> contains information of pixel movement of each position, so need to be rescaled if want to use it.
    - dvf = (B, H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
    img_shape = tuple(H, W, (D))
    """
    # TODO: implement these functions and implement SpatialTransformer
    # TODO: test the spatial transformer
    return warp / img_shape.reshape(*[1] * (len(img_shape) + 1), len(img_shape))


def convert_warp(warp, img_shape):
    """
    convert warp to flow_grid
    rescale warp to percentage of pixel moved
    - warp = (B, H, W, (D), 2 or 3) -> contains information of pixel movement of each position, so need to be rescaled if want to use it.
    - flow_grid = (B, H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
    img_shape = tuple(H, W, (D))
    """
    dvf = warp2dvf(warp, img_shape)
    return imt.utils.grid_utils.dvf2flow_grid(dvf, img_shape)


def convert_grid(grid, img_shape):
    """
    convert grid to flow_grid
    rescale grid to [-1, 1]
    - grid = (B, H, W, (D), 2 or 3) -> denotes the positions, but as the unit is the dimension of the image, so need to be rescaled if want to use it.
    - flow_grid = (B, H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
    img_shape = tuple(H, W, (D))
    """
    flow_grid = grid * 2 / img_shape.reshape(*[1] * (len(img_shape) + 1), len(img_shape)) - 1
    return flow_grid


CONVERT_DICT = {"dvf": convert_dvf, "warp": warp2dvf, "grid": convert_grid, "flow_grid": convert_warp}


class SpatialTransformer(nn.Module):
    """
    perform spatial transformation for images, 2D or 3D, in a batched mode
    """

    def __init__(self, 
                 matrix_type: str, 
                 mode="bilinear",
                 padding_mode="zeros",
                 align_corners=True,
                 **kwargs):
        """
        matrix_type: "warp", "grid", "affine_mtx", "dvf", "flow_grid"
        - dvf = (B, H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
            e.g. -10 in (x, y, z, 1) means that pixel in poistion (x, y, z) needs to move to the left of X-axis (1) for 10 percent of pixels.
        - warp = (B, H, W, (D), 2 or 3) -> contains information of pixel movement of each position, so need to be rescaled if want to use it.
        - flow_grid = (B, H, W, (D), 2 or 3) -> range from [-1, 1], denotes not the movement, but the positions of the pixels, for more info see torch.nn.functional.grid_sample.
        - grid = (B, H, W, (D), 2 or 3) -> denotes the positions, but as the unit is the dimension of the image, so need to be rescaled if want to use it.
    
        kwargs: for grid_sample: mode, padding_mode, align_corners
        """
        super().__init__()
        self.matrix_type = matrix_type
        self.convert_func = CONVERT_DICT[matrix_type]
        self.kwargs = {"mode": mode, "padding_mode": padding_mode, "align_corners": align_corners, **kwargs}
        pass

    def forward(self, img: torch.Tensor, matrix: torch.Tensor, **kwargs):
        """
        img = (B, C, H, W, (D))
        matrix = (B, H, W, (D), ndim)
        """
        # some assertions
        assert img.shape[2:] == matrix.shape[1:-1], "img and matrix should have the same shape"
        assert img.shape[0] == matrix.shape[0], "img and matrix should have the same batch size"
        assert matrix.shape[-1] == len(img.shape) - 2, f"matrix have {matrix.shape[-1]} dims but got image size = {img.shape}"
        flow_grid = self.convert_func(matrix, img.shape[2:])
        # update self.kwargs with kwargs
        kwargs = {**self.kwargs.copy(), **(kwargs or {})}
        deformed_img = F.grid_sample(input=img,
                                     grid=flow_grid, # NOTE: (B, H, W, (D), ndim)
                                     **kwargs)
        return deformed_img, {"flow_grid": flow_grid}