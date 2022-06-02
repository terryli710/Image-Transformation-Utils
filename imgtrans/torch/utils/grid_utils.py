from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.nn.functional import grid_sample

from .resize import resize_channel_last


def create_control_grid(spatial_shape, spacing, homogeneous=True, dtype=float, device=None):
    """
    control grid with two additional point in each direction
    """
    grid_shape = []
    for d, s in zip(spatial_shape, spacing):
        d = int(d)
        if d % 2 == 0:
            grid_shape.append(np.ceil((d - 1.) / (2. * s) + 0.5) * 2. + 2.)
        else:
            grid_shape.append(np.ceil((d - 1.) / (2. * s)) * 2. + 3.)
    # grid_shape = (H+4, W+4, D+4), spacing=(Hs, Ws, Ds)
    return create_grid(grid_shape, spacing, homogeneous, dtype, device=device)



def create_grid(
    spatial_size: Sequence[int],
    spacing: Optional[Sequence[float]] = None,
    homogeneous: bool = True,
    dtype=torch.float32,
    device: Optional[torch.device] = None,
):
    """
    compute a `spatial_size` mesh with the torch API.
    """
    spacing = spacing or tuple(1.0 for _ in spatial_size)
    ranges = [
        torch.linspace(-(d - 1.0) / 2.0 * s, (d - 1.0) / 2.0 * s,
                       int(d),
                       device=device,
                       dtype=dtype) for d, s in zip(spatial_size, spacing)
    ]
    coords = torch.meshgrid(*ranges, indexing="ij")
    if not homogeneous:
        return torch.stack(coords)
    return torch.stack([*coords, torch.ones_like(coords[0])])



class Resample:

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        device: Optional[torch.device] = None,
    ):
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device: device on which the tensor will be allocated.
        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device

    def __call__(
        self,
        img: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):  # -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        assert isinstance(
            img, torch.Tensor), "input img must be supplied as a torch.Tensor"
        assert isinstance(
            grid, torch.Tensor
        ), "Error, grid argument must be supplied as a torch.Tensor"

        grid = (torch.tensor(grid) if not isinstance(grid, torch.Tensor) else
                grid.detach().clone())
        if self.device:
            img = img.to(self.device)
            grid = grid.to(self.device)

        for i, dim in enumerate(img.shape[1:]):
            grid[i] = 2.0 * grid[i] / (dim - 1.0)

        grid = grid[:-1] / grid[-1:]
        index_ordering: List[int] = list(range(img.ndimension() - 2, -1, -1))
        grid = grid[index_ordering]
        grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
        out = grid_sample(
            img.unsqueeze(0).float(),
            grid.unsqueeze(0).float(),
            mode=self.mode if mode is None else mode,
            padding_mode=self.padding_mode
            if padding_mode is None else padding_mode,
            align_corners=False,  # NOTE: guess not much difference: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        )[0]
        return torch.as_tensor(out)
    
    

def dvf2flow_grid(dvf, out_shape=None):
    
    """
    convert dvf to flow_grid of torch
    Another thing we call warp, which is similar to dvf, except that it contains pixel distance instead of persentage movement, so need to be rescaled if want to move it.
    dvf = (H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
    e.g. -10 in (x, y, z, 1) means that pixel in poistion (x, y, z) needs to move to the left of X-axis for 10 percent of pixels

    flow_grid = (H, W, (D), 2 or 3)  range from [-1, 1], for more info see torch.nn.functional.grid_sample
    
    Args:
        dvf (torch.tensor): (H, W, (D), 2 or 3) a matrix contains information of pixel pertange movement of each position
        out_shape (array like): (H, W, (D)), shape of the output
    """
    
    if not out_shape:
        out_shape = dvf.shape[:-1]
    # ndim = len(out_shape)
    # 1. generate range matrix (max100 - min0 = 100)
    ls = [torch.linspace(0, 100, i).type_as(dvf) for i in dvf.shape[:-1]]
    # NOTE: indexing = "xy" doesn't work for 3D cases
    # so just using torch.flip with "ij" indexing
    mesh = torch.stack(torch.meshgrid(*ls, indexing="ij"), axis=-1) # (H, W, (D), 2 or 3)
    mesh = torch.flip(mesh, [-1]) # NOTE: somehow the format is ((z), y, x) so have to flip

    # 2. scale -> from -1 to 1, (max1 - min-1 = 2)
    assert mesh.shape == dvf.shape
    flow_grid = (mesh + dvf) / 50 - 1

    # 3. resize the flow_grid
    if flow_grid.shape != out_shape:
        flow_grid = resize_channel_last(flow_grid[None, ...], out_shape)[0, ...]
    return flow_grid


def flow_grid2dvf(flow_grid, out_shape=None):
    """
    convert dvf to flow_grid of torch
    dvf = (H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
    e.g. -10 in (x, y, z, 1) means that pixel in poistion (x, y, z) needs to move to the left of X-axis for 10 percent of pixels

    flow_grid = (H, W, (D), 2 or 3)  range from [-1, 1], for more info see torch.nn.functional.grid_sample
    
    Args:
        dvf (torch.tensor): (H, W, (D), 2 or 3) a matrix contains information of pixel pertange movement of each position
        out_shape (array like): (H, W, (D)), shape of the output
    """
    if not out_shape:
        out_shape = flow_grid.shape
    # ndim = len(out_shape)
    # 1. generate range matrix range from -1 to 1
    ls = [torch.linspace(-1, 1, i) for i in flow_grid.shape[:-1]]
    mesh = torch.stack(torch.meshgrid(*ls, indexing="ij"), axis=-1)
    mesh = torch.flip(mesh, [-1])
    
    # 2. scale -> from 0 to 100
    assert mesh.shape == flow_grid.shape
    dvf = (flow_grid - mesh) * 50
    
    # 3. resize the flow_grid
    if dvf.shape != out_shape:
        dvf = resize_channel_last(dvf[None, ...], out_shape)[0, ...]
    return dvf
