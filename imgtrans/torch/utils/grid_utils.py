from typing import Optional, Sequence, List, Union
import torch
from torch.nn.functional import grid_sample
import torch.nn as nn
import numpy as np


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
    coords = torch.meshgrid(*ranges)
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
        return torch.as_tensor(out), grid
    
    
