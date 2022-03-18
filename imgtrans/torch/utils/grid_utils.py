from typing import Optional, Sequence
import torch


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
