from typing import Union
import torch


def resample(data: torch.Tensor, grid: torch.Tensor, **kwargs):
    """
    Args:
        data: ((B dims), (C dims), H, W[, D])
        grid: ((B dims), H, W[, D], 2 or 3)
        where:
            (B dims) is arbitrary batch dimensions, if it's the same for data and grid, 
            then each data is resampled with the corresponding grid.
            if data has more batch dims than grid, then those are converted to be channels (C dims), 
            which will be resampled with the same grid.
    """
    ndim: int = grid.shape[-1]
    batch_size: torch.Size = grid.shape[:-ndim - 1]
    spatial_size: torch.Size = grid.shape[-ndim - 1:-1]
    channel_size: torch.Size = data.shape[len(batch_size):-ndim]
    data_batch_size, data_spatial_size = data.shape[:len(batch_size)], data.shape[-ndim:]
    
    assert batch_size == data_batch_size, f"batch dims must be the same for grid and data, but got {batch_size} and {data.shape[:-ndim - 1]}"
    assert spatial_size == data_spatial_size, f"spatial dims must be the same for grid and data, but got {spatial_size} and {data.shape[-ndim - 1:-1]}"
    
    # reshape data to (B, C, H, W[, D]), grid to (B, H, W[, D], 2 or 3)
    data = data.reshape(torch.prod(torch.tensor(batch_size)), 
                        torch.prod(torch.tensor(channel_size)), 
                        *spatial_size)
    grid = grid.reshape(torch.prod(torch.tensor(batch_size)),
                        *spatial_size,
                        ndim)
    resampled_data = torch.nn.functional.grid_sample(input=data, grid=grid, **kwargs)
    resampled_data = resampled_data.reshape(*batch_size, *channel_size, *spatial_size)
    return resampled_data


def get_mesh(mesh_shape, device=None, dtype=torch.float):
    """ 
    create a  meshgrid of shape `shape` 
    mesh_shape: (H, W[, D])
    Returns:
        mesh: (H, W[, D], 2 or 3)
    """
    ndims = len(mesh_shape)
    ls = [torch.arange(0, s).to(device).to(dtype) for s in mesh_shape]
    # NOTE: indexing = "xy" doesn't work for 3D cases
    # so just using torch.flip with "ij" indexing
    mesh = torch.stack(torch.meshgrid(*ls, indexing="ij")[::-1], axis=-1) # (H, W, (D), 2 or 3)
    return mesh


def rescale_grid(grid: torch.Tensor, orig_range, new_range):
    """
    Rescale grid from orig_range to new_range
    Args:
        grid: (..., H, W[, D], 2 or 3)
        orig_range: (min, max) or [(min, max), (min, max), ...] or "pixel" or "percentage"
        new_range: (min, max) or [(min, max), (min, max), ...] or "pixel" or "percentage"
    Returns:
        grid: (..., H, W[, D], 2 or 3), with values in new_range
    """
    ndim = grid.shape[-1]
    spatial_dims = grid.shape[-ndim - 1:-1]
    # ranges -> (ndim, 2)
    orig_range = process_range(orig_range, spatial_dims)
    new_range = process_range(new_range, spatial_dims)
    orig_range = torch.tensor(orig_range, device=grid.device, dtype=grid.dtype)
    new_range = torch.tensor(new_range, device=grid.device, dtype=grid.dtype)
    
    # asser the range shapes are valid
    assert orig_range.shape == new_range.shape == (ndim, 2), f"orig_range and new_range must be of shape ({ndim}, 2), but got {orig_range.shape} and {new_range.shape}"

    
    # scale on the last dimension
    times  = (new_range[:, 1] - new_range[:, 0]) / (orig_range[:, 1] - orig_range[:, 0])
    add    = new_range[:, 0]
    grid = grid * times.reshape(*[1] * (len(grid.shape) - 1), ndim) + add.reshape(*[1] * (len(grid.shape) - 1), ndim)
    return grid # (..., H, W[, D], 2 or 3)


def process_range(val_range, spatial_dims):
    """
    Args:
        val_range: (min, max) or [(min, max), (min, max), ...] or "pixel" or "percentage"
    Returns:
        val_range: [(min, max), (min, max), ...]
    """
    if val_range == "pixel":
        val_range = [(0, s) for s in spatial_dims]
    elif val_range == "percentage":
        val_range = [(0, 100) for _ in spatial_dims]
    elif isinstance(val_range, (tuple, list)):
        if len(val_range) == 2 and isinstance(val_range[0], (int, float)):
            val_range = [val_range for _ in spatial_dims]
        else:
            assert len(val_range) == len(spatial_dims), f"val_range must be of length 2 or {len(spatial_dims)}, but got {len(val_range)}"
    else:
        raise ValueError(f"val_range must be (min, max) or [(min, max), (min, max), ...] or 'pixel' or 'percentage', but got {val_range}")
    return val_range


def pos2mov_grid(pos_grid: torch.Tensor, val_range=Union[str, tuple]):
    """
    Args:
        pos_grid: (..., H, W[, D], 2 or 3), contains positions of each pixel
        val_range: either a tuple of (min, max) e.g. (-1, 1) or "pixel" or "percentage"
    """
    ndim = pos_grid.shape[-1]
    spatial_dims = pos_grid.shape[-ndim - 1:-1]
    mesh = get_mesh(spatial_dims, device=pos_grid.device, dtype=pos_grid.dtype)
    
    # deal with val_range
    mesh_range = [(0, s) for s in spatial_dims]
    
    mesh = rescale_grid(mesh, mesh_range, val_range)
    
    # pos_grid.shape = (..., H, W[, D], 2 or 3), mesh.shape = (H, W[, D], 2 or 3)
    return pos_grid - mesh


def mov2pos_grid(mov_grid: torch.Tensor, val_range=Union[str, tuple]):
    """
    Args:
        mov_grid: (..., H, W[, D], 2 or 3), contains movements of each pixel
        val_range: either a tuple of (min, max) e.g. (-1, 1) or "pixel" or "percentage"
    """
    ndim = mov_grid.shape[-1]
    spatial_dims = mov_grid.shape[-ndim - 1:-1]
    mesh = get_mesh(spatial_dims, device=mov_grid.device, dtype=mov_grid.dtype)
    
    # deal with val_range
    mesh_range = [(0, s) for s in spatial_dims]
    
    mesh = rescale_grid(mesh, mesh_range, val_range)
    
    # pos_grid.shape = (..., H, W[, D], 2 or 3), mesh.shape = (H, W[, D], 2 or 3)
    return mov_grid + mesh
    
    