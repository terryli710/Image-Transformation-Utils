import tensorflow as tf
from .resize import resize_channel_last


def dvf2flow_grid(dvf, out_shape=None):
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
        out_shape = dvf.shape[:-1]
    # 1. generate range matrix
    ls = [tf.linspace(0, 100, i) for i in dvf.shape[:-1]]
    mesh = tf.cast(tf.stack(tf.meshgrid(*ls), axis=-1), dvf.dtype) # dtype is the same as dvf

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
    ls = [tf.linspace(-1, 1, i) for i in flow_grid.shape[:-1]]
    mesh = tf.stack(tf.meshgrid(*ls), axis=-1)
    
    # 2. scale -> from 0 to 100
    assert mesh.shape == flow_grid.shape
    dvf = (flow_grid - mesh) * 50
    
    # 3. resize the flow_grid
    if dvf.shape != out_shape:
        dvf = resize_channel_last(dvf[None, ...], out_shape)[0, ...]
    return dvf

