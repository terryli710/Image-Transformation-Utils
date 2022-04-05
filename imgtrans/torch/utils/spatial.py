import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode


def resize2d(image, target_size):
    """ 
    2D pytorch resize method
    
    Args:
        image (torch.tensor): [N, C, H, W,]
        target_size (array): (H, W)
    """
    return Resize(target_size, interpolation=InterpolationMode.BILINEAR)(image)

def resize3d(image, target_size):
    """ 3D pytorch resize method

    Args:
        image (torch.tensor): [N, C, H, W, D]
        target_size (array): (H, W, D)
    """
    h = torch.linspace(-1, 1, target_size[0])
    w = torch.linspace(-1, 1, target_size[1])
    d = torch.linspace(-1, 1, target_size[2])
    meshz, meshy, meshx = torch.meshgrid((h, w, d))
    grid = torch.stack((meshx, meshy, meshz))
    grid = grid.unsqueeze(0)
    
    out = F.grid_sample(image, grid, align_corners=True)
    return out


def resize(image, target_size):
    """ 2D + 3D pytorch image resize method 

    Args:
        image (torch.tensor): [N, C, H, W(, D)]
        target_size (array): (H, W(, D))
    NOTE: be careful with input size ...
    """
    assert len(image.shape) == len(target_size) + 2
    if len(image.shape) == 4:
        # 2D
        return resize2d(image, target_size)
    elif len(image.shape) == 5:
        # 3D
        return resize3d(image, target_size)
    else:
        raise NotImplementedError
    


def draw_perlin(out_shape,
                scales,
                min_std=0,
                max_std=1,
                dtype=torch.float32,
                seed=None):
    '''
    Generate Perlin noise by drawing from Gaussian distributions at different
    resolutions, upsampling and summing. There are a couple of key differences
    between this function and the Neurite equivalent ne.utils.perlin_vol, which
    are not straightforwardly consolidated.
    Neurite function:
        (1) Iterates over scales in range(a, b) where a, b are input arguments.
        (2) Noise volumes are sampled at resolutions vol_shape / 2**scale.
        (3) Noise volumes are sampled uniformly in the interval [0, 1].
        (4) Volume weights are {1, 2, ...N} (normalized) where N is the number
            of scales, or sampled uniformly from [0, 1].
    This function:
        (1) Specific scales are passed as a list.
        (2) Noise volumes are sampled at resolutions vol_shape / scale.
        (3) Noise volumes are sampled normally, with SDs drawn uniformly from
            [min_std, max_std].
    Parameters:
        out_shape: List defining the output shape. In N-dimensional space, it
            should have N+1 elements, the last one being the feature dimension.
        scales: List of relative resolutions at which noise is sampled normally.
            A scale of 2 means half resolution relative to the output shape.
        min_std: Minimum standard deviation (SD) for drawing noise volumes.
        max_std: Maximum SD for drawing noise volumes.
        dtype: Output data type.
        seed: Integer for reproducible randomization. 
    '''
    # out_shape = torch.tensor(out_shape, dtype=torch.int32)
    if np.isscalar(scales):
        scales = [scales]
    # set random seed, would work within the function
    np.random.seed(seed)
    out = torch.zeros(out_shape[-1], *out_shape[:-1], dtype=dtype) # channel first (C, H, W, (D))
    
    for scale in scales:
        sample_shape = np.ceil(tuple(s / scale for s in out_shape[:-1])).astype("int32")
        sample_shape = (*sample_shape, out_shape[-1])
        std = np.random.uniform(0, 1) * \
            (max_std - min_std) + min_std
        gauss = np.random.normal(size=sample_shape, loc=0, scale=std)
        gauss = torch.from_numpy(gauss).type(dtype)

        # zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        
        # scale gauss and add to out
        if scale != 1:
            # change to channel first
            if len(sample_shape) == 3:
                # 2d
                gauss = gauss.permute(2, 0, 1)
            elif len(sample_shape) == 4:
                # 3d
                gauss = gauss.permute(3, 0, 1, 2)
            else:
                raise NotImplementedError
            
            gauss = resize(gauss.unsqueeze(0), # create batch dim
                           target_size=out_shape[:-1])[0, ...]

        out += gauss
    
    # channel last
    if len(sample_shape) == 3:
        # 2d
        out = out.permute(1, 2, 0)
    elif len(sample_shape) == 4:
        # 3d
        out = out.permute(1, 2, 3, 0)
    else:
        raise NotImplementedError
    return out # shape = (H, W, (D), 2 or 3)



def dvf2flow_grid(dvf, out_shape):
    """
    convert dvf to flow_grid of torch
    dvf = (H, W, (D), 2 or 3) -> contains information of pixel pertange movement of each position
    e.g. -10 in (x, y, z, 1) means that pixel in poistion (x, y, z) needs to move to the left of X-axis for 10 percent of pixels

    flow_grid = (H, W, (D), 2 or 3)  range from [-1, 1], for more info see torch.nn.functional.grid_sample
    
    Args:
        dvf (torch.tensor): (H, W, (D), 2 or 3) a matrix contains information of pixel pertange movement of each position
        out_shape (array like): (H, W, (D)), shape of the output
    """
    
    ndim = len(out_shape)
    # 1. generate range matrix
    ls = torch.linspace(0, 99, 100)
    mesh = torch.stack(torch.meshgrid(*([ls] * ndim)), axis=-1)
    
    # 2. scale -> from -1 to 1
    flow_grid = (mesh + dvf) / 50 - 1
    
    # 3. scale the whole flow_grid
    if flow_grid.shape != out_shape:
        # change to channel first
        if len(flow_grid.shape) == 3:
            # 2d
            flow_grid = flow_grid.permute(2, 0, 1)
        elif len(flow_grid.shape) == 4:
            # 3d
            flow_grid = flow_grid.permute(3, 0, 1, 2)
        else:
            raise NotImplementedError
    
        flow_grid = resize(flow_grid.unsqueeze(0), target_size=out_shape)[0,...]

        # back to channel last
        if len(flow_grid.shape) == 3:
            # 2d
            flow_grid = flow_grid.permute(1, 2, 0)
        elif len(flow_grid.shape) == 4:
            # 3d
            flow_grid = flow_grid.permute(1, 2, 3, 0)
        else:
            raise NotImplementedError
    return flow_grid