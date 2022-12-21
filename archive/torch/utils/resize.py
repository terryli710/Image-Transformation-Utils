
import torch
import torch.nn.functional as F
# from torchvision.transforms import Resize, InterpolationMode


def _resize2d(image, target_size, interp_method="bilinear"):
    """ 
    2D pytorch resize method
    
    Args:
        image (torch.tensor): [N, C, H, W,]
        target_size (array): (H, W)
    """
    scale_factor = [t / o for t, o in zip(target_size, image.shape[2:])]
    return F.interpolate(image,
                         align_corners=True,
                        #  size=target_size,
                         scale_factor=scale_factor,
                         recompute_scale_factor=True,
                         mode=interp_method)


def _resize3d(image, target_size, interp_method="bilinear"):
    """ 3D pytorch resize method

    Args:
        image (torch.tensor): [N, C, H, W, D]
        target_size (array): (H, W, D)
    """
    # put on the same device
    lns = [torch.linspace(-1, 1, target_size[i]).type_as(image) for i in range(3)]
    
    meshz, meshy, meshx = torch.meshgrid(lns, indexing="ij")
    grid = torch.stack((meshx, meshy, meshz), dim=-1)
    grid = grid.unsqueeze(0)
    grid.requires_grad = image.requires_grad

    out = F.grid_sample(image, grid, align_corners=True, mode=interp_method)
    return out


def resize(image, target_size, interp_method="bilinear"):
    """ 2D + 3D pytorch image resize method 

    Args:
        image (torch.tensor): [N, C, H, W(, D)]
        target_size (array): tuple indicating (H, W(, D))
    NOTE: be careful with input size ...
    """

    assert len(image.shape) == len(target_size) + 2, f"image shape and target_size must not match, image_shape={image.shape}, target_size={target_size}"
    if len(image.shape) == 4:
        # 2D
        return _resize2d(image, target_size, interp_method=interp_method)
    elif len(image.shape) == 5:
        # 3D
        return _resize3d(image, target_size, interp_method=interp_method)
    else:
        raise NotImplementedError
    

def resize_channel_last(image, target_size):
    """
    Resize image to target_size, channel last
    image = (N, H, W, (D), C)
    target_size = (H, W, (D))
    """
    # 3. scale the whole flow_grid
    if image.shape[1:-1] != target_size:
        # change to channel first
        if len(image.shape) == 4:
            # 2d
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 5:
            # 3d
            image = image.permute(0, 4, 1, 2, 3)
        else:
            raise NotImplementedError
    
        image = resize(image, target_size=target_size)

        # back to channel last
        if len(image.shape) == 4:
            # 2d
            image = image.permute(0, 2, 3, 1)
        elif len(image.shape) == 5:
            # 3d
            image = image.permute(0, 2, 3, 4, 1)
        else:
            raise NotImplementedError
    return image
