""" perlin transform, inspired by neurite package """
import torch
import torch.nn.functional as F
from .utils.spatial import dvf2flow_grid, draw_perlin

class RandPerlin:
    """
    random perlin transformations (use perlin noise as deformation field)
    """

    def __init__(self, scales=(32, 64), min_std=0, max_std=1):
        self.scales = scales
        self.min_std = min_std
        self.max_std = max_std
        pass

    def __call__(
        self,
        img: torch.Tensor,
        out_shape=None,
        mode="nearest",
        padding_mode="reflection",
        dtype=None,
        seed=None,
    ):  # TODO: std the input and output of torch transformations
        """ 
        img = (C or B, H, W, (D))
        
        """
        if not out_shape:
            out_shape = img.shape[1:]
        if not dtype:
            dtype = img.dtype

        ndim = len(out_shape)

        perlin_warp = draw_perlin(out_shape=(*out_shape, ndim),
                                  scales=self.scales,
                                  min_std=self.min_std,
                                  max_std=self.max_std,
                                  dtype=dtype,
                                  seed=seed)

        # convert range of warp from percentage of pixel moved (xvm.utils.transform)
        # to location of image from -1 to 1 (torch.nn.functional.grid_sample)

        flow_grid = dvf2flow_grid(perlin_warp, out_shape)

        deformed_img = F.grid_sample(input=img,
                                     gird=flow_grid,
                                     mode=mode,
                                     padding_mode=padding_mode,
                                     align_corners=True)
        
        return deformed_img, {"flow_grid": flow_grid}

