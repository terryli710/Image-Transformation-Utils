# affine transformation, from monai

from typing import Callable, List, Optional, Sequence, Union

import torch

from imgtrans.torch.utils.grid_utils import create_grid, resample
from imgtrans.torch.utils.randomize import RandomFromIntervalTorch
from imgtrans.torch.utils.type_utils import (convert_tenor_dtype,
                                             convert_to_dst_type)
from imgtrans.utils.affine_matrix import (_create_rotate, _create_scale,
                                    _create_shear, _create_translate)


class AffineKeyPoints(torch.nn.Module):
    """
    Affine transform keypoint coordinates
    """
    
    def __init__(self, 
                 device=None) -> None:
        super().__init__()
        """
        Defaults are all Nones, because the forward method is called with the
        kps where we can extract all the information
        """
        pass

    def forward(self, kps: torch.Tensor, affine_matrix: torch.Tensor):
        """
        Args:
            kps = ((*batch_size), nkps (like channel), ndim)
            affine_matrix = ((*batch_size), ndim(+1), ndim+1)
        """
        
        # initialize some parameters
        batch_size, nkps, ndim = kps.shape[:-2], kps.shape[-2], kps.shape[-1]
        device = kps.device
        # assert affine_matrix's shape is correct
        assert affine_matrix.shape[-2] in [ndim, ndim+1] and affine_matrix.shape[-1] == ndim + 1, \
            "affine matrix shape is incorrect, should be (*batch_size, ndim+1, ndim+1), but got {}".format(affine_matrix.shape)
        assert affine_matrix.shape[:-2] == batch_size, \
            f"affine matrix shape is incorrect, batch_size should be {batch_size}, but got {affine_matrix.shape[:-2]}"
        
        # convert kps to shape (B, ndim, nkps)
        kps = kps.view(-1, nkps, ndim).permute(0, 2, 1)
        B = kps.shape[0]
        
        # if 2D, needs to flip the x and y axis
        if ndim == 2:
            kps = torch.flip(kps, dims=[1])

        # compute the affine transformation
        kps_copy = kps.clone()
        ones = torch.ones(B, 1, nkps).to(device)
        kps_copy = torch.cat([kps_copy, ones], 1)
        result_kps = torch.bmm(affine_matrix[:, :ndim, :], kps_copy)
        
        # if 2D, needs to flip the x and y axis back
        if ndim == 2:
            result_kps = torch.flip(result_kps, dims=[1])
            
        # convert result_kps to shape ((*batch_size), nkps, ndim)
        result_kps = result_kps.permute(0, 2, 1).view((*batch_size, nkps, ndim))
        return result_kps


class AffineTransform(torch.nn.Module):
    
    def __init__(self, **resample_kwargs):
        super().__init__()
        self.resample_kwargs = resample_kwargs
        pass
    
    def forward(self, img: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: ((*batch_size,) (*channel_size,), *spatial_size)
            affine_matrix: (*batch_size, ndim+1, ndim+1)
        """
        ndim = affine_matrix.shape[-1] - 1
        spatial_size = img.shape[-ndim:]
        device = img.device
        
        # create the affine grid
        grid = create_grid(spatial_size, device=device)
        grid = convert_tenor_dtype(grid, device=device, dtype=float)
        affine_matrix = convert_to_dst_type(affine_matrix, grid)
        grid = torch.matmul(affine_matrix, grid)
        
        img = convert_tenor_dtype(img, device=device, dtype=float)
        return resample(img, grid, **self.resample_kwargs)
        
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)



class AffineMatrix:
    
    def __init__(self, ndim, random=True):
        """
        Args:
            ndim (int): 2 or 3
            random (bool): if True, the affine matrix will be generated randomly
            backend (str): "torch" or "numpy"
        """
        self.ndim = ndim
        self.random = random
        assert ndim in (2, 3), "Only support 2D or 3D affine matrix."
        self.rand_sampler = RandomFromIntervalTorch()
        pass
    
    def __call__(
                        self,
                        rotate: Optional[List[float]] = 0.0,
                        scale: Optional[List[float]] = 1.0,
                        translate: Optional[List[float]] = 0.0,
                        shear: Optional[List[float]] = 0.0,
                        nbatch: Optional[int] = None,
                    ):
        """
        NOTE: MONAI supports torch backend but now the env is in TF, so only numpy is used right now. Want to develop TF version if speed in need.
        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.

        Raises:
            affine image matrix (3 * 3)

        """
        if self.random:
            rotate = self.rand_sampler.get_randparams(rotate, nbatch, self.ndim)
            scale = self.rand_sampler.get_randparams(scale, nbatch, self.ndim)
            translate = self.rand_sampler.get_randparams(translate, nbatch, self.ndim)
            shear = self.rand_sampler.get_randparams(shear, nbatch, self.ndim)
        
        # affine_matrix = ((nbatch), ndim+1, ndim+1)
        if nbatch is None:
            affine_matrix = torch.eye(self.ndim + 1)
            if rotate:
                affine_matrix = affine_matrix @ create_rotate(self.ndim, rotate)
            if shear:
                affine_matrix = affine_matrix @ create_shear(self.ndim, shear)
            if translate:
                affine_matrix = affine_matrix @ create_translate(self.ndim, translate)
            if scale:
                affine_matrix = affine_matrix @ create_scale(self.ndim, scale)
        else:
            affine_matrix = torch.eye(self.ndim + 1)[None].repeat(nbatch, 1, 1)
            if rotate:
                affine_matrix = affine_matrix @ create_rotate(self.ndim, rotate)[:, None]
            if shear:
                affine_matrix = affine_matrix @ create_shear(self.ndim, shear)[:, None]
            if translate:
                affine_matrix = affine_matrix @ create_translate(self.ndim, translate)[:, None]
            if scale:
                affine_matrix = affine_matrix @ create_scale(self.ndim, scale)[:, None]

        return affine_matrix

def create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
):
    """
    create a 2D or 3D rotation matrix

    Args:
        spatial_dims: {``2``, ``3``} spatial rank
        radians: rotation radians
            when spatial_dims == 3, the `radians` sequence corresponds to
            rotation in the 1st, 2nd, and 3rd dim respectively.
        device: device to compute and store the output (when the backend is "torch").
        

    Raises:
        ValueError: When ``radians`` is empty.
        ValueError: When ``spatial_dims`` is not one of [2, 3].

    """
    return _create_rotate(
        spatial_dims=spatial_dims,
        radians=radians,
        sin_func=lambda th: torch.sin(
            torch.as_tensor(th, dtype=torch.float32, device=device)),
        cos_func=lambda th: torch.cos(
            torch.as_tensor(th, dtype=torch.float32, device=device)),
        eye_func=lambda rank: torch.eye(rank, device=device),
    )


def create_shear(
    spatial_dims: int,
    coefs: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
):
    """
    create a shearing matrix

    Args:
        spatial_dims: spatial rank
        coefs: shearing factors, a tuple of 2 floats for 2D, a tuple of 6 floats for 3D),
            take a 3D affine as example::

                [
                    [1.0, coefs[0], coefs[1], 0.0],
                    [coefs[2], 1.0, coefs[3], 0.0],
                    [coefs[4], coefs[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

        device: device to compute and store the output (when the backend is "torch").
        

    Raises:
        NotImplementedError: When ``spatial_dims`` is not one of [2, 3].

    """
    return _create_shear(spatial_dims=spatial_dims,
                         coefs=coefs,
                         eye_func=lambda rank: torch.eye(rank, device=device))


def create_scale(
    spatial_dims: int,
    scaling_factor: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
):
    """
    create a scaling matrix

    Args:
        spatial_dims: spatial rank
        scaling_factor: scaling factors for every spatial dim, defaults to 1.
        device: device to compute and store the output (when the backend is "torch").
        
    """

    return _create_scale(
        spatial_dims=spatial_dims,
        scaling_factor=scaling_factor,
        array_func=lambda x: torch.diag(torch.as_tensor(x, device=device)),
    )


def create_translate(
    spatial_dims: int,
    shift: Union[Sequence[float], float],
    device: Optional[torch.device] = None,
):
    """
    create a translation matrix

    Args:
        spatial_dims: spatial rank
        shift: translate pixel/voxel for every spatial dim, defaults to 0.
        device: device to compute and store the output (when the backend is "torch").
        
    """
    return _create_translate(
        spatial_dims=spatial_dims,
        shift=shift,
        eye_func=lambda x: torch.eye(torch.as_tensor(x), device=device
                                     ),  # type: ignore
        array_func=lambda x: torch.as_tensor(x, device=device),  # type: ignore
    )