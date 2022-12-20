# affine transformation

from typing import Optional, Sequence, Union, List

from scipy import ndimage
from imgtrans.utils.misc import ensure_tuple, ensure_tuple_size
from imgtrans.utils.aff_mtx import _create_rotate, _create_shear, _create_translate, _create_scale
import numpy as np



def AffineTransform:


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
        pass
    
    
    def __call__(
        self,
        rotate: Optional[List[float]] = None,
        shear: Optional[List[float]] = None,
        translate: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ):
        """
        NOTE: MONAI supports torch backend but now the env is in TF, so only numpy is used right now. Want to develop TF version if speed in need.
        Args:
            rotate: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.

        Raises:
            affine image matrix (3 * 3)

        """

        # _b = "torch" if isinstance(grid, torch.Tensor) else "numpy"
        # _device = grid.device if isinstance(grid, torch.Tensor) else self.device

        affine = np.eye(self.ndim + 1)
        if rotate:
            affine = affine @ create_rotate(self.ndim, rotate)
        if shear:
            affine = affine @ create_shear(self.ndim, shear)
        if translate:
            affine = affine @ create_translate(self.ndim, translate)
        if scale:
            affine = affine @ create_scale(self.ndim, scale)

        return affine


def create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    backend="numpy",
):
    """
    create a 2D or 3D rotation matrix

    Args:
        spatial_dims: {``2``, ``3``} spatial rank
        radians: rotation radians
            when spatial_dims == 3, the `radians` sequence corresponds to
            rotation in the 1st, 2nd, and 3rd dim respectively.

    Raises:
        ValueError: When ``radians`` is empty.
        ValueError: When ``spatial_dims`` is not one of [2, 3].

    """
    return _create_rotate(spatial_dims=spatial_dims,
                          radians=radians,
                          sin_func=np.sin,
                          cos_func=np.cos,
                          eye_func=np.eye)


def create_shear(
    spatial_dims: int,
    coefs: Union[Sequence[float], float],
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
        backend: APIs to use, ``numpy`` or ``torch``.

    Raises:
        NotImplementedError: When ``spatial_dims`` is not one of [2, 3].

    """
    return _create_shear(spatial_dims=spatial_dims,
                         coefs=coefs,
                         eye_func=np.eye)


def create_scale(
    spatial_dims: int,
    scaling_factor: Union[Sequence[float], float],
):
    """
    create a scaling matrix

    Args:
        spatial_dims: spatial rank
        scaling_factor: scaling factors for every spatial dim, defaults to 1.
        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.
    """
    return _create_scale(spatial_dims=spatial_dims,
                         scaling_factor=scaling_factor,
                         array_func=np.diag)
    

def create_translate(
    spatial_dims: int,
    shift: Union[Sequence[float], float],
):
    """
    create a translation matrix

    Args:
        spatial_dims: spatial rank
        shift: translate pixel/voxel for every spatial dim, defaults to 0.
        device: device to compute and store the output (when the backend is "torch").
        backend: APIs to use, ``numpy`` or ``torch``.
    """
    return _create_translate(spatial_dims=spatial_dims,
                             shift=shift,
                             eye_func=np.eye,
                             array_func=np.asarray)
