# affine transformation

import numpy as np
from scipy import ndimage
from typing import Union, Sequence, Optional, Callable, List
from imgtrans.utils.misc import ensure_tuple, ensure_tuple_size
from .utils.randomize import RandParams



class Affine:

    def __init__(
        self,
        spatial_dims=None,
        rotate: Optional[Union[Sequence[float], float]] = None,
        shear: Optional[Union[Sequence[float], float]] = None,
        translate: Optional[Union[Sequence[float], float]] = None,
        scale: Optional[Union[Sequence[float], float]] = 1.0,
        padding_mode: str = "reflect",
        image_only: bool = False,
    ):
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

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
            padding_mode: {‘reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’},
                Padding mode for outside grid values. Defaults to ``"reflect"``.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
            image_only: if True return only the image volume, otherwise return (image, affine).

        """

        self.aff_mtx = self._get_aff_mtx(
            spatial_dims=spatial_dims,
            rotate=rotate,
            shear=shear,
            translate=translate,
            scale=scale,
        )
        self.image_only = image_only
        self.padding_mode: padding_mode
        pass

    def __call__(
        self,
        img: np.ndarray,
        spatial_dims=None,
        rotate: Optional[Union[Sequence[float], float]] = None,
        shear: Optional[Union[Sequence[float], float]] = None,
        translate: Optional[Union[Sequence[float], float]] = None,
        scale: Optional[Union[Sequence[float], float]] = None,
        padding_mode: Optional[str] = None,
    ):
        if any([
                p is not None for p in
            [rotate, shear, translate, scale]
        ]):

            aff_mtx = self._get_aff_mtx(
                spatial_dims=spatial_dims
                if spatial_dims is not None else len(img.shape),
                rotate=rotate
                if rotate is not None else self.rotate,
                shear=shear
                if shear is not None else self.shear,
                translate=translate
                if translate is not None else self.translate,
                scale=scale
                if scale is not None else self.scale,
            )
        else:
            aff_mtx = self.aff_mtx
        trans_img = ndimage.affine_transform(
            img,
            aff_mtx,
            mode=padding_mode if padding_mode else self.padding_mode)
        return trans_img if self.image_only else trans_img, {
            "aff_mtx": aff_mtx
        }

    def _get_aff_mtx(
        self,
        spatial_dims,
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

        affine = np.eye(spatial_dims + 1)
        if rotate:
            affine = affine @ create_rotate(spatial_dims, rotate)
        if shear:
            affine = affine @ create_shear(spatial_dims, shear)
        if translate:
            affine = affine @ create_translate(spatial_dims, translate)
        if scale:
            affine = affine @ create_scale(spatial_dims, scale)

        return affine


class RandAffine(Affine, RandParams):

    def __init__(self,
                 spatial_dims=None,
                 rotate_range=None,
                 shear_range=None,
                 translate_range=None,
                 scale_range=None,
                 padding_mode="reflect",
                 seed=None):

        Affine.__init__(self,
                        spatial_dims=spatial_dims,
                        padding_mode=padding_mode,)
        RandParams.__init__(self, seed=seed)
        self.spatial_dims = spatial_dims
        self.rotate_range = rotate_range
        self.shear_range = shear_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        pass

    def randomize(self, spatial_dims=2, seed=None):
        if seed:
            self.set_random_state(seed)
        if spatial_dims == 2:
            params = {
                "rotate":self.get_randparam(self.rotate_range, dim=2),
                "shear":self.get_randparam(self.shear_range, dim=2),
                "translate":self.get_randparam(self.translate_range, dim=2),
                "scale":self.get_randparam(self.scale_range, dim=2, abs=True),
            }
        elif spatial_dims == 3:
            params = {
                "rotate":self.get_randparam(self.rotate_range, dim=3),
                "shear":self.get_randparam(self.shear_range, dim=6),
                "translate":self.get_randparam(self.translate_range, dim=3),
                "scale":self.get_randparam(self.scale_range, dim=3, abs=True),
            }
        else:
            raise NotImplementedError
        return params

    def __call__(self, 
                 img: np.ndarray, 
                 padding_mode="reflect", 
                 seed=None):
        spatial_dims = self.spatial_dims if self.spatial_dims else len(img.shape)
        params = self.randomize(spatial_dims=spatial_dims, seed=seed)
        aff_mtx = self._get_aff_mtx(spatial_dims=spatial_dims, **params)
        trans_img = ndimage.affine_transform(
            img,
            aff_mtx,
            mode=padding_mode if padding_mode else self.padding_mode)
        return trans_img if self.image_only else trans_img, {
            "aff_mtx": aff_mtx,
            "params": params,
        }


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


def _create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    sin_func: Callable = np.sin,
    cos_func: Callable = np.cos,
    eye_func: Callable = np.eye,
):
    radians = ensure_tuple(radians)
    if spatial_dims == 2:
        if len(radians) >= 1:
            sin_, cos_ = sin_func(radians[0]), cos_func(radians[0])
            out = eye_func(3)
            out[0, 0], out[0, 1] = cos_, -sin_
            out[1, 0], out[1, 1] = sin_, cos_
            return out  # type: ignore
        raise ValueError("radians must be non empty.")

    if spatial_dims == 3:
        affine = None
        if len(radians) >= 1:
            sin_, cos_ = sin_func(radians[0]), cos_func(radians[0])
            affine = eye_func(4)
            affine[1, 1], affine[1, 2] = cos_, -sin_
            affine[2, 1], affine[2, 2] = sin_, cos_
        if len(radians) >= 2:
            sin_, cos_ = sin_func(radians[1]), cos_func(radians[1])
            if affine is None:
                raise ValueError("Affine should be a matrix.")
            _affine = eye_func(4)
            _affine[0, 0], _affine[0, 2] = cos_, sin_
            _affine[2, 0], _affine[2, 2] = -sin_, cos_
            affine = affine @ _affine
        if len(radians) >= 3:
            sin_, cos_ = sin_func(radians[2]), cos_func(radians[2])
            if affine is None:
                raise ValueError("Affine should be a matrix.")
            _affine = eye_func(4)
            _affine[0, 0], _affine[0, 1] = cos_, -sin_
            _affine[1, 0], _affine[1, 1] = sin_, cos_
            affine = affine @ _affine
        if affine is None:
            raise ValueError("radians must be non empty.")
        return affine  # type: ignore

    raise ValueError(
        f"Unsupported spatial_dims: {spatial_dims}, available options are [2, 3]."
    )


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


def _create_shear(spatial_dims: int,
                  coefs: Union[Sequence[float], float],
                  eye_func=np.eye):
    if spatial_dims == 2:
        coefs = ensure_tuple_size(coefs, dim=2, pad_val=0.0)
        out = eye_func(3)
        out[0, 1], out[1, 0] = coefs[0], coefs[1]
        return out  # type: ignore
    if spatial_dims == 3:
        coefs = ensure_tuple_size(coefs, dim=6, pad_val=0.0)
        out = eye_func(4)
        out[0, 1], out[0, 2] = coefs[0], coefs[1]
        out[1, 0], out[1, 2] = coefs[2], coefs[3]
        out[2, 0], out[2, 1] = coefs[4], coefs[5]
        return out  # type: ignore
    raise NotImplementedError("Currently only spatial_dims in [2, 3] are supported.")


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


def _create_scale(spatial_dims: int,
                  scaling_factor: Union[Sequence[float], float],
                  array_func=np.diag):
    scaling_factor = ensure_tuple_size(scaling_factor,
                                       dim=spatial_dims,
                                       pad_val=1.0)
    return array_func(scaling_factor[:spatial_dims] + (1.0, ))  # type: ignore


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


def _create_translate(spatial_dims: int,
                      shift: Union[Sequence[float], float],
                      eye_func=np.eye,
                      array_func=np.asarray):
    shift = ensure_tuple(shift)
    affine = eye_func(spatial_dims + 1)
    for i, a in enumerate(shift[:spatial_dims]):
        affine[i, spatial_dims] = a
    return array_func(affine)  # type: ignore

