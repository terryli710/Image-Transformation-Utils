# adopted from voxelmorph and neurite, using affine_matrix -> dense warp transform
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
from imgtrans.utils.misc import ensure_tuple, ensure_tuple_size

from .utils.randomize import RandParams
from .utils.transform import AffineSpatialTransformer


class Affine:

    def __init__(
        self,
        spatial_dims=2,
        rotate: Optional[Union[Sequence[float], float]] = None,
        shear: Optional[Union[Sequence[float], float]] = None,
        translate: Optional[Union[Sequence[float], float]] = None,
        scale: Optional[Union[Sequence[float], float]] = 1.0,
        mode: str = 'linear', # not bilinear
        padding_mode: str = "reflect", # HACK: not useful
        device: Optional[str] = None,
    ):
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

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
            NOTE: need to change: padding_mode: {‘reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’},
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
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device # HACK: not useful, just for consistancy
        self.resampler = AffineSpatialTransformer(interp_method=mode,
                                  indexing="ij",)
        pass

    def _get_aff_mtx(self,
        spatial_dims,
        rotate: Optional[List[float]] = None,
        shear: Optional[List[float]] = None,
        translate: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ):
        """
        Args:
            spatial_dims:
            rotate:
            shear:
            translate:
            scale:

        Returns:

        """
        
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

    def __call__(self,
        img: tf.Tensor,
        spatial_dims=None,
        rotate: Optional[Union[Sequence[float], float]] = None,
        shear: Optional[Union[Sequence[float], float]] = None,
        translate: Optional[Union[Sequence[float], float]] = None,
        scale: Optional[Union[Sequence[float], float]] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        
        """_summary_

        Args:
            img (torch.Tensor): shape = (n_channel/B, H, W (,D))
            spatial_dims (array like, optional): spatial dimentions of the image. Defaults to len(img.shape[1:]).
            other params please refer to _get_aff_mtx

        Returns:
            tuple: img, {dict of params}
        """
        # create aff_mtx
        if any([
                p is not None for p in
            [rotate, shear, translate, scale]
        ]):

            aff_mtx = self._get_aff_mtx(
                spatial_dims=spatial_dims
                if spatial_dims is not None else len(img.shape[1:]),
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
        # aff_mtx = (N+1, N+1) -> (1, N, N+1)
        aff_mtx = aff_mtx[None, :-1, :]
        # NOTE: do not use "call" function
        trans_img, warp = self.resampler([img[..., None], aff_mtx])
        trans_img = trans_img[..., 0]
        return trans_img, {
            "aff_mtx": aff_mtx,
            "warp": warp,
        }
        

class RandAffine(Affine, RandParams):

    def __init__(self,
                 spatial_dims=2,
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
                "rotate": self.get_randparam(self.rotate_range, dim=2),
                "shear": self.get_randparam(self.shear_range, dim=2),
                "translate": self.get_randparam(self.translate_range, dim=2),
                "scale": self.get_randparam(self.scale_range, dim=2, abs=True),
            }
        elif spatial_dims == 3:
            params = {
                "rotate": self.get_randparam(self.rotate_range, dim=3),
                "shear": self.get_randparam(self.shear_range, dim=6),
                "translate": self.get_randparam(self.translate_range, dim=3),
                "scale": self.get_randparam(self.scale_range, dim=3, abs=True),
            }
        else:
            raise NotImplementedError
        return params

    def __call__(self,
                 img: tf.Tensor,
                 mode: Optional[str] = None,
                 padding_mode="reflection",
                 seed=None):

        spatial_dims = self.spatial_dims if self.spatial_dims else len(
            img.shape[1:])
        params = self.randomize(spatial_dims=spatial_dims, seed=seed)

        result = Affine.__call__(self,
                                 img=img,
                                 spatial_dims=spatial_dims,
                                 mode=mode,
                                 padding_mode=padding_mode,
                                 **params)
        if not self.image_only:
            meta = result[1]
            meta.update({"params":params})
            result = (result[0], meta)

        return result


def create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    device: Optional[str] = None,
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
        sin_func=lambda th: tf.math.sin(
            tf.convert_to_tensor(th, dtype=tf.float32)),
        cos_func=lambda th: tf.math.cos(
            tf.convert_to_tensor(th, dtype=tf.float32)),
        eye_func=lambda rank: np.eye(rank), # need to check
    )


def _create_rotate(
    spatial_dims: int,
    radians: Union[Sequence[float], float],
    sin_func: Callable,
    cos_func: Callable,
    eye_func: Callable,
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
    device: Optional[str] = None,
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
                         eye_func=lambda rank: np.eye(rank))


def _create_shear(
    spatial_dims: int,
    coefs: Union[Sequence[float], float],
    eye_func: Callable,
):
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
    raise NotImplementedError(
        "Currently only spatial_dims in [2, 3] are supported.")


def create_scale(
    spatial_dims: int,
    scaling_factor: Union[Sequence[float], float],
    device: Optional[str] = None,
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
        array_func=lambda x: tf.linalg.diag(tf.convert_to_tensor(x)),
    )


def _create_scale(spatial_dims: int, scaling_factor: Union[Sequence[float],
                                                           float],
                  array_func: Callable):
    scaling_factor = ensure_tuple_size(scaling_factor,
                                       dim=spatial_dims,
                                       pad_val=1.0)
    return array_func(scaling_factor[:spatial_dims] + (1.0, ))  # type: ignore


def create_translate(
    spatial_dims: int,
    shift: Union[Sequence[float], float],
    device: Optional[str] = None,
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
        eye_func=lambda x: np.eye(tf.convert_to_tensor(x)),
        array_func=lambda x: tf.convert_to_tensor(x, dtype=tf.float32),  # type: ignore
    )


def _create_translate(spatial_dims: int, shift: Union[Sequence[float], float],
                      eye_func: Callable, array_func: Callable):
    shift = ensure_tuple(shift)
    affine = eye_func(spatial_dims + 1)
    for i, a in enumerate(shift[:spatial_dims]):
        affine[i, spatial_dims] = a
    return array_func(affine)  # type: ignore
