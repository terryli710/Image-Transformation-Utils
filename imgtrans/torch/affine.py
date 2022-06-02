# affine transformation, from monai

from typing import Callable, List, Optional, Sequence, Union

from imgtrans.utils.misc import ensure_tuple, ensure_tuple_size

import torch

from .utils.grid_utils import Resample, create_grid
from .utils.randomize import RandParams
from .utils.type_utils import convert_tenor_dtype, convert_to_dst_type


class Affine:

    def __init__(
        self,
        spatial_dims=2,
        rotate: Optional[Union[Sequence[float], float]] = None,
        shear: Optional[Union[Sequence[float], float]] = None,
        translate: Optional[Union[Sequence[float], float]] = None,
        scale: Optional[Union[Sequence[float], float]] = 1.0,
        mode: str = 'bilinear',
        padding_mode: str = "reflection",
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
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        self.resampler = Resample(mode=mode,
                                  padding_mode=padding_mode,
                                  device=device)
        pass

    def __call__(
        self,
        img: torch.Tensor,
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

        # get grid
        spatial_size = img.shape[1:]
        grid = create_grid(spatial_size, device=self.device)
        grid = convert_tenor_dtype(grid, device=self.device, dtype=float)

        aff_mtx = convert_to_dst_type(aff_mtx, grid)

        grid = (aff_mtx @ grid.reshape(
            (grid.shape[0], -1))).reshape([-1] + list(grid.shape[1:]))

        # make sure that img is in the same device as grid
        img = convert_tenor_dtype(img, device=self.device)
        trans_img = self.resampler(img,
                                   grid=grid,
                                   mode=mode or self.mode,
                                   padding_mode=padding_mode
                                   or self.padding_mode)

        return trans_img, {
            "aff_mtx": aff_mtx,
            "grid": grid,
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

        # _b = "torch" if isinstance(grid, torch.Tensor) else "numpy"
        # _device = grid.device if isinstance(grid, torch.Tensor) else self.device

        affine = torch.eye(spatial_dims + 1)
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
                 padding_mode="reflection",
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
                 img: torch.Tensor,
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


def _create_translate(spatial_dims: int, shift: Union[Sequence[float], float],
                      eye_func: Callable, array_func: Callable):
    shift = ensure_tuple(shift)
    affine = eye_func(spatial_dims + 1)
    for i, a in enumerate(shift[:spatial_dims]):
        affine[i, spatial_dims] = a
    return array_func(affine)  # type: ignore
