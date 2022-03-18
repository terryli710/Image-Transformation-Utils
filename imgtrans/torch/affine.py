# affine transformation, from monai

import torch
from typing import Union, Sequence, Optional, Callable, List
from imgtrans.utils.misc import ensure_tuple, ensure_tuple_size
from torch.nn.functional import grid_sample
from .utils.grid_utils import create_grid
from .utils.type_utils import convert_tenor_dtype, convert_to_dst_type


class Affine:

    def __init__(
        self,
        spatial_dims=None,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = 1.0,
        mode: str = 'bilinear',
        padding_mode: str = "reflect",
        device: Optional[str] = None,
        image_only: bool = False,
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
            spatial_dims=2,
            rotate=rotate_params,
            shear=shear_params,
            translate=translate_params,
            scale=scale_params,
        )
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device
        self.resampler = Resample(mode=mode,
                                  padding_mode=padding_mode,
                                  device=device)
        self.image_only = image_only
        pass

    def __call__(
        self,
        img: torch.Tensor,
        spatial_dims=None,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        # create aff_mtx
        if any([
                p is not None for p in
            [rotate_params, shear_params, translate_params, scale_params]
        ]):

            aff_mtx = self._get_aff_mtx(
                spatial_dims=spatial_dims
                if spatial_dims is not None else len(img.shape),
                rotate=rotate_params
                if rotate_params is not None else self.rotate_params,
                shear=shear_params
                if shear_params is not None else self.shear_params,
                translate=translate_params
                if translate_params is not None else self.translate_params,
                scale=scale_params
                if scale_params is not None else self.scale_params,
            )
        else:
            aff_mtx = self.aff_mtx
        # get grid
        spatial_size = img.shape
        grid = create_grid(spatial_size, device=self.device)
        grid, *_ = convert_tenor_dtype(grid,
                                       torch.Tensor,
                                       device=self.device,
                                       dtype=float)
        affine, *_ = convert_to_dst_type(aff_mtx, grid)
        grid = (aff_mtx @ grid.reshape(
            (grid.shape[0], -1))).reshape([-1] + list(grid.shape[1:]))

        trans_img = self.resampler(img,
                                   grid=grid,
                                   mode=mode or self.mode,
                                   padding_mode=padding_mode
                                   or self.padding_mode)

        return trans_img if self.image_only else trans_img, {
            "aff_mtx": aff_mtx,
            "grid": grid
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


class Resample:

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        device: Optional[torch.device] = None,
    ):
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device: device on which the tensor will be allocated.
        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device

    def __call__(
        self,
        img: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):  # -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        assert isinstance(
            img, torch.Tensor), "input img must be supplied as a torch.Tensor"
        assert isinstance(
            grid, torch.Tensor
        ), "Error, grid argument must be supplied as a torch.Tensor"

        grid = (torch.tensor(grid) if not isinstance(grid, torch.Tensor) else
                grid.detach().clone())
        if self.device:
            img = img.to(self.device)
            grid = grid.to(self.device)

        for i, dim in enumerate(img.shape[1:]):
            grid[i] = 2.0 * grid[i] / (dim - 1.0)
        grid = grid[:-1] / grid[-1:]
        index_ordering: List[int] = list(range(img.ndimension() - 2, -1, -1))
        grid = grid[index_ordering]
        grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
        out = grid_sample(
            img.unsqueeze(0).float(),
            grid.unsqueeze(0).float(),
            mode=self.mode.value if mode is None else mode,
            padding_mode=self.padding_mode.value
            if padding_mode is None else padding_mode,
            align_corners=
            False,  # NOTE: guess not much difference: https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        )[0]
        return torch.as_tensor(out)


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
    eye_func: Callable,):
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
