from typing import Optional, Sequence, Tuple, Union
import torch
from .utils.grid_utils import Resample, create_control_grid
from .utils.randomize import RandParams
from imgtrans.utils.misc import fall_back_tuple, ensure_tuple
from .utils.type_utils import convert_to_dst_type
from .utils.crop import CenterSpatialCrop


class DeformElastic:
    """
    Generate deterministc deformation grid
    """

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ):  # -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.as_tensor_output = as_tensor_output
        self.device = device
        self.resampler = Resample(device=device)
        pass

    def __call__(
        self,
        img: torch.Tensor,
        magnitude,
        offset,
        spatial_size: Union[Tuple[int, int], None] = None,
        mode: Optional[str] = "nearest",
        padding_mode: Optional[str] = "reflection",
    ):
        """

        Args:
            img (torch.Tensor): shape = (C or B, H, W, (D))
            magnitude (float): magnitude range of the offset
            offset (array): range (0, 1) typically randomly generalized
            spatial_size (Union[Tuple[int, int], None], optional): image shape. Defaults to None.
            mode (Optional[str], optional): _description_. Defaults to None.
            padding_mode (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Tuple: image (C or B, H, W, (D)), {dict of param}
        """
        if not spatial_size:
            spatial_size = img.shape[1:]
        sp_size = fall_back_tuple(spatial_size, img.shape[1:])
        self.spacing = fall_back_tuple(self.spacing,
                                       (1.0, ) * len(spatial_size))
        control_grid = create_control_grid(
            spatial_size,
            self.spacing,
            device=self.device,
        )
        _offset, *_ = convert_to_dst_type(magnitude * offset, control_grid)
        control_grid[:len(spatial_size)] += _offset

        assert isinstance(control_grid, torch.Tensor)
        grid = torch.nn.functional.interpolate(  # type: ignore
            recompute_scale_factor=True,
            input=control_grid.unsqueeze(0),
            scale_factor=list(ensure_tuple(self.spacing)),
            mode=mode if not mode is None else "bilinear",
            align_corners=False,
        )[0]

        grid = CenterSpatialCrop(roi_size=sp_size)(grid)
        # center crop
        grid = grid[:,]
        print(grid.shape)
        ret, real_grid = self.resampler(
            img,
            grid,
            mode=mode,
            padding_mode=padding_mode,
        )
        return ret, {"flow_grid": real_grid}


class RandDeformElastic(DeformElastic, RandParams):
    """
    Perform random elastic transformation
    """

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        magnitude_range: Tuple[float, float],
        device: Optional[torch.device] = None,
    ):  # -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D. distance in between the control points.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range
        self.device = device
        self.deform_elastic = DeformElastic(spacing=self.spacing)
        pass

    def randomize(self, grid_size: Sequence[int], seed=None):  # -> None:
        if seed:
            self.set_random_state(seed)
        random_offset = self.get_gaussian(size=([len(grid_size)] + list(grid_size)))
        rand_mag = self.get_randparam(self.magnitude, dim=1, abs=True)[0]
        params = {"offset": random_offset, "magnitude": rand_mag}
        return params

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Union[Tuple[int, int], None] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        seed=None,
    ):
        """

        Args:
            img (torch.Tensor): shape = (C or B, H, W, (D))
            spatial_size (Union[Tuple[int, int], None], array): size of image (H, W, (D)). Defaults to None. If None, use image.shape[1:]
            mode (Optional[str], optional): mode for torch.nn.functional.sample_grid. Defaults to None.
            padding_mode (Optional[str], optional): for torch.nn.functional.sample_grid. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.

        Returns:
            tuple: image, {dict of params}
        """
        if not spatial_size:
            spatial_size = img.shape[1:]
        self.spacing = fall_back_tuple(self.spacing,
                                       (1.0, ) * len(spatial_size))
        control_grid = create_control_grid(spatial_size,
                                   self.spacing,
                                   device=self.device)
        rand_params = self.randomize(control_grid.shape[1:], seed=seed)
        ret, params = self.deform_elastic(img=img,
                                             spatial_size=spatial_size,
                                             mode=mode,
                                             padding_mode=padding_mode,
                                             **rand_params)
        params.update({"rand_params":rand_params})
        return ret, params
