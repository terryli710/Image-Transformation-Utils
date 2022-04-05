from typing import Union, Sequence, Optional, Tuple
import numpy as np


class Elastic: # TODO: have to find an alternative to pytorch's grid_sample method.
    """
    elastic transformation with backend to be numpy
    """

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        as_tensor_output: bool = True,
        image_only=False,
    ):
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

        self.resampler = Resample(device=device)
        self.image_only = image_only
        pass

    def __call__(
        self,
        img: np.ndarray,
        magitude,
        offset,
        spatial_size: Union[Tuple[int, int], None] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        if not spatial_size:
            spatial_size = img.shape[1:]
        sp_size = fall_back_tuple(spatial_size, img.shape[1:])
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(
            spatial_size, self.spacing, device=self.device, backend=TransformBackends.TORCH
        )
        _offset, *_ = convert_to_dst_type(magitude * offset, control_grid)
        control_grid[: len(spatial_size)] += _offset
        if not self.as_tensor_output:
            control_grid, *_ = convert_data_type(
                control_grid, output_type=np.ndarray, dtype=np.float32
            )
        assert isinstance(control_grid, torch.Tensor)
        grid = torch.nn.functional.interpolate(  # type: ignore
            recompute_scale_factor=True,
            input=control_grid.unsqueeze(0),
            scale_factor=list(ensure_tuple(self.spacing)),
            mode=InterpolateMode.BICUBIC.value,
            align_corners=False,
        )
        grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])

        ret, real_grid = self.resampler(
            img,
            grid,
            mode=mode,
            padding_mode=padding_mode,
        )
        return ret if self.image_only else (ret, real_grid)


class RandDeformElastic(Randomizable, Transform):
    """
    Generate random deformation grid.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        magnitude_range: Tuple[float, float],
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
        self.magnitude = magnitude_range

        self.as_tensor_output = as_tensor_output
        self.device = device
        self.deform_elastic = DeformElastic(spacing=self.spacing)
        pass

    def randomize(self, grid_size: Sequence[int], seed=None):  # -> None:
        if seed:
            self.R.seed(seed=seed)
        random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(
            np.float32
        )
        rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])
        params = {"offset": random_offset, "mag": rand_mag}
        return params

    def __call__(
        self,
        img: NdarrayOrTensor,
        spatial_size: Union[Tuple[int, int], None] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        seed=None,
    ):
        """
        Args:
            spatial_size: spatial size of the grid.
        import matplotlib.pyplot as plt
        plt.imsave("/home/yiheng/temp5.jpeg", ret[6,:,:])
        """
        if not spatial_size:
            spatial_size = img.shape[1:]
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(
            spatial_size, self.spacing, device=self.device, backend=TransformBackends.TORCH
        )
        rand_params = self.randomize(control_grid.shape[1:], seed=seed)
        ret, real_grid = self.deform_elastic(
            img=img,
            spatial_size=spatial_size,
            mode=mode,
            padding_mode=padding_mode,
            **rand_params
        )
        return (ret, rand_params, real_grid)