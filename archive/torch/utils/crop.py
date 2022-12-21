from typing import Union, Sequence, Optional
import torch
from imgtrans.utils.misc import fall_back_tuple
from ....imgtrans.torch.utils.type_utils import convert_tenor_dtype, convert_to_dst_type


class CenterSpatialCrop:
    """
    Crop at the center of image with specified ROI size.
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
    """

    def __init__(self, roi_size: Union[Sequence[int], int]):
        self.roi_size = roi_size

    def __call__(self, img: torch.Tensor):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        center = [i // 2 for i in img.shape[1:]]
        print(center, roi_size)
        cropper = SpatialCrop(roi_center=center, roi_size=roi_size)
        return cropper(img)
    
    
class SpatialCrop:
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    def __init__(
        self,
        roi_center: Union[Sequence[int], torch.Tensor, None] = None,
        roi_size: Union[Sequence[int], torch.Tensor, None] = None,
        roi_start: Union[Sequence[int], torch.Tensor, None] = None,
        roi_end: Union[Sequence[int], torch.Tensor, None] = None,
        roi_slices: Optional[Sequence[slice]] = None,
    ):
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        roi_start_torch: torch.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError("Only slice steps of 1/None are currently supported")
            self.slices = list(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                # roi_center, *_ = convert_tenor_dtype(
                #     data=roi_center, dtype=torch.int16, wrap_sequence=True
                # )
                # roi_size, *_ = convert_to_dst_type(src=roi_size, dst=roi_center, wrap_sequence=True)
                roi_center = torch.tensor(roi_center).type(torch.int16)
                roi_size = torch.tensor(roi_size).type(torch.int16)
                _zeros = torch.zeros_like(roi_center)  # type: ignore
                roi_start_torch = torch.max(roi_center - torch.div(roi_size, 2, rounding_mode='floor'), _zeros)  # type: ignore
                roi_end_torch = torch.max(roi_start_torch + roi_size, roi_start_torch)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_torch, *_ = convert_tenor_dtype(  # type: ignore
                    data=roi_start, dtype=torch.int16, wrap_sequence=True
                )
                roi_start_torch = torch.max(roi_start_torch, torch.zeros_like(roi_start_torch))  # type: ignore
                roi_end_torch, *_ = convert_to_dst_type(src=roi_end, dst=roi_start_torch, wrap_sequence=True)
                roi_end_torch = torch.max(roi_end_torch, roi_start_torch)
            # convert to slices (accounting for 1d)
            if roi_start_torch.numel() == 1:
                self.slices = [slice(int(roi_start_torch.item()), int(roi_end_torch.item()))]
            else:
                self.slices = [slice(int(s), int(e)) for s, e in zip(roi_start_torch.tolist(), roi_end_torch.tolist())]


    def __call__(self, img: torch.Tensor):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]
        print(self.slices, slices)
        return img[tuple(slices)]