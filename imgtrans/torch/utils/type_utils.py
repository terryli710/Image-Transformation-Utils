from typing import Any, Optional
import torch


def convert_tenor_dtype(data,
                      dtype: Optional[torch.dtype] = None,
                      device: Optional[torch.device] = None,
                      wrap_sequence: bool = False):
    """
    Utility to convert the input data to a PyTorch Tensor. If passing a dictionary, list or tuple,
    recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype,
                       device=device,
                       memory_format=torch.contiguous_format)  # type: ignore
    if isinstance(data, list):
        list_ret = [
            convert_tenor_dtype(i, dtype=dtype, device=device) for i in data
        ]
        return torch.as_tensor(
            list_ret, dtype=dtype,
            device=device) if wrap_sequence else list_ret  # type: ignore
    elif isinstance(data, tuple):
        tuple_ret = tuple(
            convert_tenor_dtype(i, dtype=dtype, device=device) for i in data)
        return torch.as_tensor(
            tuple_ret, dtype=dtype,
            device=device) if wrap_sequence else tuple_ret  # type: ignore
    elif isinstance(data, dict):
        return {
            k: convert_tenor_dtype(v, dtype=dtype, device=device)
            for k, v in data.items()
        }

    return data


def convert_to_dst_type(src: Any,
                        dst: torch.Tensor,
                        dtype: Optional[torch.dtype] = None,
                        wrap_sequence: bool = False):
    """
    Convert source data to the same data type and device as the destination data.
    If `dst` is an instance of `torch.Tensor` or its subclass, convert `src` to `torch.Tensor` with the same data type as `dst`,
    if `dst` is an instance of `numpy.ndarray` or its subclass, convert to `numpy.ndarray` with the same data type as `dst`,
    otherwise, convert to the type of `dst` directly.

    Args:
        src: source data to convert type.
        dst: destination data that convert to the same data type as it.
        dtype: an optional argument if the target `dtype` is different from the original `dst`'s data type.
        wrap_sequence: if `False`, then lists will recursively call this function. E.g., `[1, 2]` -> `[array(1), array(2)]`.
            If `True`, then `[1, 2]` -> `array([1, 2])`.

    See Also:
        :func:`convert_data_type`
    """
    device = dst.device if isinstance(dst, torch.Tensor) else None
    if dtype is None:
        dtype = dst.dtype

    return convert_tenor_dtype(data=src,
                             device=device,
                             dtype=dtype,
                             wrap_sequence=wrap_sequence)
