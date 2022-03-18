from typing import Any, Tuple
import collections


def issequenceiterable(obj: Any):
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    return isinstance(
        obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes))


def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        return (vals, )

    return tuple(vals)


def ensure_tuple_size(tup: Any, dim: int, pad_val: Any = 0):
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    new_tup = ensure_tuple(tup) + (pad_val, ) * dim
    return new_tup[:dim]