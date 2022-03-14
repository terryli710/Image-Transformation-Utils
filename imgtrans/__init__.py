# import backend
from .utils import get_backend

backend = get_backend()

if backend == "tensorflow":
    try:
        import tensorflow
    except ImportError:
        raise ImportError('cannot import backend tensorflow')

    from .np import affine

elif backend == "pytorch":
    try:
        import torch
    except torch:
        raise ImportError('cannot import backend torch')

    from .torch import affine

else:
    from .np import affine