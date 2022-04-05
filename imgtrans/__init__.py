# import backend
from .utils.backend import get_backend

backend = get_backend()

if backend == "tensorflow":
    try:
        import tensorflow
    except ImportError:
        raise ImportError('cannot import backend tensorflow')

    from .np import affine, elastic

elif backend == "pytorch":
    try:
        import torch
    except torch:
        raise ImportError('cannot import backend torch')

    from .torch import affine, elastic, svf, perlin

else:
    from .np import affine