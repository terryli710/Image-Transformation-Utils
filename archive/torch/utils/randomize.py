# random utils
from typing import Optional, Any
from abc import ABC
import numpy as np
import torch


class Randomizable(ABC):
    """
    An interface for handling random state locally, currently based on a class
    variable `R`, which is an instance of `np.random.RandomState`.  This
    provides the flexibility of component-specific determinism without
    affecting the global states.  It is recommended to use this API with
    :py:class:`monai.data.DataLoader` for deterministic behaviour of the
    preprocessing pipelines. This API is not thread-safe. Additionally,
    deepcopying instance of this class often causes insufficient randomness as
    the random states will be duplicated.
    """

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        """
        set random seed for pytorch
        """
        torch.manual_seed(seed)
        pass


class RandParams(Randomizable):
    """
    Generate random params
    """

    def __init__(self, seed=None):  # -> None:
        super().__init__()
        if seed:
            self.set_random_state(seed=seed)
        pass

    def get_randparam(self, param_range, dim=2, abs=False):
        # typically, param_range = (lowerbound, upperbound), while the range includes
        # [-ub, -lb] U [lb, ub];
        # abs: if set to True, then the param_range won't include their negative sides
        # if param_range = None then return None
        out_param = []
        # if the upperbound is equal to the lowerbound, then return the value
        if param_range[1] == param_range[0]:
            return [param_range[0]] * dim
        for i in range(dim):
            param = torch.rand(size=(1,))[0] * (param_range[1] - param_range[0]) + param_range[0]
            if not abs:
                param = param * bool(torch.rand(size=(1,)) > 0.5)
            out_param.append(param)
        return out_param
    
    def get_gaussian(self, size):
        return torch.randn(size=size)