# random utils
from typing import Optional, Any
from abc import ABC
import numpy as np


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

    R: np.random.RandomState = np.random.RandomState()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed: set the random state with an integer seed.
            state: set the random state with a `np.random.RandomState` object.

        Raises:
            TypeError: When ``state`` is not an ``Optional[np.random.RandomState]``.

        Returns:
            a Randomizable instance.

        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            self.R = np.random.RandomState(_seed)
            return self

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self


    def randomize(self, data: Any):
        """
        Within this method, :py:attr:`self.R` should be used, instead of `np.random`, to introduce random factors.

        all :py:attr:`self.R` calls happen here so that we have a better chance to
        identify errors of sync the random state.

        This method can generate the random factors based on properties of the input data.

        Raises:
            NotImplementedError: When the subclass does not override this method.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


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
        out_param = []
        for i in range(dim):
            param = self.R.uniform(param_range[0], param_range[1])
            if not abs:
                param = param * int(self.R.choice([-1, 1]))
            out_param.append(param)
        return out_param