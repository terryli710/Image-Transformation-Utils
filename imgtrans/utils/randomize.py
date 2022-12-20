
from abc import ABC, abstractmethod
from typing import Union
from dataclasses import dataclass

from imgtrans.utils.type_utils import is_array_like


@dataclass
class Interval:
    """
    A class representing an interval.
    """
    low: float
    high: float

    def __post_init__(self):
        assert self.low <= self.high, \
            f"low must be less than or equal to high, got {self.low} and {self.high}"

    def __contains__(self, item):
        return self.low <= item <= self.high

    def __repr__(self):
        return f"Interval(low={self.low}, high={self.high})"


class RandomFromInterval(ABC):
    """
    base class for generating random numbers, 
    could have different backends, e.g. numpy, torch, etc.
    """
    def __init__(self, seed=None, include_negative=False):
        self.seed = seed
        if self.seed:
            self.set_random_state(self.seed)
        self.include_negative = include_negative
        pass

    @abstractmethod
    def set_random_state(self, seed):
        pass

    def _process_param(self, param: Union[float, int, tuple, list]):
        """
        Input:
            param: a float, int, tuple, or list
        Examples:
            1. 1.0 -> (0.0, 1.0)
            2. (1.0, 2.0) -> (1.0, 2.0)
            3. [1.0, 2.0] other array like -> [(0.0, 1.0), (0.0, 2.0)]
            4. [(1.0, 2.0), (3.0, 4.0)] -> [(1.0, 2.0), (3.0, 4.0)]
        Return:
            eligible params that has Intervals
            [Interval, ...]
        """
        # if not array like
        if not is_array_like(param):
            return [Interval(0.0, param)]
        # if array like
        else:
            # if tuple
            if isinstance(param, tuple) and len(param) == 2:
                return [Interval(*param)]
            # if other array like
            else:
                return [Interval(0.0, p) if not is_array_like(p) else Interval(*p) for p in param]
            
            
    def sample(self, interval: Interval, nbatch=None):
        """
        Remember self.include_negative
        """
        return 0.0
    
        
    def get_randparams(self, param, nbatch=None, dim=None):
        """
        Process the input param to a range of valid values
        and generate parameters within the range.
        Args:
            param: a float, int, tuple, or list
                -> [Interval, ..., ]
            nbatch: number of batches
            dim: number of dimensions, only used when param doesn't not contain multiple length
        Return:
            numpy array or torch tensor
            of shape (nbatch, dim) or (nbatch, ) or (dim, )
            or float, torch tensor value
        """
        param = self._process_param(param)
        if dim is not None and len(param) == 1:
            param = param * dim
        return self.sample(param, nbatch)
        
        
        
        
class TransformWithRandom(ABC):
    """
    An abstract class for all transformation classes,
    Used to sample random parameters for transformations.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    
        
        
        
        
        
        