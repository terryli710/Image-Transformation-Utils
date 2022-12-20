

# random utils
import numpy as np
import torch
from imgtrans.utils.randomize import RandomFromInterval


class RandomFromIntervalNumpy(RandomFromInterval):
    
    def set_random_state(self, seed):
        np.random.seed(seed)
        pass
    
    def sample(self, interval, nbatch=None):
        """
        Args:
            interval: List[Interval, ...]
            nbatch: int
        Return:
            if nbatch: (B, ndim)
            if not nbatch: (ndim, )
        """
        # convert intervals to torch tensor (ndim, 2)
        interval = np.array([[i.lower, i.upper] for i in interval])
        if nbatch is None:
            # (ndim, )
            param = np.random.rand(interval.shape[0]) * (interval[:, 1] - interval[:, 0]) + interval[:, 0]
        else:
            param = np.random.rand((nbatch, interval.shape[0])) * (interval[:, 1] - interval[:, 0]) + interval[:, 0]
        
        if self.include_negative:
            # generate 1 and -1 randomly of size param
            random_negative = np.random.randint(0, 2, param.shape) * 2 - 1
            param = param * random_negative
        return param