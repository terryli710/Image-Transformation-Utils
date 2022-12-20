# random utils

import torch
from imgtrans.utils.randomize import RandomFromInterval


class RandomFromIntervalTorch(RandomFromInterval):
    
    def set_random_state(self, seed):
        torch.manual_seed(seed)
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
        interval = torch.tensor([[i.lower, i.upper] for i in interval])
        if nbatch is None:
            # (ndim, )
            param = torch.rand(interval.shape[0]) * (interval[:, 1] - interval[:, 0]) + interval[:, 0]
        else:
            param = torch.rand((nbatch, interval.shape[0])) * (interval[:, 1] - interval[:, 0]) + interval[:, 0]
        
        if self.include_negative:
            # generate 1 and -1 randomly of size param
            random_negative = torch.randint(0, 2, param.shape) * 2 - 1
            param = param * random_negative
        return param
    