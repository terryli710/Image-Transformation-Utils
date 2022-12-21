# random utils

from typing import List, Union
import torch
from imgtrans.utils.randomize import Interval, RandomFromInterval


class RandomFromIntervalTorch(RandomFromInterval):
    
    def set_random_state(self, seed):
        torch.manual_seed(seed)
        pass
    
    def sample(self, intervals: Union[Interval, tuple, List[Union[Interval, tuple]]], nbatch=None):
        """
        Args:
            intervals: List[Interval, ...]
            nbatch: int
        Return:
            if nbatch: (B, ndim)
            if not nbatch: (ndim, )
        """
        intervals = self._construct_interval(intervals)
        if isinstance(intervals, Interval):
            intervals = [intervals]
        # convert intervals to torch tensor (ndim, 2)
        intervals = torch.tensor([[i.lower, i.upper] for i in intervals])
        if nbatch is None:
            # (ndim, )
            param = torch.rand(intervals.shape[0]) * (intervals[:, 1] - intervals[:, 0]) + intervals[:, 0]
        else:
            param = torch.rand((nbatch, intervals.shape[0])) * (intervals[:, 1] - intervals[:, 0]) + intervals[:, 0]
        
        if self.include_negative:
            # generate 1 and -1 randomly of size param
            random_negative = torch.randint(0, 2, param.shape) * 2 - 1
            param = param * random_negative
        return param
    