from typing import Sequence, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GuassianFilter(nn.Module):
    """
    Perform Gaussian blur to an input tensor (2d or 3d) with an gaussian kernel.
    """
    def __init__(self, 
                 ndim: int, 
                 sigma: Union[Sequence[float], float],
                 nchannel: int = 1,
                 kernel_size: int = 5,
                 requires_grad=False, 
                 device=None):
        super().__init__()
        self.requires_grad_(requires_grad)
        self.requires_grad = requires_grad
        self.ndim = ndim
        self.nchannel = nchannel
        self.device = device
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._create_gaussian_kernel()
        # assert: if sigma is sequence, then len(sigma) == ndim
        assert isinstance(self.sigma, (float, int)) or len(self.sigma) == self.ndim, f"sigma is {self.sigma}, ndim is {self.ndim}"
        pass
        
    
    def _create_gaussian_kernel(self):
        """
        create a 2d or 3d gaussian kernel,
        Returns:
            gaussian_kernel: (C, C, *[kernel_size] * ndim)
        """
        if isinstance(self.sigma, (float, int)):
            sigma = [self.sigma] * self.ndim
        elif isinstance(self.sigma, Sequence):
            assert len(self.sigma) == self.ndim
        else:
            raise TypeError(f"sigma should be number or sequence, got {type(self.sigma)}")
        
        # create kernel
        kernel_size = self.kernel_size
        kernel = np.zeros([kernel_size] * self.ndim)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                if self.ndim == 2:
                    kernel[i, j] = np.exp(-0.5 * ((i - center) ** 2 + (j - center) ** 2) / sigma[0] ** 2)
                elif self.ndim == 3:
                    for k in range(kernel_size):
                        kernel[i, j, k] = np.exp(-0.5 * ((i - center) ** 2 + (j - center) ** 2 + (k - center) ** 2) / sigma[0] ** 2)
                else:
                    raise NotImplementedError(f"ndim should be 2 or 3, got {self.ndim}")
        # normalize
        kernel = kernel / np.sum(kernel)
        # convert to tensor
        kernel = torch.tensor(kernel, dtype=torch.float32, device=self.device)
        # expand to ndim
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(self.nchannel, self.nchannel, *([kernel_size] * self.ndim))
        return kernel
        
    
    def forward(self, x):
        """
        x: (..., C, H, W, (D)), the batch dims are arbitrary
        weights: (C, C, kH, kW, (kD)), the in and out channels are the same
        """
        # reshape x to (N, C, H, W, (D))
        batch_dims = x.shape[:-self.ndim - 1]
        x_shape = x.shape
        if batch_dims.numel() == 0: # number of elements is zero
            batch_dims = [1]
        else:
            batch_dims = [torch.prod(torch.tensor(batch_dims)).item()]
        x = x.reshape(*batch_dims, *x.shape[-self.ndim - 1:])
        
        if self.ndim == 2:
            x = F.conv2d(x, self.gaussian_kernel, padding="same")
        elif self.ndim == 3:
            x = F.conv3d(x, self.gaussian_kernel, padding="same")
        else:
            raise NotImplementedError(f"ndim should be 2 or 3, got {self.ndim}")
        
        # reshape x to original shape
        x = x.reshape(*x_shape)
        return x
        
        