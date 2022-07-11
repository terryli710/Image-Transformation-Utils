import torch
from torch import nn
import numpy as np
from .resize import resize


class DrawPerlin(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.requires_grad_(requires_grad)
        self.requires_grad = requires_grad
        pass
    
    def forward(self, 
                out_shape,
                    scales,
                    min_std=0,
                    max_std=1,
                    dtype=torch.float32,
                    device=None,
                    seed=None):
        '''
        Generate Perlin noise by drawing from Gaussian distributions at different
        resolutions, upsampling and summing. There are a couple of key differences
        between this function and the Neurite equivalent ne.utils.perlin_vol, which
        are not straightforwardly consolidated.
        Neurite function:
            (1) Iterates over scales in range(a, b) where a, b are input arguments.
            (2) Noise volumes are sampled at resolutions vol_shape / 2**scale.
            (3) Noise volumes are sampled uniformly in the interval [0, 1].
            (4) Volume weights are {1, 2, ...N} (normalized) where N is the number
                of scales, or sampled uniformly from [0, 1].
        This function:
            (1) Specific scales are passed as a list.
            (2) Noise volumes are sampled at resolutions vol_shape / scale.
            (3) Noise volumes are sampled normally, with SDs drawn uniformly from
                [min_std, max_std].
        Parameters:
            out_shape: List defining the output shape. In N-dimensional space, it
                should have N+1 elements, the last one being the feature dimension.
            scales: List of relative resolutions at which noise is sampled normally.
                A scale of 2 means half resolution relative to the output shape.
            min_std: Minimum standard deviation (SD) for drawing noise volumes.
            max_std: Maximum SD for drawing noise volumes.
            dtype: Output data type.
            seed: Integer for reproducible randomization. 
        '''
        # out_shape = torch.tensor(out_shape, dtype=torch.int32)
        if np.isscalar(scales):
            scales = [scales]
        # set random seed, would work within the function
        np.random.seed(seed)
        out = torch.zeros(out_shape[-1], *out_shape[:-1], dtype=dtype).to(device) # channel first (C, H, W, (D))

        for scale in scales:
            scaled_shape = torch.tensor([s / scale for s in out_shape[:-1]]).to(torch.int32) # ceil is not diffientiable
            sample_shape = (*scaled_shape, out_shape[-1]) # TODO: check differianblility
            std = torch.rand([1], requires_grad=self.requires_grad, device=device) * (max_std - min_std) + min_std
            gauss = torch.randn(sample_shape, requires_grad=self.requires_grad, device=device, dtype=dtype) * std

            # zoom = [o / s for o, s in zip(out_shape, sample_shape)]
            
            # scale gauss and add to out
            if scale != 1:
                # change to channel first
                if len(sample_shape) == 3:
                    # 2d
                    gauss = gauss.permute(2, 0, 1)
                elif len(sample_shape) == 4:
                    # 3d
                    gauss = gauss.permute(3, 0, 1, 2)
                else:
                    raise NotImplementedError(f"input dimension not supported: {sample_shape=}")
                
                gauss = resize(gauss.unsqueeze(0), # create batch dim
                            target_size=out_shape[:-1])[0, ...]
            
            out = out + gauss
        
        # channel last
        if len(sample_shape) == 3:
            # 2d
            out = out.permute(1, 2, 0)
        elif len(sample_shape) == 4:
            # 3d
            out = out.permute(1, 2, 3, 0)
        else:
            raise NotImplementedError
        return out # shape = (H, W, (D), 2 or 3)
