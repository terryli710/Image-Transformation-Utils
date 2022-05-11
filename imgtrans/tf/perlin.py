import tensorflow as tf
import numpy as np
from .utils.transform import SpatialTransformer
from .utils.resize import resize_channel_last

class RandPerlin:
    """
    TODO: need to clean this up
    random perlin transformations (use perlin noise as deformation field)
    """
    
    def __init__(self, scales=(32, 64), min_std=0, max_std=1, interp_method='linear', indexing='ij', fill_value=None):
        self.scales = scales
        self.min_std = min_std
        self.max_std = max_std
        self.interp_method = interp_method
        self.indexing = indexing
        self.fill_value = fill_value
        self.spatial_transformer = SpatialTransformer(interp_method=interp_method, 
                        indexing=indexing, 
                        fill_value=fill_value)
        self.default_in_shape = None
        pass
    
    def __call__(self, 
                 img, 
                 out_shape=None, 
                 mode="nearest", 
                 padding_mode="reflection", 
                 dtype=None, 
                 seed=None):
        """
        img  = (C or B, H, W, (D))
        """
        if not out_shape:
            out_shape = img.shape[1:]
        if not dtype:
            dtype = img.dtype
        else:
            img = tf.cast(img, dtype)

        ndim = len(out_shape)
        
        perlin_warp = draw_perlin(out_shape=(*out_shape, ndim),
                                  scales=self.scales,
                                  min_std=self.min_std,
                                  max_std=self.max_std,
                                  dtype=dtype,
                                  seed=seed)

        if self.default_in_shape is None:
            self.default_in_shape = img.shape[1:]
        if img.shape[1:] != self.default_in_shape:
            sptrans = SpatialTransformer(interp_method=self.interp_method, 
                        indexing=self.indexing, 
                        fill_value=self.fill_value)
            deformed_img = sptrans([img[..., None], perlin_warp[None, ...]])[..., 0]
        else:
            # use SpatialTransformer to warp the image
            # NOTE: perlin warp is a DVF that denotes the percentage of displacement
            deformed_img = self.spatial_transformer([img[..., None], perlin_warp[None, ...]])[..., 0]

        return deformed_img, {"dvf": perlin_warp}
        


def draw_perlin(out_shape, scales, min_std=0, max_std=1, modulate=True, dtype=tf.float32, seed=None):
    '''Generate Perlin noise by drawing from Gaussian distributions at different
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
        (3) Noise volumes are sampled normally, using SD 1.
        (4) Volume weights are all max_sigma, or sampled uniformly from
            [0, max_sigma], where max_sigma is an input argument.

    Parameters:
        out_shape: List defining the output shape. In N-dimensional space, it
            should have N+1 elements, the last one being the feature dimension.
        scales: List of relative resolutions at which noise is sampled normally.
            A scale 2 means half resolution relative to the output shape.
        max_std: Maximum standard deviation (SD) for drawing noise volumes.
        modulate: Whether the SD for each scale is drawn from [0, max_std].
        dtype: Output data type.
        seed: Integer for reproducible randomization. This may only have an
            effect if the function is wrapped in a Lambda layer.
    '''
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max)
    out = tf.zeros(out_shape, dtype=dtype)
    for scale in scales:
        sample_shape = np.ceil(out_shape[:-1] / scale).astype(int)
        sample_shape = (*sample_shape, out_shape[-1])

        std = max_std
        if modulate:
            std = tf.random.uniform((1,), minval=min_std, maxval=max_std, dtype=dtype, seed=seed())
        gauss = tf.random.normal(sample_shape, stddev=std, dtype=dtype, seed=seed())
        if scale != 1:
            gauss = resize_channel_last(image=gauss[None, ...], target_size=out_shape[:-1])[0, ...]
        out += gauss

    return out


