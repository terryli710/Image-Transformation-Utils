import warnings

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from .spatial import is_affine_shape, affine_to_dense_shift, volshape_to_meshgrid, interpn


class SpatialTransformer(Layer):
    """
    ND spatial transformer layer

    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    I dopted the code from voxelmorph.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 **kwargs):
        """
        Parameters: 
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            indexing: Must be 'ij' (matrix) or 'xy' (cartesian). 'xy' indexing will
                have the first two entries of the flow (along last axis) flipped
                compared to 'ij' indexing.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms.
        """
        self.interp_method = interp_method
        assert indexing in [
            'ij', 'xy'
        ], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError(
                'Spatial Transformer must be called on a list of length 2: '
                'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        self.is_affine = is_affine_shape(input_shape[1][1:])

        # make sure inputs are reasonable shapes
        if self.is_affine:
            expected = (self.ndims, self.ndims + 1)
            actual = tuple(self.trfshape[-2:])
            if expected != actual:
                raise ValueError(
                    f'Expected {expected} affine matrix, got {actual}.')
        else:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(
                    f'Dense transform shape {dense_shape} does not match '
                    f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # convert affine matrix to warp field
        if self.is_affine:
            fun = lambda x: affine_to_dense_shift(x,
                                                  vol.shape[1:-1],
                                                  shift_center=self.
                                                  shift_center,
                                                  indexing=self.indexing)
            trf = tf.map_fn(fun, trf)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]),
                             vol), trf
        else:
            return tf.map_fn(self._single_transform, [vol, trf],
                             fn_output_signature=vol.dtype), trf

    def _single_transform(self, inputs):
        return transform(inputs[0],
                         inputs[1],
                         interp_method=self.interp_method,
                         fill_value=self.fill_value)


def transform(vol,
              loc_shift,
              interp_method='linear',
              indexing='ij',
              fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    loc = [
        tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)
    ]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(tf.cast(mesh[-1], 'float32'))

    # test single
    return interpn(vol,
                   loc,
                   interp_method=interp_method,
                   fill_value=fill_value)


class WarpSpatialTransformer(Layer):
    """
    ND spatial transformer layer

    Applies dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    I dopted the code from voxelmorph.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 **kwargs):
        """
        Parameters: 
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            indexing: Must be 'ij' (matrix) or 'xy' (cartesian). 'xy' indexing will
                have the first two entries of the flow (along last axis) flipped
                compared to 'ij' indexing.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms.
        """
        self.interp_method = interp_method
        assert indexing in [
            'ij', 'xy'
        ], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        super().__init__(**kwargs)
        pass

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError(
                'Spatial Transformer must be called on a list of length 2: '
                'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]+[self.ndims]
        self.is_affine = is_affine_shape(input_shape[1][1:])

        # make sure inputs are reasonable shapes
        image_shape = tuple(self.imshape[:-1])
        dense_shape = tuple(self.trfshape[:-1])
        if image_shape != dense_shape:
            warnings.warn(
                f'Dense transform shape {dense_shape} does not match '
                f'image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """
        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))
        print(f"in warp: {trf.shape=}, {vol.shape=}")
        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            return tf.map_fn(lambda x: self._single_transform([x, trf[0, :]]),
                             vol), trf
        else:
            return tf.map_fn(self._single_transform, [vol, trf],
                             fn_output_signature=vol.dtype), trf

    def _single_transform(self, inputs):
        return transform(inputs[0],
                         inputs[1],
                         interp_method=self.interp_method,
                         fill_value=self.fill_value), 


class AffineSpatialTransformer(WarpSpatialTransformer):

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 **kwargs):

        super(AffineSpatialTransformer, self).__init__(interp_method=interp_method,
                         indexing=indexing,
                         single_transform=single_transform,
                         fill_value=fill_value,
                         shift_center=shift_center,
                         **kwargs)

        pass

    def build(self, input_shape):
        # sanity check on input list
        assert len(
            input_shape
        ) <= 2, 'Spatial Transformer must be called on a list of length 2: first argument is the image, second is the transform.'
        # sanity check on affine shape
        assert is_affine_shape(
            input_shape[1][1:]
        ), f"The input matrix is not of the shape of an affine matrix: {input_shape[1][1:]=}"

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.afmshape = input_shape[1][1:]
        self.trfshape =  input_shape[0][1:-1]+[self.ndims] # using image as the reference for transform shape so made some modifications

        # make sure inputs are reasonable shapes
        expected = (self.ndims, self.ndims + 1)
        actual = tuple(self.afmshape[-2:])
        if expected != actual:
            raise ValueError(
                f'Expected {expected} affine matrix, got {actual}.')

        # confirm built
        self.built = True
        pass

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is an affine matrix of shape [B, N, N+1].
        """
        # necessary for multi-gpu models
        print(f"{inputs[0].shape=}, {inputs[1].shape=}")
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.afmshape))

        # convert affine matrix to warp field
        fun = lambda x: affine_to_dense_shift(x,
                                              vol.shape[1:-1],
                                              shift_center=self.shift_center,
                                              indexing=self.indexing)
        trf = tf.map_fn(fun, trf)
        print(f"{trf.shape=}, {vol.shape=}, {self.imshape=}, {self.afmshape=}, {self.trfshape=}")
        return super().call([vol, trf])
        
