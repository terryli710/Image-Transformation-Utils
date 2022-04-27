import tensorflow as tf
from .spatial import ndgrid, interpn


def _resize(vol, zoom_factor, interp_method='linear'):
    """
    adopted from voxelmorph.utils
    vol = (H, W, (D), C)
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of 
        length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    If you find this function useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims
    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [
        tf.linspace(0., vol_shape[d] - 1., new_shape[d]) for d in range(ndims)
    ]
    grid = ndgrid(*lin)

    return interpn(vol, grid, interp_method=interp_method)


def resize(image, target_size, interp_method='linear'):
    """ 2D + 3D tensorflow image resize method 

    Args:
        image (tf.tensor): [N, C, H, W(, D)]
        target_size (array): tuple indicating (H, W(, D))
    """

    assert len(image.shape) == len(
        target_size
    ) + 2, f"image shape and target_size must not match, image_shape={image.shape}, target_size={target_size}"
    target_size = [int(ts) for ts in target_size]  # convert to int
    # compress the first two dims and put channel last
    imgshape = image.shape
    image = tf.reshape(image,
                       (-1, *imgshape[2:]))  # combine the first two dimensions
    image = tf.transpose(image, perm=list(range(1,
                                                len(imgshape) - 1)) +
                         [0])  # put channel last

    # resize
    image = _resize(
        image,
        zoom_factor=[o / i for o, i in zip(target_size, imgshape[2:])],
        interp_method=interp_method
        )

    # put channel first and expand the first two dims
    image = tf.transpose(image,
                         perm=[len(imgshape) - 2] +
                         list(range(0,
                                    len(imgshape) - 2)))
    image = tf.reshape(image, list(imgshape[:2]) + target_size)

    return image


def resize_channel_last(image, target_size, interp_method='linear'):
    """
    Resize image to target_size, channel last
    image = (N, H, W, (D), C)
    target_size = (H, W, (D))
    """
    # 3. scale the whole flow_grid
    if image.shape[1:-1] != target_size:
        # change to channel first
        if len(image.shape) == 4:
            # 2d
            image = tf.transpose(image, perm=[0, 3, 1, 2])
        elif len(image.shape) == 5:
            # 3d
            image = tf.transpose(image, perm=(0, 4, 1, 2, 3))
        else:
            raise NotImplementedError

        image = resize(image, target_size=target_size, interp_method="linear")

        # back to channel last
        if len(image.shape) == 4:
            # 2d
            image = tf.transpose(image, perm=(0, 2, 3, 1))
        elif len(image.shape) == 5:
            # 3d
            image = tf.transpose(image, perm=(0, 2, 3, 4, 1))
        else:
            raise NotImplementedError
    return image