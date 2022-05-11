# add resize function for numpy
from skimage.transform import resize

 
def resize(image, target_size):
    """
    resize the image to target size using scikit-image
    """
    return resize(image, target_size, mode='constant', preserve_range=True)
