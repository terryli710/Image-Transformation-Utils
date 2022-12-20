## Image Transformation Utils

A collection of 2D/3D image random/fixed transformation functions in `NumPy`, `TensorFlow` and `PyTorch` backends.

Author: Yiheng Li

## Installation

```shell
git clone https://github.com/terryli710/Image-Transformation-Utils
pip install -e .
```

## Usage

Specify backend for the package with the environmental variable: `"IMG_TRANS_BACKEND"`;

```shell
os.environ["IMG_TRANS_BACKEND"] = "pytorch" # "tf" or "numpy"
```

### Example Usage: Affine Transformation
```
import imgtrans as imt
rand_affine_trans = imt.affine.RandAffine()
transformed_img, params = rand_affine_trans(img)
```

## Available Functionalities

### Affine Transformation

- Affine matrix generator
    - 2D, 3D
    - Random, not random
    - Random batch
- Image affine transformation
    - Input image + matrix
    - output transformed image
- Random affine image transformer
    - Input image
    - Output transformed image + randomly generated matricies

### Deformable Transformation

- Spatial transformer
    - 2D, 3D
    - Input image + DVF
    - Output transformed image
    - Interp mode support
- DVF generator
    - Random, not random
    - batch
- Convertors
    - Pixel value/Standardized
    - Movement/Location

## Scope

### Universal

#### Utils

- Converters

### Numpy

#### Affine Transformation

#### Elastic Transformation

#### Utils
- 

### PyTorch

#### Affine Transformation

#### Elastic Transformation

#### Perlin Transformation

### Tensorflow

#### Affine Transformation

#### Perlin Transformation