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
