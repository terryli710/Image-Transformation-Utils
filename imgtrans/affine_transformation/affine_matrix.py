

# Get affine matrix with parameters or range


from imgtrans.np.utils.randomize import RandomFromIntervalNumpy
from imgtrans.torch.utils.randomize import RandomFromIntervalTorch


class AffineMatrix:
    BACKENDS = ["torch", "numpy"]
    def __init__(self, ndim, random=True, backend="torch"):
        """
        Args:
            ndim (int): 2 or 3
            random (bool): if True, the affine matrix will be generated randomly
            backend (str): "torch" or "numpy"
        """
        self.ndim = ndim
        self.random = random
        assert backend in self.BACKENDS, f"Only support {self.BACKENDS} backend."
        self.backend = self._get_backend(backend)
        assert ndim in (2, 3), "Only support 2D or 3D affine matrix."
        pass

    def _get_backend(self, backend):
        # import backend function from imgtrans.[backend].affine_transformation.affine_matrix
        backend = getattr(__import__(f"imgtrans.{backend}.affine_transformation.affine_matrix", fromlist=[""]), "AffineMatrix")
        return backend(self.ndim, random=self.random)

    def __call__(self, rotate, scale, traslate, shear, nbatch=None, **kwargs):
        # params = (nbatch, ndim)
        mtx = self.backend.__call__(rotate=rotate,
                                    scale=scale,
                                    traslate=traslate,
                                    shear=shear,
                                    nbatch=nbatch,
                                    **kwargs)
        return mtx
        
            
        



