# pyright: reportGeneralTypeIssues=false

# Generate deformation field

import torch
import torch.nn.functional as nnf
from .utils.spatial import draw_perlin
from .utils.grid_utils import dvf2flow_grid


class SVFTransform:

    def __init__(
        self,
        nsteps: int = 3,
    ):
        self.nsteps = nsteps

    pass

    def __call__(
        self,
        img: torch.Tensor,
        svf: torch.Tensor,
        out_shape=None,
        mode="nearest",
        padding_mode="reflection",
    ):
        """

        Args:
            img (torch.Tensor): image (C or B, H, W, (D))
            svf (torch.Tensor): stationary velocity field, array like, (H, W, (D), num_dim)
            out_shape (_type_, optional): image shape. Defaults to None.
            mode (str, optional): str. Defaults to "nearest".
            padding_mode (str, optional): str. Defaults to "reflection".

        Returns:
            Tuple: image (C or B, H, W, (D)), {dict of param}
        """
        if not out_shape:
            out_shape = img.shape[1:]
        flow_svf = dvf2flow_grid(
            svf, out_shape)  # convert range from percentage to -1, 1
        img_copy = img.clone().unsqueeze(0).type(torch.float32) # have to create another dim for grid_sample
        for i in range(self.nsteps):
            img_copy = nnf.grid_sample(
                input=img_copy,
                grid=flow_svf.unsqueeze(0),
                mode=mode,
                align_corners=False,
                padding_mode=padding_mode,
            )
        return img_copy[0,...], {"flow_svf": flow_svf}


class RandSVFTransform(SVFTransform):

    def __init__(
        self,
        scale: int = 16,
        max_std: float = 3.0,
        nsteps: int = 7,
    ):
        super().__init__(nsteps)
        self.scale = scale
        self.max_std = max_std
        self.nsteps = nsteps
        pass

    def __call__(self,
                 img: torch.Tensor,
                 out_shape=None,
                 max_std=None,
                 mode="nearest",
                 padding_mode="reflection",
                 seed=None):
        """_summary_

        Args:
            image (torch.Tensor): image = (C or B, H, W, (D))
            max_std (_type_, optional): max standard deviation. Defaults to None.
            mode (str, optional): str. Defaults to "nearest".
            padding_mode (str, optional): str. Defaults to "reflection".
            seed (_type_, optional): int. Defaults to None.

        Returns:
            Tuple: image (C or B, H, W, (D)), {dict of param}
        """
        if not out_shape:
            out_shape = img.shape[1:]

        ndim = len(out_shape)

        svf = draw_perlin(out_shape=(*out_shape, ndim),
                          scales=[self.scale],
                          max_std=self.max_std if max_std is None else max_std,
                          seed=seed)
        return SVFTransform.__call__(self,
                                     img=img,
                                     svf=svf,
                                     out_shape=out_shape,
                                     mode=mode,
                                     padding_mode=padding_mode)


# class RandSVFTrans(RandSVFTransform):

#     def __init__(self,
#                  img_size: Tuple[int, int] = ...,
#                  downratio: int = 16,
#                  std_upperbound: float = 3,
#                  batch_size: int = 1,
#                  nsteps: int = 7,
#                  tensor_output=True,
#                  int_output=True,
#                  seed=None):
#         super().__init__(img_size=img_size,
#                          downratio=downratio,
#                          std_upperbound=std_upperbound,
#                          batch_size=batch_size,
#                          nsteps=nsteps,
#                          int_output=int_output,
#                          seed=seed)
#         self.tensor_output = tensor_output

#     def __call__(self,
#                  img: torch.Tensor,
#                  mode="nearest",
#                  padding_mode="reflection",
#                  seed=None,
#                  **kwgs):

#         result = self.forward(image=img,
#                               mode=mode,
#                               padding_mode=padding_mode,
#                               seed=seed,
#                               **kwgs)
#         deformed_img = result[0]
#         if not self.tensor_output:
#             return deformed_img.numpy(), result[1:]
#         else:
#             return deformed_img, result[1:]

# class SVF2DVF(nn.Module):
#     """yields an SVF flow grid, returns DVF fullsize"""

#     def __init__(self,
#                  img_size: Tuple[int, int],
#                  downratio: int,
#                  nsteps: int = 7):
#         super(SVF2DVF, self).__init__()
#         self.img_size = img_size
#         self.nsteps = nsteps
#         self.integrate = VecInt(self.img_size, nsteps=self.nsteps)
#         pass

#     def forward(self, flow_grid):
#         # 1. upsample to img_size
#         fullsize_grid = self.resize.forward(flow_grid)
#         # 2. integrate
#         dvf = self.integrate(fullsize_grid)
#         # 3. return
#         # NOTE: for this deformation vector field, shape = (B, 2, H, W)
#         # the value of each position (h1, w1) -> (X, Y) in dvf[b, h1, w1]
#         # indicates the direction that the pixel finally moved to, unit = pixel
#         return dvf

# class RandSVFFlowGrid(SVFFlowGrid):
#     """perform random SVF transformation"""

#     def __init__(
#         self,
#         img_size: Tuple[int, int] = (160, 160),
#         downratio: int = 16,
#         std_upperbound: int = 3,
#         batch_size: int = 1,
#         nsteps: int = 7,
#         seed=None,
#     ):
#         super().__init__(downratio=downratio, img_size=img_size, nsteps=nsteps)
#         self.batch_size = batch_size
#         grid_size = tuple([int(s / downratio) for s in img_size])
#         self.rand_svf_grid = RandGaussianGrid(
#             size=grid_size,
#             std_upperbound=std_upperbound,
#             batch_size=self.batch_size,
#             seed=seed,
#         )
#         pass

#     def forward(self, batch_size=None, std_upperbound=None, seed=None):
#         if batch_size:
#             self.rand_svf_grid.set_batch_size(batch_size=batch_size)
#         if std_upperbound:
#             self.rand_svf_grid.set_std_upperbound(
#                 std_upperbound=std_upperbound)
#         svf = draw_perlin(out_shape=self.img_size,
#                           scales=[self.downratio],
#                           max_std=std_upperbound,
#                           seed=seed)
#         return super().forward(svf)

# # class DVF2FlowGrid(nn.Module):
# #     # convert a SVF matrix to a flow grid (range = (-1, 1))

# #     def __init__(self, channel_last=False, pixel_range=(-1, 1)):
# #         super().__init__()
# #         self.channel_last = channel_last
# #         self.pixel_range = pixel_range
# #         pass

# #     def forward(self, dvf: torch.Tensor):
# #         # dvf shape = (B, 2, H, W)
# #         img_shape = dvf.shape[2:]
# #         H, W = img_shape
# #         coord_map = torch.stack(
# #             torch.meshgrid(
# #                 torch.linspace(0, H - 1, steps=H), torch.linspace(0, W - 1, steps=W), indexing="ij",
# #             ),
# #             axis=0,
# #         )
# #         # coord_map = (2, H, W); coord_map[: ,h , w] = [h, w]
# #         new_coord = coord_map + dvf
# #         # new coord that each pixel should be in (B, 2, H, W)
# #         # scale
# #         scale_factor = torch.ones([1, 2, 1, 1])
# #         scale_factor[0, :, 0, 0] = torch.tensor(img_shape) / 2
# #         new_coord -= scale_factor
# #         new_coord /= scale_factor
# #         if self.channel_last:
# #             return self.permute(new_coord)
# #         else:
# #             return new_coord

# #     def permute(self, dvf):
# #         # permute to be channel last
# #         # input dvf = (B, 2, H, W)
# #         # output dvf = (B, H, W, 2)
# #         # return dvf.permute(0, 2, 3, 1)
# #         return dvf.permute(0, 3, 2, 1)  # HACK: debug

# class SpatialTransformer(nn.Module):
#     """
#     N-D Spatial Transformer
#     """

#     def __init__(self, size, mode="bilinear"):
#         super(SpatialTransformer, self).__init__()

#         self.mode = mode

#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors, indexing="ij")
#         grid = torch.stack(grids)
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.type(torch.FloatTensor)

#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer("grid", grid)
#         pass

#     def forward(self, src, flow):
#         # new locations
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]

#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i,
#                      ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

#         # move channels dim to last position
#         # also not sure why, but the channels need to be reversed
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]

#         return nnf.grid_sample(src,
#                                new_locs,
#                                align_corners=False,
#                                mode=self.mode)

# class VecInt(nn.Module):
#     """
#     Integrates a vector field via scaling and squaring. TODO: modify this
#     """

#     def __init__(self, inshape, nsteps):
#         super(VecInt, self).__init__()

#         assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
#         self.nsteps = nsteps
#         self.scale = 1.0 / (2**self.nsteps)
#         self.transformer = SpatialTransformer(inshape)

#     def forward(self, vec):
#         vec = vec * self.scale
#         for _ in range(self.nsteps):
#             vec = vec + self.transformer(vec, vec)
#         return vec

# class ResizeTransform(nn.Module):
#     """
#     Resize a transform, which involves resizing the vector field *and* rescaling it.
#     """

#     def __init__(self, factor, ndims):
#         super(ResizeTransform, self).__init__()
#         self.factor = factor  #  = downratio
#         self.mode = "linear"
#         if ndims == 2:
#             self.mode = "bi" + self.mode
#         elif ndims == 3:
#             self.mode = "tri" + self.mode

#     def forward(self, x):
#         if self.factor < 1:
#             # resize first to save memory
#             x = nnf.interpolate(
#                 x,
#                 align_corners=False,  # set to false or will have warnings
#                 scale_factor=self.factor,
#                 mode=self.mode,
#             )
#             x = self.factor * x

#         elif self.factor > 1:
#             # multiply first to save memory
#             x = self.factor * x
#             x = nnf.interpolate(
#                 x,
#                 align_corners=False,  # set to false or will have warnings
#                 scale_factor=self.factor,
#                 mode=self.mode,
#             )

#         # don't do anything if resize is 1
#         return x