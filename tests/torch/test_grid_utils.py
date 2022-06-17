import unittest
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch

os.environ["IMG_TRANS_BACKEND"] = "torch"


class TestGridUtils(unittest.TestCase):

    DIRPATH = os.path.dirname(os.path.realpath(__file__))

    def test_dvf2flow_grid(self):
        import imgtrans as imt

        # test if flow_grid is correctly generated from dvf
        dvf = torch.zeros((5, 5, 2))
        flow_grid = imt.torch.utils.grid_utils.dvf2flow_grid(dvf[None, ...])
        synth_flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                                     torch.linspace(-1, 1, 5),
                                                     indexing="xy"),
                                      dim=-1)

        self.assertTrue(torch.allclose(flow_grid, synth_flow_grid[None, ...]))
        pass

    def test_flow_grid2dvf(self):
        import imgtrans as imt

        # test if dvf is correctly generated from flow_grid
        flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                               torch.linspace(-1, 1, 5)),
                                                     indexing="xy",
                                dim=-1)
        dvf = imt.torch.utils.grid_utils.flow_grid2dvf(flow_grid[None, ...])
        synth_dvf = torch.zeros((5, 5, 2))

        self.assertTrue(torch.allclose(dvf, synth_dvf[None, ...]))
        pass


class DebugGridUtils:

    def debug_flow_grid2dvf_2d(self):
        import imgtrans as imt

        flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                               torch.linspace(-1, 1, 5), 
                                               indexing="xy",),
                                             dim=-1)
        dvf = imt.torch.utils.grid_utils.flow_grid2dvf(flow_grid[None, ...])
        synth_dvf = torch.zeros((5, 5, 2))

        assert np.array_equal(dvf, synth_dvf[None, ...])
        print("debug_flow_grid2dvf_2d passed")
        pass


    def debug_flow_grid2dvf(self):
        import imgtrans as imt

        flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                               torch.linspace(-1, 1, 5),
                                               torch.linspace(-1, 1, 5),
                                                  indexing="xy",),
                                dim=-1)
        dvf = imt.torch.utils.grid_utils.flow_grid2dvf(flow_grid[None, ...])
        synth_dvf = torch.zeros((5, 5, 5, 3))

        assert np.array_equal(dvf, synth_dvf[None, ...])
        print("debug_flow_grid2dvf passed")
        pass


    def debug_dvf2flow_grid(self):
        import imgtrans as imt

        # test if flow_grid is correctly generated from dvf
        dvf = torch.zeros((5, 5, 5, 3))
        flow_grid = imt.torch.utils.grid_utils.dvf2flow_grid(dvf[None, ...])
        synth_flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                                     torch.linspace(-1, 1, 5),
                                                     torch.linspace(-1, 1, 5),
                                                     indexing="xy"),
                                      dim=-1)

        assert np.array_equal(synth_flow_grid[None, ...], flow_grid)
        print("debug_dvf2flow_grid passed")
        pass

    def debug_dvf2flow_grid_2d(self):
        import imgtrans as imt

        # test if flow_grid is correctly generated from dvf
        dvf = torch.zeros((5, 5, 2))
        flow_grid = imt.torch.utils.grid_utils.dvf2flow_grid(dvf[None, ...])
        synth_flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                                     torch.linspace(-1, 1, 5),
                                                     indexing="xy"),
                                      dim=-1)
        
        assert np.array_equal(synth_flow_grid[None, ...], flow_grid)
        print("debug_dvf2flow_grid_2d passed")
        pass


    def debug_flow_grid2dvf_real_labelmap(self):
        import imgtrans as imt
        perlin = imt.perlin.RandPerlin(scales=(4, 8), min_std=0, max_std=0.2)
        def load_labelmap():
            labelmap = np.load("/home/yiheng/processed_data/lspine_labelmap/imgs_320-320-16/406.npy", allow_pickle=True)
            return labelmap
        label_map = load_labelmap()
        label_map = torch.from_numpy(label_map[None, ...].astype("float32")) # for perlin, the image is (B or C, H, W, D)
        label_map_1, param1 = perlin(label_map)
        
        pass

if __name__ == '__main__':
    # unittest.main()
    # NOTE: for debug
    dgu = DebugGridUtils()
    dgu.debug_flow_grid2dvf_2d()
    dgu.debug_flow_grid2dvf()
    dgu.debug_dvf2flow_grid_2d()
    dgu.debug_dvf2flow_grid()
    pass