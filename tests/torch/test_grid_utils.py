import unittest
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
os.environ["IMG_TRANS_BACKEND"] = "torch"

class TestGridUtils(unittest.TestCase):

    DIRPATH = os.path.dirname(os.path.realpath(__file__))

    def test_flow_grid_to_dvf(self):
        import imgtrans as imt
        
        # test if flow_grid is correctly generated from dvf
        dvf = torch.zeros((5, 5, 2))
        flow_grid = imt.torch.utils.grid_utils.dvf2flow_grid(dvf)
        synth_flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5), 
                                      torch.linspace(-1, 1, 5)), dim=-1)

        self.assertTrue(torch.allclose(flow_grid, synth_flow_grid))
        pass


    def test_dvf_to_flow_grid(self):
        import imgtrans as imt

        # test if dvf is correctly generated from flow_grid
        flow_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5), 
                                               torch.linspace(-1, 1, 5)), dim=-1)
        dvf = imt.torch.utils.grid_utils.flow_grid2dvf(flow_grid)
        synth_dvf = torch.zeros((5, 5, 2))

        self.assertTrue(torch.allclose(dvf, synth_dvf))
        pass


if __name__ == '__main__':
    unittest.main()