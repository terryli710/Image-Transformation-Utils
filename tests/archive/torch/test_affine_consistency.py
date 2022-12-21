import torch
import sys
import os


class TestConsistencyWithKeyMorph:
    """ test the consistancy of affine transform with keymorph's affine solution """

    def __init__(self):
        pass

    @staticmethod
    def affine_matrix(size, s=0.2, o=0.001, a=0.785, z=0.1, cuda=True, random=True):
        """
        Creates a random affine matrix that is used for augmentatiojn
        
        Arguments
        ---------
        size  : size of input .size() method
        s     : scaling interval     [1-s,1+s]
        o     : translation interval [-o,o]
        a     : rotation interval    [-a,a]
        z     : shear interval       [-z,z]
        
        Return
        ------
            out : random affine matrix with paramters drawn from the input parameters 
        """

        n_batch = size[0]
        device = torch.device('cuda:0') if cuda else torch.device('cpu')

        if random:
            scale = torch.FloatTensor(n_batch, 3).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(n_batch, 3).uniform_(-o, o)
            theta = torch.FloatTensor(n_batch, 3).uniform_(-a, a)
            shear = torch.FloatTensor(n_batch, 6).uniform_(-z, z)
        else:
            scale = torch.FloatTensor(n_batch, 3).fill_(1+s)
            offset = torch.FloatTensor(n_batch, 3).fill_(o)
            theta = torch.FloatTensor(n_batch, 3).fill_(a)
            shear = torch.FloatTensor(n_batch, 6).fill_(z)

        ones = torch.ones(n_batch).float()

        if cuda:
            scale = scale.cuda()
            offset = offset.cuda()
            theta = theta.cuda()
            shear = shear.cuda()
            ones = ones.cuda()

        """Scaling"""
        Ms = torch.zeros([n_batch, 4, 4],
                        device=device)
        Ms[:,0,0] = scale[:,0]
        Ms[:,1,1] = scale[:,1]
        Ms[:,2,2] = scale[:,2]
        Ms[:,3,3] = ones

        """Translation"""
        Mt = torch.zeros([n_batch, 4, 4],
                        device=device)
        Mt[:,0,3] = offset[:,0]
        Mt[:,1,3] = offset[:,1]
        Mt[:,2,3] = offset[:,2]
        Mt[:,0,0] = ones
        Mt[:,1,1] = ones
        Mt[:,2,2] = ones
        Mt[:,3,3] = ones

        """Rotation"""
        dim1_matrix = torch.zeros([n_batch, 4, 4], device=device)
        dim2_matrix = torch.zeros([n_batch, 4, 4], device=device)
        dim3_matrix = torch.zeros([n_batch, 4, 4], device=device)

        dim1_matrix[:,0,0] = ones
        dim1_matrix[:,1,1] = torch.cos(theta[:,0])
        dim1_matrix[:,1,2] = -torch.sin(theta[:,0])
        dim1_matrix[:,2,1] = torch.sin(theta[:,0])
        dim1_matrix[:,2,2] = torch.cos(theta[:,0])
        dim1_matrix[:,3,3] = ones

        dim2_matrix[:,0,0] = torch.cos(theta[:,1])
        dim2_matrix[:,0,2] = torch.sin(theta[:,1])
        dim2_matrix[:,1,1] = ones
        dim2_matrix[:,2,0] = -torch.sin(theta[:,1])
        dim2_matrix[:,2,2] = torch.cos(theta[:,1])
        dim2_matrix[:,3,3] = ones

        dim3_matrix[:,0,0] = torch.cos(theta[:,2])
        dim3_matrix[:,0,1] = -torch.sin(theta[:,2])
        dim3_matrix[:,1,0] = torch.sin(theta[:,2])
        dim3_matrix[:,1,1] = torch.cos(theta[:,2])
        dim3_matrix[:,2,2] = ones
        dim3_matrix[:,3,3] = ones

        """Sheer"""
        Mz = torch.zeros([n_batch, 4, 4],
                        device=device)

        Mz[:,0,1] = shear[:,0]
        Mz[:,0,2] = shear[:,1]
        Mz[:,1,0] = shear[:,2]
        Mz[:,1,2] = shear[:,3]
        Mz[:,2,0] = shear[:,4]
        Mz[:,2,1] = shear[:,5]
        Mz[:,0,0] = ones
        Mz[:,1,1] = ones
        Mz[:,2,2] = ones
        Mz[:,3,3] = ones

        Mr = torch.bmm(dim3_matrix, torch.bmm(dim2_matrix, dim1_matrix))
        M = torch.bmm(Mz,torch.bmm(Ms,torch.bmm(Mt, Mr)))
        return M

    def test_affine_mtx(self):
        dims = (1, 50, 50, 50)
        keymorph_mtx = self.affine_matrix(size=dims,
                                     s=0.2,
                                     o=0.001,
                                     a=0.785,
                                     z=0.1,
                                     cuda=True,)

        # our affine matrix
        os.environ["IMG_TRANS_BACKEND"] = "pytorch"
        import imgtrans as imt
        affine = imt.affine.Affine(spatial_dims=len(dims) - 1, 
                                    rotate=0.1,
                                    shear=0,
                                    translate=0.1,
                                    scale=0)
        affine_mtx = affine.aff_mtx
        print(f"{keymorph_mtx=}") # (batch, 4, 4)
        print(f"{affine_mtx=}") # (4, 4)
        pass


    @staticmethod
    def test_affine_grid():
        # TODO: test reverting affine transform if that works
        pass


if __name__ == "__main__":
    test = TestConsistencyWithKeyMorph()
    test.test_affine_mtx()



