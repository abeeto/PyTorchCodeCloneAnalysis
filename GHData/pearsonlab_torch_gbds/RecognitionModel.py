"""
Base class for a generative model and linear dynamical system implementation.
Based on Evan Archer's code here: https://github.com/earcher/vilds/blob/master/code/RecognitionModel.py
"""
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
from lib.sym_blk_tridiag_inv import *
from lib.blk_tridiag_chol_tools import *

class RecognitionModel(torch.nn.Module):
    """
    Recognition Model Interface Class
    Recognition model approximates the posterior given some observations (Data)
    Different forms of recognition models will have this interface
    appropriate sampling expression.
    """

    def __init__(self, Data, xDim, yDim):
        super(RecognitionModel, self).__init__()
        self.xDim = xDim
        self.yDim = yDim
        self.Data = Data

    def evalEntropy(self):
        """
        Evaluates entropy of posterior approximation
        H(q(x))
        This is NOT normalized by the number of samples
        """
        raise Exception("Please implement me. This is an abstract method.")

    def getSample(self):
        """
        Returns a PyTorch Tensor of samples from the recognition model
        given the input
        """
        raise Exception("Please implement me. This is an abstract method.")

    def setTrainingMode(self):
        """
        changes the internal state so that `getSample` will possibly return
        noisy samples for better generalization
        """
        raise Exception("Please implement me. This is an abstract method.")

    def setTestMode(self):
        """
        changes the internal state so that `getSample` will supress noise
        (e.g., dropout) for prediction
        """
        raise Exception("Please implement me. This is an abstract method.")

class SmoothingTimeSeries(RecognitionModel):
    """
    A "smoothing" recognition model. The constructor accepts neural networks which are used to parameterize mu and Sigma.
    x ~ N( mu(y), sigma(y) )
    This version is described in Section 4.1 of Archer et al.
    """

    def __init__(self, RecognitionParams, Data, xDim, yDim):
        """
        :parameters:
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * NN_Mu: [T x xDim]
                              network calculating mu(y)
                     * NN_Lambda: [T x xDim x xDim]
                                  network calculating the block diagonals of
                                  the Cholesky factor of the _precision_ matrix
                     * NN_LambdaX: [T-1 x xDim x xDim]
                                   network calculating the block
                                   off-diagonals of the Cholesky factor of
                                   the _precision_ matrix
            - xDim, yDim: (integers) dimension of
                latent space (x) and observation (y)
        """
        super(SmoothingTimeSeries, self).__init__(Data, xDim, yDim)

        self.add_module('mu', RecognitionParams['NN_Mu'])
        self.add_module('Lambda', RecognitionParams['NN_Lambda'])
        self.add_module('LambdaX', RecognitionParams['NN_LambdaX'])
        self.Tt = Data.size(0)
        self.postX = self.mu(Data)
        self.AAChol = self.Lambda(Data).view(-1, xDim, xDim) + Variable(torch.eye(xDim), requires_grad=False)
        self.BBChol = self.LambdaX(Data).view(-1, xDim, xDim)

        self._initialize_posterior_distribution()

    def _initialize_posterior_distribution(self):
        """
        Compute precision matrix and other expensive objects once up front.
        """
        xDim = self.xDim
        diagsquare = torch.bmm(self.AAChol, self.AAChol.transpose(1 ,2))

        odsquare = torch.bmm(self.BBChol, self.BBChol.transpose(1, 2))

        self.AA = diagsquare + torch.cat([Variable(torch.zeros(1, xDim, xDim), requires_grad=False), odsquare]) + 1e-6 * Variable(torch.eye(self.xDim), requires_grad=False)

        self.BB = torch.bmm(self.AAChol[:-1], self.BBChol.transpose(1, 2))

        # compute Cholesky decomposition
        self.the_chol = blk_tridiag_chol(self.AA, self.BB)

        # symbolic recipe for computing the the diagonal (V) and
        # off-diagonal (VV) blocks of the posterior covariance
        self.V, self.VV, self.S = compute_sym_blk_tridiag(self.AA, self.BB)

        # The determinant of the covariance is the square of the determinant of the cholesky factor (twice the log).
        # Determinant of the Cholesky factor is the product of the diagonal elements of the block-diagonal.
        self.logdet = 0
        for i in range(self.the_chol[0].size(0)):
            self.logdet += -2 * self.the_chol[0][i].log().sum()
