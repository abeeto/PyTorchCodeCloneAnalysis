import numpy

import torch
import torch.nn as nn

import OcelotMinimal.cpbd.elements as elements


class LinearMap(nn.Linear):
    def __init__(self, element, rMatrix: numpy.ndarray, dtype: torch.dtype = torch.float32):
        self.element = element
        self.dtype = dtype

        # dimension of transfer matrix
        self.dim = rMatrix.shape[0]

        # set symplectic structure matrix
        if self.dim == 2:
            symStruct = torch.tensor([[0,1],[-1,0]], dtype=dtype)
        elif self.dim == 4:
            symStruct = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], dtype=dtype)
        elif self.dim == 6:
            symStruct = torch.tensor(
                [[0,1,0,0,0,0],[-1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,-1,0,0,0],[0,0,0,0,0,1],
                 [0,0,0,0,-1,0]], dtype=dtype)
        else:
            raise NotImplementedError("phase space dimension of {} not supported".format(self.dim))

        super().__init__(in_features=self.dim, out_features=self.dim, bias=False)
        self.register_buffer("symStruct", symStruct)

        # set initial weights
        weightMatrix = torch.as_tensor(rMatrix, dtype=dtype)
        self.weight = nn.Parameter(weightMatrix)

        # modify gradients during backward pass
        if type(element) is elements.Quadrupole:
            self.weight.register_hook(self.updateGradient)

            # quad trainable elements
            self.quadIndices = torch.zeros((self.dim, self.dim), dtype=torch.bool)
            for i,j in [[0,0], [0,1], [1,0], [1,1], [2,2], [2,3], [3,2], [3,3]]:
                self.quadIndices[i,j] = 1

        return

    def symplecticRegularization(self):
        """Calculate norm of Transpose(J).S.J-S symplectic condition."""
        penalty = torch.matmul(self.weight.transpose(1,0), torch.matmul(self.symStruct, self.weight)) - self.symStruct
        penalty = torch.norm(penalty)
        return penalty

    def updateGradient(self, grad):
        """Update gradient during backward pass."""
        newGrad = torch.zeros((self.dim, self.dim), dtype=self.dtype)
        newGrad[self.quadIndices] = grad[self.quadIndices]
        return newGrad


class SecondOrderMap(nn.Module):
    def __init__(self, element, rMatrix: numpy.ndarray, tMatrix: numpy.ndarray, dtype: torch.dtype = torch.float32):
        super(SecondOrderMap, self).__init__()
        self.element = element

        # dimension of transfer matrix
        self.dim: int = rMatrix.shape[0]

        # first order
        self.w1 = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        w1Weights = torch.as_tensor(rMatrix, dtype=dtype)
        self.w1.weight = nn.Parameter(w1Weights)

        # second order
        w2 = torch.as_tensor(tMatrix, dtype=dtype)
        w2 = torch.reshape(w2, (self.dim, self.dim, self.dim))
        w2.requires_grad_(True)
        self.register_parameter("w2", nn.Parameter(w2))

        return

    def forward(self, x):
        # evaluate linear map
        x1 = self.w1(x)

        # second order map, expressions are equivalent to bij,...i,...j->...b
        x2 = torch.einsum("...i,...j->...ij", x, x)
        x2 = torch.einsum("...ij,bij->...b", x2, self.w2)
        return x1 + x2


if __name__ == "__main__":
    from TorchOcelot.Lattice import SIS18_Lattice_minimal

    # get transfer map
    lattice = SIS18_Lattice_minimal()
    latticeElementIterator = lattice.getTransferMaps(dim=2)

    rMatrix, _, element, _ = next(latticeElementIterator)

    # create layer
    myLayer = LinearMap(element, rMatrix)


