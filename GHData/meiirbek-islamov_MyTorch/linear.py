import numpy as np

class Linear:

    def __init__(self, in_features, out_features, debug = False):

        self.W    = np.zeros((out_features, in_features), dtype="f")
        self.b    = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")

        self.debug = debug

    def forward(self, A):

        self.A    = A
        self.N    = A.shape[0]
        self.Ones = np.ones((self.N,1), dtype="f")
        Z         = A @ np.transpose(self.W) + self.Ones @ np.transpose(self.b)

        return Z

    def backward(self, dLdZ):

        dZdA      = np.transpose(self.W)
        dZdW      = self.A
        dZdi      = None #np.transpose(self.b)
        dZdb      = self.Ones
        dLdA      = dLdZ @ np.transpose(dZdA)
        dLdW      = np.transpose(dLdZ) @ dZdW
        dLdi      =  None #dLdZ @ np.transpose(dZdi)
        dLdb      = np.transpose(dLdZ) @ dZdb
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi

        return dLdA
