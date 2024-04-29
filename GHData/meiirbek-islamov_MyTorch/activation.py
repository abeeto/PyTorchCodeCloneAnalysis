import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


# Activation Functions: Sigmoid, Tanh, ReLU

class Sigmoid:

    def forward(self, z):
        self.A = 1/(1 + np.exp(-z))
        return self.A

    def backward(self):

        dAdZ = self.A - (self.A)**2
        return dAdZ

class Tanh:

    def forward(self, z):
        self.A = np.tanh(z)
        return self.A

    def backward(self):

        dAdZ = 1 - (self.A)**2
        return dAdZ

class ReLU:

    def forward(self, z):
        self.A = np.maximum(z, 0)
        return self.A

    def backward(self):

        dAdZ = np.where(self.A>0, 1, 0)
        return dAdZ
