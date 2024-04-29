import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]
        output_width, output_height = input_width - self.kernel + 1, input_height - self.kernel + 1
        out_channels = in_channels
        index = np.empty((batch_size, out_channels, output_width, output_height), dtype=object)
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_width):
                    for h in range(output_height):
                        Z[i][j][k][h] = np.max(A[i][j][k:k + self.kernel,h:h + self.kernel])
                        index[i][j][k][h] = np.array((np.unravel_index((A[i][j][k:k + self.kernel,h:h + self.kernel]).argmax(),
                        (A[i][j][k:k + self.kernel,h:h + self.kernel]).shape))) + np.array((k, h))
        # print(self.index[0][0])
        self.index = index
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, input_width, input_height = self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))
        batch_size, out_channels, output_width, output_height = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2], dLdZ.shape[3]
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_width):
                    for h in range(output_height):
                        x, y = self.index[i][j][k][h]
                        dLdA[i][j][x, y] += dLdZ[i][j][k][h]
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]
        output_width, output_height = input_width - self.kernel + 1, input_height - self.kernel + 1
        out_channels = in_channels
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_width):
                    for h in range(output_height):
                        Z[i][j][k][h] = np.mean(A[i][j][k:k + self.kernel,h:h + self.kernel])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channels, input_width, input_height = self.A.shape[0], self.A.shape[1], self.A.shape[2], self.A.shape[3]
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))
        batch_size, out_channels, output_width, output_height = dLdZ.shape[0], dLdZ.shape[1], dLdZ.shape[2], dLdZ.shape[3]
        for i in range(batch_size):
            for j in range(out_channels):
                for k in range(output_width):
                    for h in range(output_height):
                        for m in range(self.kernel):
                            for n in range(self.kernel):
                                dLdA[i][j][k + m][h + n] += (1/((self.kernel)**2)) * dLdZ[i][j][k][h]
        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA)

        return dLdA
