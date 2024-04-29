# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        batch_size, input_size = A.shape[0], A.shape[2]
        output_size = input_size - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(batch_size):
          for j in range(self.out_channels):
            for k in range(output_size):
              Z[i][j][k] = np.tensordot(A[i][:, k:k + self.kernel_size], self.W[j]) + self.b[j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # dLdW
        batch_size, output_size = dLdZ.shape[0], dLdZ.shape[2]
        out_channels, in_channels, kernel_size = self.W.shape[0], self.W.shape[1], self.W.shape[2]
        for batch_idx in range(batch_size):
            for i in range(out_channels):
                for k in range(kernel_size):
                    self.dLdW[i][:, k] += np.multiply(self.A[batch_idx][:, k:k + output_size], np.resize(dLdZ[batch_idx][i], (in_channels, output_size))).sum(axis=1)

        # dLdb
        for batch_idx in range(batch_size):
            for i in range(out_channels):
                self.dLdb[i] += dLdZ[batch_idx][i].sum()

         # dLdA
        # batch_size, output_size = dLdZ.shape[0], dLdZ.shape[2]
        input_size = self.A.shape[2]
        dLdA = np.zeros((batch_size, self.in_channels, input_size))
        W = np.transpose(self.W, (1, 0, 2))
        for i in range(batch_size):
          dLdZ_i = np.pad(dLdZ[i], ((0, 0), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=(0, 0))
          for j in range(self.in_channels):
            W_flipped = np.flip(W[j], axis=1)
            for k in range(input_size):
              dLdA[i][j][k] = np.tensordot(dLdZ_i[:, k:k + self.kernel_size], W_flipped)

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        Z = self.conv1d_stride1.forward(A)
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        dLdA = self.downsample1d.backward(dLdZ)
        dLdA = self.conv1d_stride1.backward(dLdA)

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        batch_size, input_width, input_height  = A.shape[0], A.shape[2], A.shape[3]
        output_width = input_width - self.kernel_size + 1
        output_height = input_height - self.kernel_size + 1
        Z = np.zeros((batch_size, self.out_channels, output_width, output_height))

        for i in range(batch_size):
          for j in range(self.out_channels):
            for k in range(output_width):
                for h in range(output_height):
                    Z[i][j][k][h] = np.multiply(A[i][:, k:k + self.kernel_size, h:h + self.kernel_size], self.W[j]).sum() + self.b[j]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # dLdW
        batch_size, output_width, output_height = dLdZ.shape[0], dLdZ.shape[2], dLdZ.shape[3]
        out_channels, in_channels, kernel_size = self.W.shape[0], self.W.shape[1], self.W.shape[2]
        for batch_idx in range(batch_size):
            for i in range(out_channels):
                for j in range(in_channels):
                    for k in range(kernel_size):
                        for h in range(kernel_size):
                            self.dLdW[i][j][k][h] += np.multiply(self.A[batch_idx][j][k:k + output_width, h:h + output_height], dLdZ[batch_idx][i]).sum()

        #dLdb
        for batch_idx in range(batch_size):
            for i in range(out_channels):
                self.dLdb[i] += dLdZ[batch_idx][i].sum()

        #dLdA
        batch_size, output_width, output_height = dLdZ.shape[0], dLdZ.shape[2], dLdZ.shape[3]
        input_width, input_height  = self.A.shape[2], self.A.shape[3]
        dLdA = np.zeros((batch_size, self.in_channels, input_width, input_height))
        for i in range(batch_size):
            for j in range(self.in_channels):
                for o in range(self.out_channels):
                    dLdZ_j = np.pad(dLdZ[i][o], ((self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=(0, 0))
                    W_flipped = np.flip(np.flip(self.W[o][j], axis=1), axis=0)
                    for k in range(input_width):
                        for h in range(input_height):
                            dLdA[i][j][k][h] += np.multiply(dLdZ_j[k:k + self.kernel_size, h:h + self.kernel_size], W_flipped).sum()

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        # Initialize Conv2d() and Downsample2d() isntance

        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.conv2d_stride1.forward(A)
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
        dLdA = self.conv2d_stride1.backward(dLdA)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsampling_factor = upsampling_factor

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

        # Initialize Conv1d stride 1 and upsample1d isntance

        self.upsample1d = Upsample1d(upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        A_upsampled = self.upsample1d.forward(A)
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        delta_out = self.conv1d_stride1.backward(dLdZ)
        dLdA = self.upsample1d.backward(delta_out)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.upsample2d = Upsample2d(upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        A_upsampled = self.upsample2d.forward(A)
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        delta_out = self.conv2d_stride1.backward(dLdZ)
        dLdA = self.upsample2d.backward(delta_out)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.shape = A.shape
        Z = A.reshape(self.shape[0], self.shape[1] * self.shape[2])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.shape[0], self.shape[1], self.shape[2])

        return dLdA
