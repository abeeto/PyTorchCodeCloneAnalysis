import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        input_width = A.shape[2]
        ups_pos = np.repeat(np.arange(1, input_width), (self.upsampling_factor - 1))
        Z = np.insert(A, ups_pos, 0, axis=2)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        arr_keep = np.arange(0, dLdZ.shape[2], self.upsampling_factor)
        arr = np.arange(dLdZ.shape[2])
        arr_del = np.delete(arr, arr_keep)
        dZdA = np.delete(dLdZ, arr_del, axis=2)

        return dZdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        self.input_shape = A.shape

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        arr_keep = np.arange(0, A.shape[2], self.downsampling_factor)
        arr = np.arange(A.shape[2])
        arr_del = np.delete(arr, arr_keep)
        Z = np.delete(A, arr_del, axis=2)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, input_width = self.input_shape[0], self.input_shape[1], self.input_shape[2]
        dLdA = np.zeros((batch_size, in_channels, input_width))
        for i in range(batch_size):
            for j in range(in_channels):
                m = 0
                for k in range(1, len(dLdZ[0][0]) + 1):
                    dLdA[i][j][m] = dLdZ[i][j][k - 1]
                    m = self.downsampling_factor * k

        return dLdA
class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):


        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        input_width, input_height = A.shape[2], A.shape[3]
        ups_pos_h = np.repeat(np.arange(1, input_height), (self.upsampling_factor - 1))
        ups_pos_w = np.repeat(np.arange(1, input_width), (self.upsampling_factor - 1))
        Z = np.insert(A, ups_pos_h, 0, axis=2)
        Z = np.insert(Z, ups_pos_w, 0, axis=3)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        arr_keep_h = np.arange(0, dLdZ.shape[2], self.upsampling_factor)
        arr_keep_w = np.arange(0, dLdZ.shape[3], self.upsampling_factor)
        arr_h = np.arange(dLdZ.shape[2])
        arr_w = np.arange(dLdZ.shape[3])
        arr_del_h = np.delete(arr_h, arr_keep_h)
        arr_del_w = np.delete(arr_w, arr_keep_w)
        dZdA = np.delete(dLdZ, arr_del_h, axis=2)
        dZdA = np.delete(dZdA, arr_del_w, axis=3)

        return dZdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        self.input_shape = A.shape
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        arr_keep_h = np.arange(0, A.shape[2], self.downsampling_factor)
        arr_keep_w = np.arange(0, A.shape[3], self.downsampling_factor)
        arr_h = np.arange(A.shape[2])
        arr_w = np.arange(A.shape[3])
        arr_del_h = np.delete(arr_h, arr_keep_h)
        arr_del_w = np.delete(arr_w, arr_keep_w)
        Z = np.delete(A, arr_del_h, axis=2)
        Z = np.delete(Z, arr_del_w, axis=3)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, input_height, input_width = self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]
        dLdA = np.zeros((batch_size, in_channels, input_height, input_width))
        for i in range(batch_size):
            for j in range(in_channels):
                n = 0
                for d in range(1, len(dLdZ[0][0][:, 0]) + 1):
                    m = 0
                    for w in range(1, len(dLdZ[0][0][0, :]) + 1):
                        dLdA[i][j][n][m] = dLdZ[i][j][d - 1][w - 1]
                        m = self.downsampling_factor * w
                    n = self.downsampling_factor * d

        return dLdA
