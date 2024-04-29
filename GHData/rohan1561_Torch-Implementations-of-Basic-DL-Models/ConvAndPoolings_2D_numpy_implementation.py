import numpy as np

'''
This is an implementation of the forward Prop in Conv2d, AvgPool2d and MaxPool2d
'''

class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        kernel_size (tuple): the shape of the kernel. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, kernel_height, kernel_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            out_channels).
        kernel_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, kernel_size, stride, padding):
        self.W = np.random.randn(*kernel_size)
        self.b = np.random.randn(kernel_size[0], 1)
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        o_height = (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1
        o_width = (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 
        output = np.zeros((self.W.shape[0], int(o_height), int(o_width)))
        for c in range(self.W.shape[0]):
            o_h = -s[0]
            for h in range(0, int(o_height)):
                o_h += s[0]
                o_w = -s[1]
                for w in range(0, int(o_width)):
                    o_w += s[1]
                    output[c, h, w] = np.sum(x_padded[:,\
                    o_h:o_h+self.W.shape[2], o_w:o_w+self.W.shape[3]]*self.W[c]) + self.b[c]

        return output


class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        o_height = (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1
        o_width = (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1
        output = np.zeros((x.shape[0], int(o_height), int(o_width)))
        for c in range(x.shape[0]):
            o_h = -s[0]
            for h in range(0, int(o_height)):
                o_h += s[0]
                o_w = -s[1]
                for w in range(0, int(o_width)):
                    o_w += s[1]
                    output[c, h, w] = np.max(x_padded[c,\
                    o_h:o_h+self.kernel_size[0], o_w:o_w+self.kernel_size[1]])

        return output


class AvgPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        o_height = (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1
        o_width = (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1
        output = np.zeros((x.shape[0], int(o_height), int(o_width)))
        for c in range(x.shape[0]):
            o_h = -s[0]
            for h in range(0, int(o_height)):
                o_h += s[0]
                o_w = -s[1]
                for w in range(0, int(o_width)):
                    o_w += s[1]
                    output[c, h, w] = np.mean(x_padded[c,\
                    o_h:o_h+self.kernel_size[0], o_w:o_w+self.kernel_size[1]])

        return output

