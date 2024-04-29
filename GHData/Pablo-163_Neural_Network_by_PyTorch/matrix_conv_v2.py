import torch
from abc import ABC, abstractmethod


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
        [[[[0., 1, 0],
           [1, 2, 1],
           [0, 1, 0]],

          [[1, 2, 1],
           [0, 3, 3],
           [0, 1, 10]],

          [[10, 11, 12],
           [13, 14, 15],
           [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)
    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
           and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dMatrixV2(ABCConv2d):
    # Reformat kernel
    def _convert_kernel(self):
        converted_kernel = torch.zeros(1,  # self.kernel.shape[1],
                                       self.kernel.shape[2] * self.kernel.shape[3] * self.in_channels)
        k_size_n = self.kernel_size ** 2
        for i in range(self.in_channels):
            converted_kernel[0, i * k_size_n:i * k_size_n + k_size_n] = torch.flatten(self.kernel[0, i])
        # print("kernel size is ", converted_kernel.shape)
        return converted_kernel

    # Reformat input
    def _convert_input(self, torch_input, output_height, output_width):
        converted_input = torch.zeros(self.kernel.shape[2] * self.kernel.shape[3] * torch_input.shape[1],
                                      output_width * output_height * torch_input.shape[
                                          0])
        k = 0
        for num_batch, batch in enumerate(torch_input):
            # go through batches
            for i in range(output_height):
                for j in range(output_width):
                    # go through input by kernel step
                    current_row = self.stride * i
                    current_column = self.stride * j
                    for num_channel, chanel in enumerate(batch):
                        converted_input[num_channel * self.kernel_size ** 2:(num_channel + 1) * self.kernel_size ** 2, k] = \
                            torch.flatten(chanel[current_row:current_row + self.kernel_size,
                                          current_column:current_column + self.kernel_size])
                    k += 1
        return converted_input

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width \
            = calc_out_shape(
            input_matrix_shape=torch_input.shape,
            out_channels=self.kernel.shape[0],
            kernel_size=self.kernel.shape[2],
            stride=self.stride,
            padding=0)

        converted_kernel = self._convert_kernel()
        converted_input = self._convert_input(torch_input, output_height, output_width)

        conv2d_out_alternative_matrix_v2 = converted_kernel @ converted_input
        return conv2d_out_alternative_matrix_v2.view(torch_input.shape[0],
                                                     self.out_channels,
                                                     output_height,
                                                     output_width)


# Checking
print(test_conv2d_layer(Conv2dMatrixV2))
