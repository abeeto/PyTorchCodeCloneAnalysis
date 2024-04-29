####################################################################
# Convolution and review how the different operations change the
# relationship between input and output.
# 1. What is Convolution
# 2. Determining the size of output
# 3. Stride: define the number of step that kernel moves each time
# 4. Zero Padding: add zeros around the original data matrix
####################################################################
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc


# Create a 2D convolution object
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
print(conv)
print(conv.state_dict())
# Give some values to parameters in conv
conv.state_dict()['weight'][:][:] = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0],
                                                 [1.0, 0.0, -1.0]])  # [0][0]
# print(conv.state_dict())
conv.state_dict()['bias'][:] = 0.0
print(conv.state_dict())

# Create a dummy image tensor (# of inputs, # of outputs, # of rows, # of columns)
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1
print(image)

# Call the object conv on the tensor image as an input to perform the convolution
# and assign the result to the tensor z.
z = conv(image)
print(z)


####################################################################
# Determine the Size of Output
# size of output = size of image - size of kernel + 1
####################################################################
# Create a kernel of size 2
K = 2
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K)
conv1.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv1.state_dict()['bias'][0] = 0.0

# Create an image of size 4
M = 4
image1 = torch.ones(1, 1, M, M)

# Perform convolution and verify the size
z1 = conv1(image1)
print("z1:", z1)
print("shape:", z1.shape[2:4], z1.size())  # [3, 3], [1, 1, 3, 3]


####################################################################
# Stride parameter
# 1. The parameter stride changes the number of shifts the kernel
#    moves per iteration.
# 2. size of output = (size of image - size of kernel)/stride + 1
####################################################################
# Create a convolution object with a stride of 2
conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
conv3.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv3.state_dict()['bias'][0] = 0.0

# Perform convolution and verify the size
z3 = conv3(image1)
print("z3:", z3)
print("shape:", z3.shape[2:4])  # [2, 2]


####################################################################
# Zero Padding
# 1. As you apply successive convolutions, the image will shrink.
#    You can apply zero padding to keep the image at a reasonable
#    size, which also holds information at the borders.
# 2. Add rows and columns of zeros around the image
# 3. size of image = size of image + 2 * zero_padding
####################################################################
conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3)
conv4.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv4.state_dict()['bias'][0] = 0.0

z4 = conv4(image1)
print("z4:", z4)
print("z4:", z4.shape[2:4])  # [1, 1]  math.floor((4-2)/3 + 1) = 1

conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3, padding=1)
conv5.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv5.state_dict()['bias'][0] = 0.0

z5 = conv5(image1)
print("z5:", z5)
print("z5:", z5.shape[2:4])  # [2, 2], math.floor((4+2*1 - 2)/3 + 1) = 2


