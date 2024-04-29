####################################################################
# Multiple Inputs & Multiple Output Channels
####################################################################
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
torch.manual_seed(0)


##################### Multiple Output Channels ####################
# For each channel, a kernel is created, and each kernel performs
# a convolution independently. As a result, the number of outputs
# is equal to the number of channels.
###################################################################
# Create a Conv2d with three channels
conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)

# By default, pytorch randomly assigns values to each kernel.
Gx = torch.tensor([[1.0, 0., -1.0], [2.0, 0., -2.0], [1.0, 0., -1.0]])
Gy = torch.tensor([[1.0, 2.0, 1.0], [0., 0., 0.], [-1.0, -2.0, -1.0]])

conv1.state_dict()['weight'][0][:] = Gx
conv1.state_dict()['weight'][1][:] = Gy
conv1.state_dict()['weight'][2][:] = torch.ones(3, 3)

conv1.state_dict()['bias'][:] = torch.zeros(3).view(1, -1)
# print(conv1.state_dict())

# Create an image
# (size of mini batch, # of input channel, row of image, column of image)
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1.0

# # Plot the image
# plt.figure()
# plt.imshow(image[0, 0, :, :].numpy(), interpolation='nearest', cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()

# Perform convolution
out = conv1(image)
# print(out.shape, out[:])

# # Print out each channel as an image
# for channel, image in enumerate(out[0]):
#     plt.figure()
#     plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
#     print(image)
#     plt.title("channel {}".format(channel))
#     plt.colorbar()
#     plt.show()


##################### Multiple Input Channels ####################
# For each input channel, a corresponding kernel is created.
# For two inputs, you can create two kernels. Each kernel performs
# a convolution on its associated input channel.
##################################################################
# Create an input with two channels
image2 = torch.zeros(1, 2, 5, 5)
image2[0, 0, 2, :] = -2
image2[0, 1, 2, :] = 1

# # Plot out each image
# for channel, image in enumerate(image2[0]):
#     plt.figure()
#     plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
#     print(image)
#     plt.title("channel {}".format(channel))
#     plt.colorbar()
#     plt.show()

# Create a Con2d object with two inputs
# Note: there is only one bias
# weight0 * image0 + weight1 * image1 + bias = output
conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3)
# print(conv3.state_dict())
Gx1 = torch.tensor([[0.0, 0.0, 0.0], [0, 1.0, 0], [0.0, 0.0, 0.0]])
conv3.state_dict()['weight'][0][0] = Gx1
conv3.state_dict()['weight'][0][1] = -2*Gx1
conv3.state_dict()['bias'][:] = torch.tensor([0.0])

# Perform convolution
out3 = conv3(image2)
# print(out3)

# # Plot the image
# plt.figure()
# plt.imshow(out3[0, 0, :, :].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
# plt.colorbar()
# plt.show()


########### Multiple Input and Multiple Output Channels ##########
# When using multiple inputs and outputs, a kernel is created for
# each input, and the process is repeated for each output.
##################################################################
# Create an example with two inputs and three outputs
conv4 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3)

# ['weight'][output channel][input channel]
conv4.state_dict()['weight'][0][0] = torch.tensor([[0., 0., 0.], [0, 0.5, 0],
                                                   [0., 0., 0.]])
conv4.state_dict()['weight'][0][1] = torch.tensor([[0., 0., 0.], [0, 0.5, 0],
                                                   [0., 0., 0.]])

conv4.state_dict()['weight'][1][0] = torch.tensor([[0., 0., 0.], [0, 1, 0],
                                                   [0., 0., 0.]])
conv4.state_dict()['weight'][1][1] = torch.tensor([[0., 0., 0.], [0, -1, 0],
                                                   [0., 0., 0.]])

conv4.state_dict()['weight'][2][0] = torch.tensor([[1., 0, -1.], [2., 0, -2.],
                                                   [1., 0., -1.]])
conv4.state_dict()['weight'][2][1] = torch.tensor([[1., 2., 1.], [0., 0., 0.],
                                                   [-1., -2., -1.]])

# For each output, there is a bias.
# # of bias = # of output
conv4.state_dict()['bias'][:] = torch.tensor([0.0, 0.0, 0.0])

# Create two images
image4 = torch.zeros(1, 2, 5, 5)
image4[0][0] = torch.ones(5, 5)
image4[0][1][2][2] = 1
# for channel, image in enumerate(image4[0]):
#     plt.figure()
#     plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
#     print(image)
#     plt.title("channel {}".format(channel))
#     plt.colorbar()
#     plt.show()

# Perform convolution
out4 = conv4(image4)

# Print out each channel as an image
for channel, image in enumerate(out4[0]):
    plt.figure()
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

