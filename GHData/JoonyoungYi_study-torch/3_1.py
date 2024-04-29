import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import visdom

vis = visdom.Visdom()

mnist_train = dset.MNIST(
    "./",
    train=True,
    transform=transforms.ToTensor(),
    target_transform=None,
    download=True)
# print(mnist_train)

# dataset.__getitem__(idx)
image, label = mnist_train.__getitem__(0)
# print(image.size(), label)

# dataset[idx]
image, label = mnist_train[0]
# print(image.size(), label)

# dataset.__len__()
# print(mnist_train.__len__())

# len(dataset)
# print(len(mnist_train))

# for i in range(3):
#     img = mnist_train[i][0].numpy()
#     vis.image(img)

image, label = mnist_train[0]
image = image.view(-1, *image.size())

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
output = conv_layer(Variable(image))
print(output.size())
#
# for i in range(3):
#     img = output[0, i, :, :].data.numpy()
#     vis.image(img)

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
output = conv_layer(Variable(image))
print(output.size())
vis.image(output[0, 0, :, :].data.numpy())
print(output.size())

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=3)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, padding=1)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=2)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
vis.image(output[0, 0, :, :].data.numpy())
print(output.size())
#
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, padding=3)
output = conv_layer(Variable(image))
# plt.imshow(output[0, 0, :, :].data.numpy(), cmap='gray')
# plt.show()
vis.image(output[0, 0, :, :].data.numpy())
print(output.size())
