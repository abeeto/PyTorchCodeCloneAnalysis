import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16 = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=False)


vgg16.classifier[6] = nn.Linear(4096,10)
print(vgg16)