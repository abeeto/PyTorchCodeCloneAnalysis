# This code show how to download existing standard data sets, and change them into iterable one using data loader
# the DATA below can be any dataset: MNIST, FashionMNIST, COCO, LSUN, ImageFolder, Imagenet-12, CIFAR10, CIFAR100, STL10, SVHN, PhotoTour

import torchvision

# transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# training set
trainset = torchvision.datasets.DATA(root='./data', train=True,
                                        download=True, transform=transform)
# training loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Test set
testset = torchvision.datasets.DATA(root='./data', train=False,
                                       download=True, transform=transform)

# test loader
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# training loop
for batch_idx, (data, target) in enumerate(trainloader):
	...