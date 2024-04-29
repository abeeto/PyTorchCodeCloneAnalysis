import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size: int = 128,
                    num_workers: int = 8):
    # torchvisionの出力は[0, 1]
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=trans)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=max(1, num_workers//4))
    return trainloader, testloader
