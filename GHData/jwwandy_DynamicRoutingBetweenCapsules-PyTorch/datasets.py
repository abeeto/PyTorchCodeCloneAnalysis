import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def onehot_transform(label, num_class=10):
    target = torch.zeros(num_class)
    target[label] = 1
    return target

def get_shift_MNIST(root="data", shift=2):
    rand_shift_transform = transforms.RandomAffine(degrees=0,
                                                   translate=(shift/28, shift/28))
    pil_to_tensor = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=True,
                transform=transforms.Compose([rand_shift_transform,
                                              pil_to_tensor]),
                target_transform=onehot_transform)
    test = datasets.MNIST(root=root, train=False, download=True,
                          transform=pil_to_tensor,
                          target_transform=onehot_transform)
    return train, test


