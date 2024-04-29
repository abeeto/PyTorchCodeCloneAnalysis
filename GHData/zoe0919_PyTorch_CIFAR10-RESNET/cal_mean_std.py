import torch
import torchvision
from torchvision.datasets import ImageFolder


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    #   ------------------------------------------------------
    #   将数据分割成batch，打乱
    #   num_workers：提前将workers加载到RAM
    #   pin_memory：锁页内存
    #   torchloader产生的tensor 分别是 [batch，channel，height，width]
    #   -----------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    #   将图片转换成Tensor类型
    train_dataset = ImageFolder(root= r'C:\Users\Tingkai_Bao\Desktop\PyTorch_CIFAR10-RESNET\data\ml-cifar10', transform=torchvision.transforms.ToTensor())
    print(getStat(train_dataset))

