################# load packages #################
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main():
    ################# set data transform #################
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    ################# download data #################
    data_train = datasets.MNIST(root="/Users/shaoqi/Desktop/MNIST/", transform=transform, train=True, download=True)

    ################# data loader #################
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)

    for batch_idx, (data, target) in enumerate(data_loader_train):
        print(batch_idx)
        print(data.max())
        print(data.min())
        print(data.shape)
        plt.imshow(data[0, 0, :, :])
        plt.show()
        print(target)

if __name__ == '__main__':
    main()