import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self._getData()

    def _getData(self):
        train = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        test = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        self.train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
        self.test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


if __name__ == '__main__':
    dataset = Data()
    for data in dataset.test_set:
        x, y = data[0][0], data[1][0]
        print(y)
        plt.imshow(x.view(28, 28))
        plt.show()
        input()

