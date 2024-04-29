import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

def createDatasets():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

# def softmax(y):
#     if not isinstance(y, torch.Tensor):
#         y = torch.Tensor(y)
#     x_exp = y.exp()
#     x_sum = x_exp.sum(dim=1, keepdim=True)
#     return x_exp / x_sum


class MyNet(nn.Module):
    def __init__(self, feature_number, classes):
        super(MyNet, self).__init__()
        self.linear = nn.Linear(feature_number, classes)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


def accuracy(y_hat, y):
    """
    计算精度
    :param y_hat:
    :param y:
    :return:
    """
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.Tensor(y_hat)
    if not isinstance(y, torch.Tensor):
        y = torch.Tensor(y)
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# def cross_entropy(y_hat, y):
#     return - torch.log(y_hat.gather(1, y.view(-1, 1)))


if __name__ == '__main__':
    num_iters = 10
    batch_size = 256
    num_workers = 4
    feature_number = 28 * 28
    classes = 10
    lr = 0.1
    loss = nn.CrossEntropyLoss()

    net = MyNet(feature_number, classes)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = createDatasets()

    for epoch in range(1, num_iters + 1):
        loss_sum, n = 0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()
            loss_sum += l.item()
            l.backward()
            opt.step()
            opt.zero_grad()
            n += x.shape[0]
        print(loss_sum / n)