import torch
import torchvision
from torch import nn
from torchvision import transforms

lr = 0.5
batch_size = 256
num_workers = 4
num_iters = 5
def createDatasets():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

train_iter, test_iter = createDatasets()
num_input, num_hidden, num_out = 28 * 28, 256, 10
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_input, num_hidden),
                    nn.ReLU(),
                    nn.Linear(num_hidden, num_out),
                    )


for param in net.parameters():
    torch.nn.init.normal_(param, mean=0, std=0.01)

loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr)


if __name__ == '__main__':

    for epoch in range(1, num_iters + 1):
        n = 0
        sum = 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()
            net.zero_grad()
            l.backward()
            opt.step()
            n += x.shape[0]
            sum += l.item()
        print(f"epoch = {epoch}, loss = {sum / n}")