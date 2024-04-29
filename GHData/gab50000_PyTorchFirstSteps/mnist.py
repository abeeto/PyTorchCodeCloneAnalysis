#!/usr/bin/env python3


from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms


def main():
    batch_size = 50
    in_size, hidden_size, out_size = 28 * 28, 100, 1
    
    x = Variable(torch.randn(batch_size, in_size))
    y = Variable(torch.randn(batch_size, out_size), requires_grad=False)

    model = torch.nn.Sequential(
        torch.nn.Linear(in_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, 10),
    )

    loss_func = torch.nn.CrossEntropyLoss()

    trainset = datasets.MNIST(root="data", train=True, download=True,
                              transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset)

    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    for ep in range(1000):
        for data, target in trainloader:
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            y_net = model(data.view(-1, 784))
            loss = loss_func(y_net, target)
            loss.backward()
            optimizer.step()
            import ipdb; ipdb.set_trace()
        print(ep)



if __name__ == "__main__":
    main()
