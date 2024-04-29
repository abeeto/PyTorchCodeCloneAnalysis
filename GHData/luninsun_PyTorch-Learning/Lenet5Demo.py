import torch
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

from Lenet5 import Lenet5


def main():
    batchsz = 128
    lr = 1e-3
    epochs = 1000

    cifar_train = datasets.CIFAR10('./data/cifar',
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                                   ]))
    cifar_test = datasets.CIFAR10('./data/cifar',
                                  train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                                  ]))

    cifar_train = torch.utils.data.DataLoader(cifar_train,
                                              batch_size=batchsz,
                                              shuffle=True)
    cifar_test = torch.utils.data.DataLoader(cifar_test,
                                             batch_size=batchsz,
                                             shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

    model = Lenet5()
    criteon = nn.CrossEntropyLoss()
    optimzer = optim.Adam(model.parameters(), lr=lr)

    print(model)

    for epoch in range(epochs):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):

            # forward propagation
            logits = model(x)
            # logits: [b, 10]
            # label: [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # back propagation
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

        print('epoch: ', epoch, 'loss: ', loss.item())

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0

            for x, label in cifar_test:
                logits = model(x)

                pred = logits.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num
            print('epoch: ', epoch, 'test acc: ', acc)


if __name__ == "__main__":
    main()