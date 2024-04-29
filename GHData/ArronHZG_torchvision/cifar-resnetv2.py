import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 128

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=val_transform)

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss().to(device)


def train(net, optimizer, scheduler, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        print(f'[{str(epoch + 1).zfill(2)}] '
              f'lr: {round(optimizer.param_groups[0]["lr"], 4)} '
              f'loss: {round(running_loss / len(trainloader), 4)} '
              f'acc: {acc}%', end=' | ')

        val_loss = valid(net)
        scheduler.step(val_loss)

    print('Finished Training')


def valid(net, print_class=False):
    correct = 0
    total = 0
    running_loss = 0.0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    val_loss = running_loss / len(trainloader)
    acc = 100 * correct / total
    print(f"val_loss: {round(val_loss, 4)} acc: {acc}%")

    if print_class:
        for i in range(10):
            print(f"{classes[i]}", end=" | ")
        print()
        for i in range(10):
            print('%2d %%' % (100 * class_correct[i] / class_total[i]), end=" | ")
        print()
    return acc


if __name__ == '__main__':
    from models.resnet import resnet101 as model

    lr = 0.1

    net = model(pretrained=False, num_classes=10).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)
    epochs = 80
    train(net, optimizer, scheduler, epochs=epochs)
    valid(net, print_class=True)
