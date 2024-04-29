import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out


########################################################################################################################
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 10, 3, 1, padding=1)
        self.conv1_2 = nn.Conv2d(10, 10, 3, 1, padding=1)
        self.conv2_1 = nn.Conv2d(10, 20, 3, 1, padding=1)
        self.conv2_2 = nn.Conv2d(20, 20, 3, 1, padding=1)
        self.conv3_1 = nn.Conv2d(20, 40, 3, 1, padding=1)
        self.conv3_2 = nn.Conv2d(40, 40, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(640, 128)
        # self.dropout_f1 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.dropout1(x)
        #
        x = F.max_pool2d(x, 2)
        #
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.dropout2(x)
        #
        x = F.max_pool2d(x, 2)
        #
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.dropout3(x)
        #
        x = F.max_pool2d(x, 2)
        #
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = self.dropout_f1(x)
        x = self.fc2(x)
        #
        out = F.log_softmax(x, dim=1)
        return out

########################################################################################################################
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(batch_y.view_as(pred)).sum().item()
        loss = F.nll_loss(output, batch_y)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        # print('loss = ', loss.item())
        # print('grad = ', model.fc2.weight.grad)
        # sys.exit(0)

        train_loss += last_loss
        if batch_idx % 10 == 0:
            print('*', end='')
            # print('Train epoch {} [{}/{}]  loss = {}'.format(
            #     epoch, batch_idx * len(batch_x), len(train_loader.dataset), last_loss
            # ))
    print('')
    train_loss /= len(train_loader.dataset)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc


########################################################################################################################
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            test_loss += F.nll_loss(output, batch_y, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(batch_y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = correct * 100. / len(test_loader.dataset)
    return test_loss, test_acc


########################################################################################################################
def main():
    use_cuda = True
    batch_size = 64
    num_epochs = 100

    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load dataset
    my_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=my_transforms),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=my_transforms),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    model = Net2().to(device)
    print(f'model = {model}')

    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        print('TRAIN EPOCH {}  loss = {}, acc = {}, test_loss = {}, test_acc = {}'.format(
            epoch, train_loss, train_acc, test_loss, test_acc))


########################################################################################################################
if __name__ == '__main__':
    main()
