import sys

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
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
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
    train_mode = False
    batch_size = 64
    num_epochs = 5

    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'device = {device}')
    print(f'torch.__version__ = {torch.__version__}')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load dataset
    my_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True, transform=my_transforms),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True, transform=my_transforms),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    model = Net1().to(device)
    print(f'model = {model}')

    print('Model state_dict :')
    for vn in model.state_dict():
        print(vn, '  ', model.state_dict()[vn].size())

    if train_mode:
        optimizer = torch.optim.Adam(model.parameters())
        print('Optimizer state_dict :')
        for vn in optimizer.state_dict():
            print(vn, '  ', optimizer.state_dict()[vn])

        # Train
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
            model.eval()
            test_loss, test_acc = test(model, device, test_loader)
            print('TRAIN EPOCH {}  loss = {}, acc = {}, test_loss = {}, test_acc = {}'.format(
                epoch, train_loss, train_acc, test_loss, test_acc))
        # Save model
        torch.save(model.state_dict(), 'my_model.pt')
    else:
        # Load model
        model.load_state_dict(torch.load('my_model.pt'))
        model.eval()

    # Convert to torchscript
    if False:
        xx = torch.rand(batch_size, 1, 28, 28, device=device)
        ts_model = torch.jit.trace(model, xx)
        ts_model.save('ts_model.pt')
    else:
        ts_model = torch.jit.load('ts_model.pt')

    print(f'ts_model = {ts_model}')

    # Final test on "real model"
    model.eval()
    print(f'is_cuda = {next(model.parameters()).is_cuda}')
    test_loss, test_acc = test(model, device, test_loader)
    print('\nTEST MODEL: test_loss = {}, test_acc = {}'.format(test_loss, test_acc))

    # Final test on ts model
    print(f'is_cuda (ts) = {next(ts_model.parameters()).is_cuda}')
    test_loss, test_acc = test(ts_model, device, test_loader)
    print('\nTEST TS_MODEL: test_loss = {}, test_acc = {}'.format(test_loss, test_acc))

    # Infer ones on the "REAL" model
    ones_in = torch.ones(1, 1, 28, 28, device=device)
    ones_out = model(ones_in)
    print(f'Infer ones : out = {ones_out}')


########################################################################################################################
if __name__ == '__main__':
    main()
