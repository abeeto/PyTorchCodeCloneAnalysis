import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from model.model import Net
from model.utils import get_dataloader, get_dataset, get_transform

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--bs', default=32, type=int)
    return parser.parse_args()


def train(model, trainloader, optimizer, loss_function, device):
    model.train()
    running_loss = 0
    for i, (input, target) in enumerate(trainloader, 0):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(input)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss/len(trainloader.dataset)
    return total_loss


def test(model, testloader, loss_function, device):
    model.eval()
    test_loss = 0
    correct   = 0
    with torch.no_grad():
        for idx, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            test_loss += loss_function(output, target).item()
            
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()
    
    test_loss /= len(testloader)
    test_accuracy = 100. * correct / len(testloader.dataset)
    return test_loss, test_accuracy

if __name__ == '__main__':
    opt = get_args()
    epochs, lr, momentum, bs = opt.epoch, opt.lr, opt.momentum, opt.bs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set, test_set = get_dataset(transform=get_transform())
    trainloader, testloader = get_dataloader(train_set, test_set, bs)

    model = Net().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer     = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # training
    train_losses, test_losses, test_accuracy = [], [], []
    best_val_loss = 100

    for epoch in range(epochs):
        train_loss = train(model, trainloader, optimizer, loss_function, device)
        train_losses.append(train_loss)

        test_loss, test_acc = test(model, testloader, loss_function, device)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)

        print(f'Epoch {epoch+1} | Train loss: {train_loss} | Test loss: {test_loss}')

        if best_val_loss > test_loss:
            print('Model saved')
            torch.save(model.state_dict(), './model/weights/net.pt')
