import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from model.resnext import ResNeXt

import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

def main():
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    net = ResNeXt(3, 32**2, 10, 256, 10, 4, 32).to(device)

    opt = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    for epoch in range(1, EPOCHS + 1):
        print('[Epoch %d]' % epoch)
        
        train_loss = 0
        train_correct, train_total = 0, 0

        start_point = time.time()

        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            opt.zero_grad()

            preds = F.log_softmax(net(inputs), dim=1)
            
            loss = F.cross_entropy(preds, labels)
            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)

        print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader)))
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0, 0

        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

                preds = F.softmax(net(inputs), dim=1)

                test_loss += F.cross_entropy(preds, labels).item()

                test_correct += (preds.argmax(dim=1) == labels).sum().item()
                test_total += len(preds)

        print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader)))
        
        torch.save(net.state_dict(), './checkpoint/checkpoint-%04d.bin' % epoch)

if __name__ == '__main__':
    main()
