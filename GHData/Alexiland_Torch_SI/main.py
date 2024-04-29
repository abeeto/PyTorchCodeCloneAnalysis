'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from SI.SI import image_complexity as SI

import torchvision
import torchvision.transforms as transforms

import numpy as np
import csv

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device= 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.ImageFolder(
    root='/data5/ILSVRC/Data/CLS-LOC/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.ImageFolder(
    root='/data5/ILSVRC/Data/CLS-LOC/val', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)

#classes = ('plane', 'car', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, file):
    print('\nEpoch: %d' % epoch)
    net.train()
    writer = csv.writer(file)
    train_loss = 0
    correct = 0
    total = 0
    writer.writerow(["image_name", "image_name_copy", "image_complexity"])
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print("Train Size: ")
        print(SI(inputs).size())
        print("Train tensor: ")
        si_score = SI(inputs).tolist()
        tar = targets.tolist()
        print(si_score)
        for idx in range(len(si_score) - 1):
            writer.writerow([str(batch_idx) + "_"+ str(idx), tar[idx], si_score[idx]])
        # np.savetxt(fname=str(epoch)+".csv", X=1, delimiter=",")
        # inputs, targets = inputs.to(device), targets.to(device)
        # optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()
        #
        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        #
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, file):
    global best_acc
    net.eval()
    writer = csv.writer(file)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        writer.writerow(["image_name", "image_name_copy", "image_complexity"])
        for batch_idx, (inputs, targets) in enumerate(testloader):
            print("Train Size: ")
            print(SI(inputs).size())
            print("Train tensor: ")
            si_score = SI(inputs).tolist()
            tar = targets.tolist()
            print(si_score)
            for idx in range(len(si_score) - 1):
                writer.writerow([str(batch_idx) + "_"+ str(idx), tar[idx], si_score[idx]])
            # inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs)
            # loss = criterion(outputs, targets)
            #
            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            #
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+10):
        file_train = open("/home/hg31/work/Torch_SI/csv/ImageNet_" +
                          str(epoch) + ".train_rndCropFlip.SI.train"+".csv", "a")
        file_test = open("/home/hg31/work/Torch_SI/csv/ImageNet_" + str(
            epoch) + ".test.SI.test" + ".csv", "a")
        train(epoch, file_train)
        test(epoch, file_test)
        file_train.close()
        file_test.close()
