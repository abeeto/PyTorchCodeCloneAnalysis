from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.datasets.cifar import CIFAR10
from utils import progress_bar, summary
from models import HiResC
from resnet20 import ResNet, BasicBlock
# from hardcodedmodels import *
from torch.optim.lr_scheduler import MultiStepLR

try_no = ['r20-swish', 'r20-relu', 'r20-elu', 'r20-lrelu', 'r20-linear']
activate = ['swish', 'relu', 'elu', 'lrelu', 'linear']
# mods = [HiResC([18, 18, 18]), ResNet(BasicBlock, [18, 18, 18])]

if __name__ == '__main__':
    best_acc = 0
    start_epoch = 1
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

    trainset = CIFAR10(root='/home/palm/PycharmProjects/DATA/cifar10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = CIFAR10(root='/home/palm/PycharmProjects/DATA/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    device = 'cuda'
    for elm in range(len(try_no)):
        log = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
        n = try_no[elm]
        model = ResNet(BasicBlock, [2, 2, 2], activation=activate[elm])
        # model = HiResC(1)
        model = torch.nn.DataParallel(model).cuda()
        # summary((3, 32, 32), model)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 1e-2,
                                    momentum=0.9,
                                    weight_decay=1e-6, nesterov=True)
        # first_scheduler = MultiStepLR(optimizer, milestones=[2], gamma=10)
        scheduler = MultiStepLR(optimizer, milestones=[100, 175])
        cudnn.benchmark = True

        def train(epoch):
            print('\nEpoch: %d' % epoch)
            # first_scheduler.step()
            scheduler.step()
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'loss: %.3f, acc: %.3f%%'
                    % (train_loss/(batch_idx+1), 100.*correct/total))
            log['acc'].append(100.*correct/total)
            log['loss'].append(train_loss/(batch_idx+1))

        def test(epoch):
            global best_acc
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader), 'loss: %.3f, acc: %.3f%%'
                        % (test_loss/(batch_idx+1), 100.*correct/total))
            log['val_acc'].append(100.*correct/total)
            log['val_loss'].append(test_loss/(batch_idx+1))

            # Save checkpoint.
            acc = 100.*correct/total
            if acc > best_acc:
                print('Saving..')
                state = {
                    'net': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.t7')
                best_acc = acc


        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)
        with open('log/try_{}.json'.format(n), 'w') as wr:
            wr.write(log.__str__())
