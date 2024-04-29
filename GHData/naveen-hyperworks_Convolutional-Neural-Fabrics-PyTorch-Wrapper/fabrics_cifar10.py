### import torch stuffs
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn

### import other stuffs
from neural_fabrics import *
from fabric_utils import *

import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

experiment = 'type_a_8x6_128_'
train_loss_file = open("/shared/fusor/home/suriya/cnf/pytorch_fabrics/logs/"+experiment+"train_loss.txt", "w", 0)
val_loss_file = open("/shared/fusor/home/suriya/cnf/pytorch_fabrics/logs/"+experiment+"val_loss.txt", "w", 0)

num_classes = 10
scales = 6
classifier_scale = scales -1      ### add classifier at last scale
channels = [128, 128, 128, 128, 128, 128]


### Exp1 fabric unit = a, layers = 8 (Saxena and Verbeek, NIPS, 2016). Acc = 93.79, GPU = 1979, Params (M) = 18.006666
layers = 8
net = ConvolutionalNeuralFabrics(layers = layers, scales = scales, channels = channels,
                                 num_classes = num_classes, classifier_scale = classifier_scale, fabric_unit='a',
                                 block='conv')

### Exp2 fabric unit = d, layers = 8. Acc = 93.55, GPU = 2219, Params (M) = 22.430346
# layers = 8
# net = ConvolutionalNeuralFabrics(layers = layers, scales = scales, channels = channels,
#                                  num_classes = num_classes, classifier_scale = classifier_scale, fabric_unit='d',
#                                  block='conv')

### Exp3 fabric unit = a, layers = 3, block = post activation bottleneck. Acc = 88.06, GPU = 1735, Params (M) = 0.756618
# layers = 3
# net = ConvolutionalNeuralFabrics(layers = layers, scales = scales, channels = channels,
#                                  num_classes = num_classes, classifier_scale = classifier_scale, fabric_unit='a',
#                                  block='bottleneck', activations='post')

### Exp4 fabric unit = a, layers = 3, block = pre activation bottleneck. Acc = 87.23, GPU = 1239, Params (M) = 0.752266
# layers = 3
# net = ConvolutionalNeuralFabrics(layers = layers, scales = scales, channels = channels,
#                                  num_classes = num_classes, classifier_scale = classifier_scale, fabric_unit='a',
#                                  block='bottleneck', activations='pre')

### Exp5 fabric unit = a, layers = 3, block = pre activation bottleneck, gating = scaling. Acc = 90.04, GPU = 1239, Params (M) = 0.752266 
# layers = 3
# net = ConvolutionalNeuralFabrics(layers = layers, scales = scales, channels = channels,
#                                  num_classes = num_classes, classifier_scale = classifier_scale, fabric_unit='a',
#                                  block='bottleneck', activations='pre', gating = 'scaling')

### Exp6 fabric unit = a, layers = 3, block = pre activation bottleneck, gating = 1x1. Acc = 90.01, GPU = 1435, Params (M) = 1.451146
# layers = 3
# net = ConvolutionalNeuralFabrics(layers = layers, scales = scales, channels = channels,
#                                  num_classes = num_classes, classifier_scale = classifier_scale, fabric_unit='a',
#                                  block='bottleneck', activations='pre', gating = '1x1')




net.cuda()
print net

print 'Net params count (M): ', param_counts(net)/(1000000.0)

### apply mirroring and random crop on training images
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

trainset = torchvision.datasets.CIFAR10(root='/tmp/suriya/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='/tmp/suriya/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        size_ = outputs.size()
        outputs_ = outputs.view(size_[0], num_classes)
        loss = criterion(outputs_, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs_.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    train_loss_file.write('%d %.3f %.3f\n' %(epoch, train_loss/len(trainloader), 100.*correct/total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        size_ = outputs.size()
        outputs_ = outputs.view(size_[0], num_classes)
        loss = criterion(outputs_, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs_.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    print 'val_loss: ',  test_loss/len(testloader), 'accuracy: ', 100.0*correct/total
    val_loss_file.write('%d %.3f %.3f\n' %(epoch,  test_loss/len(testloader), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/shared/fusor/home/suriya/cnf/pytorch_fabrics/checkpoint'):
            os.mkdir('/shared/fusor/home/suriya/cnf/pytorch_fabrics/checkpoint')
        torch.save(state, '/shared/fusor/home/suriya/cnf/pytorch_fabrics/checkpoint/'+experiment+'ckpt.t7')
        best_acc = acc


for epoch in range(0, 120):
    if epoch == 80:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif epoch == 60:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif epoch == 0:
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    
    train(epoch)
    test(epoch)
    
train_loss_file.close()
val_loss_file.close()