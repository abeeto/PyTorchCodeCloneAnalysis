'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import PIL

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from att_dataset import ATTDataset

import Stat_Collector
import logging

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

writer = SummaryWriter("./tensorboard/statistics")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--gpus", default="0", type=str, required=False, help="GPUs id, separated by comma withougt space, for example: 0,1,2")
parser.add_argument("--model_name", default="resnet", type=str, required=True, choices=["resnet", "vgg", "lenet"], help="choose from [resnet, vggm, lenet]")
parser.add_argument("--dataset", default="cifar10", type=str, required=True, choices=["cifar10", "mnist", "svhn", "att"], help="choose from [cifar10, mnist, svhn, att]")
parser.add_argument('--num_example', default=10, type=int, help='The number of examples for collecting intermedia results')
parser.add_argument('--epoch', default=350, type=int, help='The number of epoch for training')
parser.add_argument("--ckpt_dir", default=None, type=str, help="The path to the load checkpoint, start with ./checkpoint")
parser.add_argument("--save_dir", default=None, type=str, help="The path to the save checkpoint, start with ./checkpoint")
args = parser.parse_args()


# Setting up gpu environment
gpus = args.gpus.split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


train_batch_size = 128
test_batch_size = 128
# Data
print('==> Preparing data..')
if args.dataset == "cifar10":
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    stat_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.num_example, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
elif args.dataset == "mnist":
    data_transform = torchvision.transforms.Compose([
                               transforms.Resize((32, 32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

    trainset =  torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=data_transform)
    testset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=data_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    stat_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.num_example, shuffle=False, num_workers=2)
elif args.dataset == "svhn":
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]
    train_transform = torchvision.transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])

    valid_transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])

    trainset =  torchvision.datasets.SVHN('./data/', split="train", download=True,
                                transform=train_transform)
    testset = torchvision.datasets.SVHN('./data/', split="test", download=True,
                                transform=valid_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    stat_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.num_example, shuffle=False, num_workers=2)
elif args.dataset == 'att':
    # data_transform = torchvision.transforms.Compose([
    #                            transforms.Resize((32, 32)),
    #                            torchvision.transforms.ToTensor(),
    #                            torchvision.transforms.Normalize(
    #                              (0.1307,), (0.3081,))
    #                          ])

    data_transform = torchvision.transforms.Compose([
                               transforms.Resize((92, 112)),
                               #transforms.Grayscale(),
                               torchvision.transforms.ToTensor()
                             ])

    trainset_folder =  torchvision.datasets.ImageFolder('./data/att/train')
    trainset =  ATTDataset(imageFolderDataset=trainset_folder, transform = data_transform)

    testset_folder = torchvision.datasets.ImageFolder('./data/att/test')
    testset = ATTDataset(imageFolderDataset=testset_folder, transform = data_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=5, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=5, shuffle=False, num_workers=2)

    stat_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.num_example, shuffle=False, num_workers=2)
else:
    raise NotImplementedError("The specifed dataset is not supported")


# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.model_name == "resnet":
    if args.dataset == "mnist":
        net = ResNet18_Mnist()
    else:
        net = ResNet18()
elif args.model_name == "lenet":
    if args.dataset == "mnist":
        net = LeNet_Mnist()
    elif args.dataset == "att":
        net = LeNet_ATT()
    else:
        net = LeNet()
else:
    net = VGG('VGG19')
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
#net = RegNetX_200MF()
net = net.to(device)

print ("The structure of model:", net)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.ckpt_dir is None:
        raise NotImplementedError("The path to the checkpoint is not specifed")
    else:
        checkpoint = torch.load(args.ckpt_dir)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.dataset == "cifar10":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
elif args.dataset == "mnist":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
elif args.dataset == "svhn":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
elif args.dataset == "att":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
else:
    raise NotImplementedError("The specifed optimizer is not supported")

scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.save_dir is None:
            raise NotImplementedError("The specifed optimizer is not supported")
        else:
            torch.save(state, args.save_dir)
        best_acc = acc

if not args.resume:
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        test(epoch)
        scheduler.step()

logging.info("Collecting the statistics by running test set")
target_module_list = [nn.BatchNorm2d,nn.Linear] # Insert hook after BN and FC
net, intern_outputs = Stat_Collector.insert_hook(net, target_module_list)

cur_example = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(stat_loader):
        cur_example += 1
        if cur_example > 20:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            #cur_example += 1
            break 
        #print ("current example:", cur_example)
        #if cur_example > args.num_example:
        #    break

# Drawing the distribution
for i, intern_output in enumerate(intern_outputs):
    stat_features = intern_output.out_features.view(-1)
    print ("No.", i, " ", intern_output.out_features.shape)
    print ("Numpy No.", i, " ", intern_output.out_features.cpu().data.numpy().shape)
    print ("No.", i, " ", stat_features.shape)
    print ("Numpy No.", i, " ", stat_features.cpu().data.numpy().shape)
    #ploting the distribution
    writer.add_histogram("conv%d" % (i), intern_output.out_features.view(-1).cpu().data.numpy(), bins='tensorflow')



