# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import argparse
import logging
import numpy as np
# from tensorboardX import SummaryWriter
import math
import sys
import os
import shutil
from torch_ACA import odesolve_endtime as odesolve

from data_helper import get_galaxyzoo_loaders, get_mtvso_loaders, tiny_imagenet, get_mnist_loaders

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from plots import plot_roc, precision_recall
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,4,5'



parser = argparse.ArgumentParser()
parser.add_argument('--network', type = str, choices = ['resnet', 'sqnxt', 'pytorch_resnet50_single'], default = 'sqnxt')
parser.add_argument('--method', type = str, choices=['Euler', 'RK2', 'RK4','RK23','RK45','RK12','Dopri5'], default = 'RK12')
parser.add_argument('--dataset', type = str, choices = ['cifar10', 'galaxyzoo', 'mtvso', 'tiny_imagenet', 'mnist'], default = 'galaxyzoo')
parser.add_argument('--dataset_type', type = str, choices = ['anp', 'adj', 'noun'], default = 'anp')
parser.add_argument('--dataset_source', type = str, choices = ['local', 'server_main', 'server_nilesh', 'server_other'], default = 'server_other')
parser.add_argument('--dataset_size', type = str, choices = ['small', 'smallFull', 'normal', 'large'], default = 'normal')
parser.add_argument('--num_epochs', type = int, default = 15)
parser.add_argument('--start_epoch', type = int, default = 0)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_path', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--h', type=float, default = None, help='Initial Stepsize')
parser.add_argument('--t0', type=float, default = 0.0, help='Initial time')
parser.add_argument('--t1', type=float, default = 1.0, help='End time')
parser.add_argument('--rtol', type=float, default = 1e-2, help='Releative tolerance')
parser.add_argument('--atol', type=float, default = 1e-2, help='Absolute tolerance')
parser.add_argument('--print_neval', type=bool, default = False, help='Print number of evaluation or not')
parser.add_argument('--neval_max', type=int, default = 50000, help='Maximum number of evaluation in integration')

parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--test_batch_size', type = int, default = 32)
parser.add_argument('--crop_type', type = str, choices = ['center','random'], default = 'center')
parser.add_argument('--crop_size', type = int, default = 32)
parser.add_argument('--resize', type = int, default = 32)
parser.add_argument('--aug_dim', type = int, default = 10)
args = parser.parse_args()
if args.network == 'sqnxt':
    from cifar_classification.models.sqnxt import SqNxt_23_1x
    # writer = SummaryWriter('sqnxt/'+args.network +'_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')
elif args.network == 'resnet':
    from cifar_classification.models.resnet import ResNet18
    # writer = SummaryWriter('resnet/'+args.network+'_'+ args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')


num_epochs = int(args.num_epochs)
lr           = float(args.lr)
start_epoch  = int(args.start_epoch)
batch_size   = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > .8 * num_epochs:
        optim_factor = 2
    elif epoch > .6 * num_epochs:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'method':args.method})
        self.options.update({'h': args.h})
        self.options.update({'t0': args.t0})
        self.options.update({'t1': args.t1})
        self.options.update({'rtol': args.rtol})
        self.options.update({'atol': args.atol})
        self.options.update({'print_neval': args.print_neval})
        self.options.update({'neval_max': args.neval_max})
        print(self.options)

    def forward(self, x):
        out = odesolve(self.odefunc, x, self.options, regenerate_graph = False)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        

# Data Preprocess
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_train, train = True, download = True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_test, train = False, download = True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 4, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.test_batch_size, num_workers = 4, shuffle = False)

# if args.crop_type == 'center':
#     transform_crop = transforms.CenterCrop(args.crop_size)
# elif args.crop_type == 'random':
#     transform_crop = transforms.RandomCrop(args.crop_size, padding=4)

# transform_train = transforms.Compose([
#         transform_crop,
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

# transform_test = transforms.Compose([
#     transforms.CenterCrop(args.crop_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
elif args.dataset == 'galaxyzoo':
    train_loader, test_loader, train_dataset = get_galaxyzoo_loaders(batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        dataset_size=args.dataset_size, resize=args.resize, network=args.network, dataset_type=args.dataset_type,
        dataset_source=args.dataset_source)
elif args.dataset == 'tiny_imagenet':
    train_loader, test_loader, train_dataset = tiny_imagenet(batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        resize=args.resize, dataset_source=args.dataset_source)
elif args.dataset == 'mnist':
    train_loader, test_loader, train_dataset = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=args.test_batch_size)

if args.dataset == 'mtvso':
    # if args.dataset_size=='normal':
    #     num_classes = [100, 78, 79]
    # elif args.dataset_size=='large':
    #     num_classes = [581, 123, 221]
    # elif args.dataset_size=='small':
    #     num_classes = 10
    if args.network == 'pytorch_resnet50_single':
        if args.dataset_type == 'anp':
            num_classes=num_classes[0]
        elif args.dataset_type == 'adj':
            num_classes=num_classes[1]
        elif args.dataset_type == 'noun':
            num_classes=num_classes[2]
elif args.dataset == 'galaxyzoo':
    num_classes = 5
elif args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'tiny_imagenet':
    num_classes = 200
elif args.dataset == 'mnist':
    num_classes = 10

if args.dataset == 'mnist':
    num_channels = 1
else:
    num_channels = 3

"""here is model definition fff"""

if args.dataset == 'mnist':
    num_channels = 1
else:
    num_channels = 3

"""here is model definition fff"""

if args.network == 'sqnxt':
    net = SqNxt_23_1x(num_classes, ODEBlock)
elif args.network == 'resnet':
    net = ResNet18(ODEBlock, device, num_classes=num_classes, augment_dim=args.aug_dim, num_channels=num_channels)
elif args.network == 'pytorch_resnet50_single':
    # if args.dataset_type == 'anp':
    #     num_classes=num_classes[0]
    # elif args.dataset_type == 'adj':
    #     num_classes=num_classes[1]
    # elif args.dataset_type == 'noun':
    #     num_classes=num_classes[2]

    net = models.resnet34(pretrained=False)
    # net = models.densenet201(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    net.fc = nn.Sequential(
               nn.Linear(512, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, num_classes))

net.apply(conv_init)


print(net)
if is_use_cuda:
    net = nn.DataParallel(net)
    net.to(device)#to(device)
criterion = nn.CrossEntropyLoss()
optimizer  = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 5e-4)

def train(epoch):
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    
    print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        # import pdb; pdb.set_trace()
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()

        optimizer.step()
        # writer.add_scalar('Train/Loss', loss.item(), epoch* 50000 + batch_size * (idx + 1)  )
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Loss: %.4f Acc@1: %.3f\n'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, 
                           # idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), correct / total))
        sys.stdout.flush()
    # writer.add_scalar('Train/Accuracy', correct / total, epoch )
        
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_test =[]
    y_score =[]

    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if idx==0:
            a=labels.detach().cpu() # all labels.cpu()
            b=outputs.detach().cpu() # all outputs.cpu()
        else:
            a=np.concatenate((a,labels.detach().cpu())) # a actual class
            b=np.concatenate((b,outputs.detach().cpu())) # b predicted class
        # print('outputs',outputs)
        # print('labels',labels)
        # if epoch==(start_epoch + num_epochs-1):
        #     precision_recall(labels.detach().cpu(), outputs.detach().cpu(),5)
        
        test_loss  += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        # writer.add_scalar('Test/Loss', loss.item(), epoch* 50000 + test_loader.batch_size * (idx + 1)  )

        # --------------------------------------------------------
        target = F.one_hot(labels,num_classes)
        y_test.append(target.cpu().detach().numpy())
        y_score.append(outputs.cpu().detach().numpy())
        # --------------------------------------------------------
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch  [%d/%d] Loss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, test_loss / (100 * (idx + 1)), correct / total))
        sys.stdout.flush()

    # if epoch==(start_epoch + num_epochs-1):
    from datetime import datetime
    now_str=str(datetime.now().date())+'_'+str(datetime.now().time())
    if not os.path.isfile('../node_aca_outputs_arrays_'+now_str):
        open('node_aca_outputs_arrays_'+now_str,'a').close()

    with open('node_aca_outputs_arrays_'+now_str,'wb') as fpw:
        import pickle; pickle.dump([a, b, 5],fpw)

    precision_recall(a, b,5)

    acc = correct / total
    # writer.add_scalar('Test/Accuracy', acc, epoch )
    return acc, y_test, y_score

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

best_acc = 0.0

if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if args.resume_path:
            resume_path_dir = args.resume_path
            resume_path = os.path.join(resume_path_dir, 'checkpoint.pth.tar')
            assert os.path.isfile(resume_path), 'Error: no directory found for resume_path!'
        else:
            resume_path_dir = parser.get_default('checkpoint')+'_' \
                +args.network+'_'+args.method+'_'+args.dataset
            resume_path = os.path.join(resume_path_dir, 'checkpoint.pth.tar')
            print('looking for ',resume_path, ' as **DEFAULT** dir')
            assert os.path.isfile(resume_path), 'Error: no directory found using resume_path! **DEFAULT**'
        args.checkpoint = os.path.dirname(resume_path)
        checkpoint = torch.load(resume_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()

    _lr = lr_schedule(args.lr, _epoch)
    adjust_learning_rate(optimizer, _lr)

    # train(_epoch)
    print()
    test_acc, y_test, y_score = test(_epoch)
    print()
    print()
    end_time   = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

    # save model
    is_best = test_acc > best_acc

    if args.checkpoint != './checkpoint':
        checkpoint = args.checkpoint 
    else:
        checkpoint = args.checkpoint+'_'+args.network+'_'+args.method+ \
            '_'+args.dataset

    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': _epoch + 1,
        'state_dict': net.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=checkpoint)

    
    # -----------------------------------------------------------------
    if _epoch == (start_epoch + num_epochs -1):
        print('********** last epoch. calculating roc and precision ********')
        y_test=[np.array(x) for x in y_test]
        y_score=[np.array(x) for x in y_score]
        plot_roc(y_test, y_score, num_classes)
        precision_recall(y_test, y_score, num_classes)
    # -----------------------------------------------------------------


print('Best Acc@1: %.4f' % (best_acc * 100))
# writer.close()
