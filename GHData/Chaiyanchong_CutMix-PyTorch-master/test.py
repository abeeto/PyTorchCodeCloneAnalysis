# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM

import warnings

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from models.experimental import attempt_load
from models.yolo import Model
from utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)


warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Test')
parser.add_argument('--net_type', default='yolov5n', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='LSD', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--pretrained', default='E:/DLPROJTCT/CutMix-PyTorch-master/runs/TEST/model_best.pth.tar', type=str, metavar='PATH')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

class MyDataset(Dataset):
    def __init__(self, root, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(root+datatxt,'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split('\t')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self ,index):
        fn,label = self.imgs[index]
        #img = Image.open(root+fn)
        img = cv2.imread(fn)
        #print(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        PIL_IMG=Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(PIL_IMG)
        #img = img.transpose(2, 0, 1)
        return img,label,fn

    def __len__(self):
        return len(self.imgs)


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':

        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        numberofclass = 1000

    elif args.dataset == 'LSD':
        transform_train = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.0],
             #                    std=[0.3922])
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),#(H,W,C)转换为（C,H,W）,取值范围是0-1
            #transforms.Normalize(mean=[0.0],
            #                     std=[0.3922])
        ])
        trainset = MyDataset(root='G:/DataSet/LSD/', datatxt='train.txt',transform=transform_train)
        train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
        testset = MyDataset(root='G:/DataSet/LSD/', datatxt='val.txt', transform=transform_test)
        val_loader = DataLoader(testset, batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)

    elif args.net_type == 'yolov5n':
        cfg = 'E:/DLPROJTCT/CutMix-PyTorch-master/models/yolov5n_class.yaml'
        #weights = 'E:/DLPROJTCT/CutMix-PyTorch-master/runs/TEST/model_best.pth.tar'
        model = Model(cfg, ch=1, nc=2)
        #model.load_state_dict(weights, strict=False)  # load
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.pretrained))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.pretrained))

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # evaluate on validation set
    err1, err5, val_loss = validate(val_loader, model, criterion)

    print('Accuracy (top-1 and 5 error):', err1, err5)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target,paths) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        sm=nn.Softmax(dim=1)
        cls_res=sm(output)
        #print(cls_res[0])
        #save result to txt
        import numpy as np
        a = np.array(cls_res.data.cpu())
        t = np.array(target.data.cpu())
        resulttxt_path='G:/DataSet/LSD/cutmix_result.txt'
        with open(resulttxt_path, 'a+') as f:
            id=0
            for path in paths:
                f.write((path + '\t' + str(a[id][0]) + '\t' + str(a[id][1]) + '\t' +str(t[id]) + '\n'))
                if a[id][0]>a[id][1] and t[id]==1:
                    with open('G:/DataSet/LSD/night2day.txt', 'a+') as f1:
                        f1.write((path + '\t' + str(a[id][0]) + '\t' + str(a[id][1]) + '\t' + str(t[id]) + '\n'))
                if a[id][0]<a[id][1] and t[id]==0:
                    with open('G:/DataSet/LSD/day2night.txt', 'a+') as f2:
                        f2.write((path + '\t' + str(a[id][0]) + '\t' + str(a[id][1]) + '\t' + str(t[id]) + '\n'))
                id=id+1


        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 1))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
