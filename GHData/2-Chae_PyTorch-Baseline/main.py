import os
import argparse
import random
import warnings
import timm
import wandb
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import horovod.torch as hvd


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def run(args):
    # Data loading (ImageNet)
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    num_classes = 1000   
    
    if args.distributed:
        print('DISTRIBUTED!!')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=hvd.size(),
                                                                        rank=hvd.rank())
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, 
                                                                       num_replicas=hvd.size(), 
                                                                       rank=hvd.rank())
        
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # wandb
    if hvd.rank() == 0:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(f'runs/{current_time}')
        os.makedirs(log_dir, exist_ok=True)
        run = wandb.init(project=args.project, name=current_time, dir=log_dir)
        run.config.update(args)
    else:
        run = None


    # Model creation
    if args.pretrained:
        model = timm.create_model(args.arch, pretrained=True, num_classes=num_classes)
    else:
        model = timm.create_model(args.arch, num_classes=num_classes)

    model = model.cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters())

    hvd.broadcast_parameters(
        model.state_dict(),
        root_rank=0)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, train_sampler, epoch, args)
        val_loss, val_acc = validate(val_loader, model, criterion)
        
        if hvd.rank() == 0:
            run.log({"train": {"loss": train_loss, "acc": train_acc}, "val":{"loss": val_loss, "acc": val_acc}})

def train(train_loader, model, criterion, optimizer, sampler, epoch, args):
    train_loss = Metric('train_loss')
    train_acc = Metric('train_acc')
    
    model.train()
    sampler.set_epoch(epoch) # Horovod: set epoch to sampler for shuffling.
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
 
        optimizer.zero_grad()
        logits = model(data)
        step_loss = criterion(logits, target)
        train_loss.update(step_loss)
        
        step_loss.backward()
        optimizer.step()
        
        pred = logits.max(1, keepdim=True)[1]
        pred = pred.eq(target.view_as(pred)).cpu().float().mean()
        
        train_acc.update(pred)
        
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(sampler),
                100. * batch_idx / len(train_loader), step_loss.item()))

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            train_loss.avg, 100. * train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion):
    val_loss = Metric('val_loss')
    val_acc = Metric('val_acc')
    
    model.eval()
    for data, target in val_loader:
 
        data = data.cuda()
        target = target.cuda()
 
        logits = model(data)
        step_loss = criterion(logits, target)
 
        val_loss.update(step_loss)
        
        pred = logits.max(1, keepdim=True)[1]
        pred = pred.eq(target.view_as(pred)).cpu().float().mean()
        val_acc.update(pred)
 
    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            val_loss.avg, 100. * val_acc.avg))
    return val_loss.avg, val_acc.avg

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()
    
def setup(args):
    if args.distributed:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('PyTorch Baseline Code')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=timm.list_models(),
                    help='model architecture: ' +
                        ' | '.join(timm.list_models()) +
                        ' (default: resnet18)')
    parser.add_argument('-d', '--distributed', action='store_true',
                    help='if run with distributed version')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')                    
    parser.add_argument('-p', '--pretrained', action='store_true',
                    help='if use pretrained model')
    
    # wandb logger
    parser.add_argument('--wandb', action='store_true', help='if use wandb logger')
    parser.add_argument('--project', default='Toy Exp', type=str, help='wandb project name')
    
    args = parser.parse_args()

    setup(args)
    run(args)