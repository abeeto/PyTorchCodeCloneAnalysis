import argparse
from utils import *
from model import Model
import os
import infolog
import shutil
import time
import warnings
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pickle

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.datasets import AudiobookDataset, collate
log = infolog.log


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x_inputs, mels, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            x_inputs = x_inputs.cuda(args.gpu)
            mels = mels.cuda(args.gpu)
            target = target.cuda(args.gpu)

        import pdb
        pdb.set_trace()
        # compute output
        outputs = model(x_inputs, mels)
        loss = criterion(outputs, target)
        losses.update(loss.item(), x_inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                batch_time=batch_time, data_time=data_time,
                                                                loss=losses, top1=top1, top5=top5
                                                                )
                )


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x_inputs, mels, target) in enumerate(val_loader):
            if args.gpu is not None:
                x_inputs = x_inputs.cuda(args.gpu)
                target = target.cuda(args.gpu)

            # compute output
            outputs = model(x_inputs)
            loss = criterion(outputs, target)
            losses.update(loss.item(), x_inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                log('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader),
                                                                    batch_time=batch_time, loss=losses,
                                                                    top1=top1, top5=top5
                                                                    )
                    )

        log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

model_names = ['Fachord', 'MlWoo']


def main():
    parser = argparse.ArgumentParser(description='WaveRNN-PyTorch Training')
    parser.add_argument('--data_dir', metavar='DIR', default='dataset',
                        help='path to dataset')
    parser.add_argument('--base_dir', metavar='DIR', default='/home/wumenglin/workspace/WaveRNN_pytorch',
                        help='path to workspace')
    parser.add_argument('--model_path', metavar='DIR', default='',
                        help='path to workspace')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Fachord',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: Fachord)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-q', '--quant_bits', default=9, type=int,
                        metavar='N', help='quantilization bits (default: 9)')
    parser.add_argument('--test_batches', default=12, type=int,
                        metavar='N', help='quantilization bits (default: 9)')

    global args, best_prec1
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        log("The system set the random number to:{}".format(args.seed))

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = False #args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
    import pdb
    pdb.set_trace()
    # create model
    model = Model(rnn_dims=512, fc_dims=512, bits=args.quant_bits, pad=2,
                  upsample_factors=(5, 5, 11), feat_dims=80,
                  compute_dims=128, res_out_dims=128, res_blocks=10).cuda(args.gpu)
    
    if args.resume and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        step = np.load(args.step_path)


    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            log("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    dataset_dir = os.path.join(args.base_dir, args.data_dir)

    ids = os.path.join(dataset_dir, 'dataset_ids.pkl')
    with open(ids, 'rb') as f:
        dataset_ids = pickle.load(f)

    test_size = args.test_batches * args.batch_size
    indices = np.arange(len(dataset_ids))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=args.seed)

    train_dataset_ids = [dataset_ids[i] for i in train_indices]

    train_dataset = AudiobookDataset(train_dataset_ids, dataset_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=args.batch_size,
                              num_workers=2, shuffle=True, pin_memory=True)
    #import pdb
    #pdb.set_trace()

    val_dataset_ids = [dataset_ids[i] for i in val_indices]

    val_dataset = AudiobookDataset(val_dataset_ids, dataset_dir)
    val_loader = DataLoader(val_dataset, collate_fn=collate, batch_size=args.batch_size,
                            num_workers=2, shuffle=True, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    main()
