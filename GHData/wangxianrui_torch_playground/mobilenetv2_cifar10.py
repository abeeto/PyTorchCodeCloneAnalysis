import torch
import os
import time
from tensorboardX import SummaryWriter
from dataset.cifar10 import Cifar10
from model.mobilenetv2 import MobileNetV2Classify

RESUME = True


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


class MobileNetV2Cifar10:
    def __init__(self):
        self.image_shape = 224
        self.batch_size = 64
        self.num_workers = 8
        self.lr_init = 0.1
        self.lr_decay = 0.1
        self.lr_epoch = [50, 90, 120, 140, 150]
        self.max_epoch = 150
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_step = 20
        self.save_step = 500
        self.save_path = 'checkpoint'
        self.writer = SummaryWriter('log')

    def train(self):
        train_set = Cifar10(self.batch_size, self.num_workers)
        train_loader = train_set.get_train_loader()
        model = MobileNetV2Classify(train_set.NUM_CLASSES)
        model = torch.nn.parallel.DataParallel(model.cuda())
        if RESUME:
            file_list = os.listdir(self.save_path)
            file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(self.save_path, x)))
            assert len(file_list) > 0, 'no checkpoint to restore'
            print('restoring checkpoint named {}'.format(file_list[-1]))
            model.load_state_dict(torch.load(os.path.join(self.save_path, file_list[-1])))
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            model.parameters(),
            self.lr_init,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        model.train()
        for epoch in range(self.max_epoch):
            self.adjust_learning_rate(optimizer, epoch)
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()
            for step, (inputs, targets) in enumerate(train_loader, 1):
                data_time.update(time.time() - end)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = self.accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if step % self.log_step == 0:
                    index = epoch * len(train_loader) + step
                    self.writer.add_scalar('train/loss', loss.item(), index)
                    self.writer.add_scalar('train/prec1', prec1.item(), index)
                    self.writer.add_scalar('train/prec5', prec5.item(), index)
                    print('Epoch: [{}][{}/{}]\t'
                          'Lr: {:.6f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, step, len(train_loader), optimizer.param_groups[0]['lr'], batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))
                if step % self.save_step == 0:
                    index = epoch * len(train_loader) + step
                    torch.save(model.state_dict(), 'checkpoint/mobilenetv2_cifar10_{:d}.pth'.format(index))
                    self.eval()

    def eval(self):
        eval_set = Cifar10(self.batch_size, self.num_workers)
        eval_loader = eval_set.get_eval_loader()
        model = MobileNetV2Classify(eval_set.NUM_CLASSES)
        model = torch.nn.parallel.DataParallel(model.cuda())

        # restore
        file_list = os.listdir(self.save_path)
        file_list = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(self.save_path, x)))
        assert len(file_list) > 0, 'no checkpoint to restore'
        print('restoring checkpoint named {}'.format(file_list[-1]))
        model.load_state_dict(torch.load(os.path.join(self.save_path, file_list[-1])))

        criterion = torch.nn.CrossEntropyLoss().cuda()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()
        end = time.time()
        for step, (inputs, targets) in enumerate(eval_loader, 1):
            inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % self.log_step == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    step, len(eval_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

        return top1.avg

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    mobilenetv2_cifar10 = MobileNetV2Cifar10()
    mobilenetv2_cifar10.train()
