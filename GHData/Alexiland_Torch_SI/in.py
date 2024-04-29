from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from SI.SI import image_complexity as SI
from SI.kmeans import kmeans_cluster, sort_cluster
from models.preresnet import resnet
from ut.loss import SegmentationLosses
import math
import random
import csv


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

downsampling_ratio_min = 0.5
downsampling_ratio_max = 1.0
keep_ratio_max = 1.0
keep_ratio_min = 0.0
loss_scale_max = 1.0
loss_scale_min = 0.0
crop_size = 32
nclass = 10


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           # transforms.Pad(4),
                           # transforms.RandomCrop(32),
                           # transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()
                           # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           # transforms.Pad(4),
                           # transforms.RandomCrop(32),
                           # transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()
                           # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

# model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
model = resnet(20, dataset=args.dataset)
if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

history_score = np.zeros((args.epochs - args.start_epoch + 1, 3))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch, file):
    train_loss = 0.0
    model.train()
    loss = 0
    global history_score
    avg_loss = 0.
    train_acc = 0.
    writer = csv.writer(file)
    writer.writerow(["image_name", "image_name_copy", "image_complexity"])
    for i, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # print("Train Size: ")
        # print(SI(data).size())
        # print("Train tensor: ")
        si_score = SI(data).tolist()
        tar = target.tolist()
        # print(si_score)
        for idx in range(len(si_score) - 1):
            writer.writerow([str(i) + "_" + str(idx), tar[idx], si_score[idx]])
        # # data, target = Variable(data), Variable(target)
        # complexity = SI(data)
        # # print("complexity: ")
        # # print(complexity)
        # projected_complexity = project_IC_func(complexity)
        # # cluster into three group
        # indices_list, mean_list = cluster_func(projected_complexity)
        # [indices_simple, indices_middle, indices_hard] = indices_list
        # [mean_simple, mean_middle, mean_hard] = mean_list
        # downsampling_size_simple, keep_ratio_simple, loss_scale_simple = project_hyperparam(mean_simple)
        # downsampling_size_middle, keep_ratio_middle, loss_scale_middle = project_hyperparam(mean_middle)
        # downsampling_size_hard, keep_ratio_hard, loss_scale_hard = project_hyperparam(mean_hard)
        #
        # images_simple = torch.index_select(data, 0, indices_simple)
        # images_middle = torch.index_select(data, 0, indices_middle)
        # images_hard = torch.index_select(data, 0, indices_hard)
        #
        # target_simple = torch.index_select(target, 0, indices_simple)
        # target_middle = torch.index_select(target, 0, indices_middle)
        # target_hard = torch.index_select(target, 0, indices_hard)
        #
        # weight_dict = {"simple": loss_scale_simple, "middle": loss_scale_middle, "hard": loss_scale_hard}
        #
        # keep_prob_dict = {"simple": keep_ratio_simple, "middle": keep_ratio_middle, "hard": keep_ratio_hard}
        #
        # downsample_dict = {"simple": downsampling_size_simple, "middle": downsampling_size_middle,
        #                    "hard": downsampling_size_hard}
        #
        # #
        # print(images_simple.shape)
        # if (images_simple.shape[0]):
        #     image_pick_prob_tensor = torch.ones(images_simple.shape[0]).cuda()
        #     images_simple_keep_num = int(round(images_simple.shape[0] * keep_prob_dict["simple"]))
        #     if images_simple_keep_num:
        #         indices = torch.multinomial(image_pick_prob_tensor, images_simple_keep_num)
        #         images_simple = torch.index_select(images_simple, 0, indices)
        #         target_simple = torch.index_select(target_simple, 0, indices)
        #     else:
        #         images_simple = torch.zeros(0, 3, crop_size, crop_size).cuda()
        #         target_simple = torch.zeros(0, crop_size, crop_size).cuda()
        #
        # if (images_middle.shape[0]):
        #     image_pick_prob_tensor = torch.ones(images_middle.shape[0]).cuda()
        #     images_middle_keep_num = int(round(images_middle.shape[0] * keep_prob_dict["middle"]))
        #     if images_middle_keep_num:
        #         indices = torch.multinomial(image_pick_prob_tensor, images_middle_keep_num)
        #         images_middle = torch.index_select(images_middle, 0, indices)
        #         target_middle = torch.index_select(target_middle, 0, indices)
        #     else:
        #         images_middle = torch.zeros(0, 3, crop_size, crop_size).cuda()
        #         target_middle = torch.zeros(0, crop_size, crop_size).cuda()
        #
        # if (images_hard.shape[0]):
        #     image_pick_prob_tensor = torch.ones(images_hard.shape[0]).cuda()
        #     images_hard_keep_num = int(round(images_hard.shape[0] * keep_prob_dict["hard"]))
        #     if images_hard_keep_num:
        #         indices = torch.multinomial(image_pick_prob_tensor, images_hard_keep_num)
        #         images_hard = torch.index_select(images_hard, 0, indices)
        #         target_hard = torch.index_select(target_hard, 0, indices)
        #     else:
        #         images_hard = torch.zeros(0, 3, crop_size, crop_size).cuda()
        #         target_hard = torch.zeros(0, crop_size, crop_size).cuda()
        #
        # # check if the smaple number == 1, if so, drop them because of BN's requirement in training
        # if (images_simple.shape[0] == 1):
        #     images_simple = torch.zeros(0, 3, crop_size, crop_size).cuda()
        #     target_simple = torch.zeros(0, crop_size, crop_size).cuda()
        # if (images_middle.shape[0] == 1):
        #     images_middle = torch.zeros(0, 3, crop_size, crop_size).cuda()
        #     target_middle = torch.zeros(0, crop_size, crop_size).cuda()
        # if (images_hard.shape[0] == 1):
        #     images_hard = torch.zeros(0, 3, crop_size, crop_size).cuda()
        #     target_hard = torch.zeros(0, crop_size, crop_size).cuda()
        #
        # # images_reorder = torch.cat((images_simple, images_middle, images_hard), dim=0)
        # # print(images_reorder)
        # # # target_reorder = torch.cat((target_simple, target_middle, target_hard), 0)
        #
        # run_flag_simple = False
        # run_flag_middle = False
        # run_flag_hard = False
        # print("shape: ")
        # print(images_simple.shape[0])
        # if (images_simple.shape[0]):
        #     print("images_simple.shape[0] is true")
        #     images_simple = torch.nn.Upsample(size=(downsample_dict["simple"], downsample_dict["simple"]),
        #                                       mode='nearest')(images_simple)
        #     run_flag_simple = True
        # if (images_middle.shape[0]):
        #     print("images_middle.shape[0] is true")
        #     images_middle = torch.nn.Upsample(size=(downsample_dict["middle"], downsample_dict["middle"]),
        #                                       mode='nearest')(images_middle)
        #     run_flag_middle = True
        # if (images_hard.shape[0]):
        #     print("images_hard.shape[0] is true")
        #     images_hard = torch.nn.Upsample(size=(downsample_dict["hard"], downsample_dict["hard"]), mode='nearest')(
        #         images_hard)
        #     run_flag_hard = True
        # print("Run flag")
        # print(run_flag_simple)
        # print(run_flag_middle)
        # print(run_flag_hard)
        # optimizer.zero_grad()
        # if run_flag_simple:
        #     print(images_simple.size())
        #     output_simple = model(images_simple)
        #     # output_simple = torch.nn.Upsample(size=(crop_size, crop_size), mode='nearest')(
        #     #     output_simple)
        #     loss_simple = F.cross_entropy(output_simple, target_simple)
        # else:
        #     # output_simple = torch.zeros(0, nclass, crop_size, crop_size).cuda()
        #     loss_simple = torch.zeros(1, ).cuda()
        #
        # if weight_dict["simple"] is not None:
        #     weight_simple = weight_dict["simple"] * images_simple.shape[0]
        # else:
        #     print("weight_simple set to be 0")
        #     weight_simple = 0.0
        # weighted_loss_simple = weight_simple * loss_simple
        #
        # if run_flag_middle:
        #     output_middle = model(images_middle)
        #     # output_middle = torch.nn.Upsample(size=(crop_size, crop_size), mode='nearest')(
        #     #     output_middle)
        #     loss_middle = F.cross_entropy(output_middle, target_middle)
        # else:
        #     # output_middle = torch.zeros(0, nclass, crop_size, crop_size).cuda()
        #     loss_middle = torch.zeros(1, ).cuda()
        # if weight_dict["middle"] is not None:
        #     weight_middle = weight_dict["middle"] * images_middle.shape[0]
        # else:
        #     weight_middle = 0.0
        #     print("weight_middle set to be 0")
        # weighted_loss_middle = weight_middle * loss_middle
        #
        # if run_flag_hard:
        #     output_hard = model(images_hard)
        #     # output_hard = torch.nn.Upsample(size=(crop_size, crop_size), mode='nearest')(
        #     #     output_hard)
        #     loss_hard = F.cross_entropy(output_hard, target_hard)
        # else:
        #     # output_hard = torch.zeros(0, nclass, crop_size, crop_size).cuda()
        #     loss_hard = torch.zeros(1, ).cuda()
        # if weight_dict["hard"] is not None:
        #     weight_hard = weight_dict["hard"] * images_hard.shape[0]
        # else:
        #     print("weight_hard set to be 0")
        #     weight_hard = 0.0
        # weighted_loss_hard = weight_hard * loss_hard
        #
        # # output_reorder = torch.cat((output_simple, output_middle, output_hard), 0)
        #
        # print(loss_scale_simple)
        # print(loss_scale_middle)
        # print(loss_scale_hard)
        #
        # print(weighted_loss_simple)
        # print(weighted_loss_middle)
        # print(weighted_loss_hard)
        # if (not weight_simple + weight_middle + weight_hard):
        #     loss = (weighted_loss_simple + weighted_loss_middle + weighted_loss_hard) / (
        #                 weight_simple + weight_middle + weight_hard)
        #     print("loss: ")
        #     # print(loss)
        #
        #     loss.backward()
        #
        #     if args.sr:
        #         updateBN()
        #     optimizer.step()
        #     train_loss += loss.item()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))
    # history_score[epoch][0] = avg_loss / len(train_loader)
    # history_score[epoch][1] = train_acc / float(len(train_loader))


def test(file):
    model.eval()
    test_loss = 0
    correct = 0
    writer = csv.writer(file)
    writer.writerow(["image_name", "image_name_copy", "image_complexity"])
    for i, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # print("Train Size: ")
        # print(SI(data).size())
        # print("Train tensor: ")
        si_score = SI(data).tolist()
        tar = target.tolist()
        # print(si_score)
        for idx in range(len(si_score) - 1):
            writer.writerow([str(i) + "_" + str(idx), tar[idx], si_score[idx]])

    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    # return correct / float(len(test_loader.dataset))


def efficient_test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


# def init_project_IC_hyperparam(self):
# linear_hyperparam = np.load("/data1/datasets/dataset_distribution/{}_{}_train_linear.npy".format("CIFAR-10", crop_size))
# mb_hyperparam = np.load("/data1/datasets/dataset_distribution/{}_{}_train_S.npy".format("CIFAR-10", crop_size))
# linear_min = linear_hyperparam[0]
# linear_max = linear_hyperparam[1]
# mb_a = mb_hyperparam[0]


def project_hyperparam(mean_value):
    if mean_value is None:
        return None, None, None
    else:
        downsampling_ratio = (downsampling_ratio_max - downsampling_ratio_min) * mean_value + downsampling_ratio_min
        keep_ratio = (keep_ratio_max - keep_ratio_min) * mean_value + keep_ratio_min
        loss_scale = (loss_scale_max - loss_scale_min) * mean_value + loss_scale_min
        downsampling_size = downsampling_ratio*crop_size
        downsampling_size = int(downsampling_size)
        return downsampling_size, float(keep_ratio), float(loss_scale)

def linear_project(x, _min, _max):
    raw_projected = (x - _min) / (_max - _min)
    return torch.clamp(raw_projected, 0.0, 1.0)


# def project_IC_func(complexity):
#   #  projection
#   return maxwell_boltzmann_cdf(complexity, mb_a)


# default kmean
def cluster_func(projected_complexity):
  return kmeans_cluster(projected_complexity, 3, 30)


def maxwell_boltzmann_cdf(x, a):  # torch version
  term_1 = torch.erf(x / np.sqrt(2.0) / a)
  term_2 = np.sqrt(2 / np.pi) * x * torch.exp(-(x ** 2) / (2 * a * a)) / a
  return term_1 - term_2

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    print("On Epoch: " + str(epoch))
    file_train = open("/home/hg31/work/Torch_SI/csv/CIFAR100_" +
                      str(epoch) + ".train_no_preprocessing.SI.train" + ".csv", "a")
    file_test = open("/home/hg31/work/Torch_SI/csv/CIFAR100_" + str(
        epoch) + ".test_no_preprocessing.SI.test" + ".csv", "a")
    # if epoch in [args.epochs*0.5, args.epochs*0.75]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1
    train(epoch, file_train)
    test(file_test)
    file_train.close()
    file_test.close()
    # history_score[epoch][2] = prec1
    # np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')
    # is_best = prec1 > best_prec1
    # best_prec1 = max(prec1, best_prec1)
    # save_checkpoint({
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'best_prec1': best_prec1,
    #     'optimizer': optimizer.state_dict(),
    # }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'record.txt'), history_score, fmt = '%10.5f', delimiter=',')