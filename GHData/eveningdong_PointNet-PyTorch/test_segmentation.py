import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from datasets import SenSatTestDataset
from pointnet import PointNetDenseCls
from torch.autograd import Variable

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--npt', type=int, default = 4096,  help='number of points each example')


opt = parser.parse_args()
print (opt)

test_dataset = SenSatTestDataset(mode='test')

num_classes = test_dataset.num_classes

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetDenseCls(num_points=opt.npt, k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

if torch.cuda.is_available():
    classifier.cuda()

num_batch = len(test_dataset)
acc_list = []

for i, data in enumerate(test_dataset, 0):
    points, target = data
    points, target = Variable(points), Variable(target).long()
    points = points.transpose(2,1)
    if torch.cuda.is_available():
        points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred = classifier(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1,1)[:,0]

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    acc = correct.item()/float(points.shape[0]*opt.npt)
    print('[%d/%d] %s accuracy: %f' %(i, num_batch, 'test', acc))
    acc_list.append(acc)

print('overal accuracy: %f' %(np.mean(acc_list)))
        