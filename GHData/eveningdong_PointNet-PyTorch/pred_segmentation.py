import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from datasets import SenSatPredDataset
from pointnet import PointNetDenseCls
from torch.autograd import Variable

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, help='model path')
parser.add_argument('--npt', type=int, default = 4096,  help='number of points each example')
parser.add_argument('--input', type=str, help='numpy file dir path for the input')
parser.add_argument('--output', type=str, default='', help='numpy file path dir path for the prediction')


opt = parser.parse_args()
print (opt)

dataset = SenSatPredDataset(root=opt.input, mode='pred')

num_classes = dataset.num_classes

try:
    os.makedirs(opt.output)
except OSError:
    pass

classifier = PointNetDenseCls(num_points=opt.npt, k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

if torch.cuda.is_available():
    classifier.cuda()

num_batch = len(dataset)

for i, data in enumerate(dataset, 0):
    points, file_name = data
    points = Variable(points)
    points = points.transpose(2,1)
    if torch.cuda.is_available():
        points = points.cuda()
    classifier = classifier.eval()
    pred = classifier(points)
    pred = pred.view(-1, num_classes)

    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().numpy()
    save_file_name = os.path.join(opt.output, file_name.replace('data', 'pred'))
    np.save(save_file_name, pred_np)
    print('Save {}.'.format(save_file_name))

        