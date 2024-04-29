# -*- coding: utf-8 -*-
import sys
import argparse
import os
import shutil
import time

import fnmatch

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import csv

import numpy
from numpy import matrix
from numpy import linalg
import cv2
from PIL import Image


def printProgress(iteration, total, prefix='', suffix='', decimals=2, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def find_coeffs_similarity(pa, pb):

    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0])
        matrix.append([p1[1], -p1[0], 0, 1])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(2 * len(pa))

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    res = res[0][0]
    T = numpy.matrix([[res[0, 0], res[0, 1], res[0, 2]],
                      [-res[0, 1], res[0, 0], res[0, 3]]])

    return (T[0, 0], T[0, 1], T[0, 2], T[1, 0], T[1, 1], T[1, 2])


def load_csv(filename):
    data = []
    titles = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            d = {}
            if i == 0:
                for item in row:
                    titles.append(item)
            else:
                for j, item in enumerate(row):
                    d[titles[j]] = item

                data.append(d)
        # print titles
    return (data, titles)


def recglob(directory, ext):
    l = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, ext):
            l.append(os.path.join(root, filename))
    return l


class RajeevNet(nn.Module):
    def __init__(self):
        super(RajeevNet, self).__init__()

    def forward(self, input):
        # print input.size()
        x = nn.AdaptiveAvgPool2d(1)(input)
        # print x.size()
        # x = torch.squeeze(x)
        x = 20 * F.normalize(x)
        # print x.size()
        x = x.contiguous()
        return x


class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, input):
        print 'here2'
        return input


def process(item):

    filename = item['FILE']

    try:
        im = Image.open(filename).convert('RGB')
    except:
        return

    # idx = [6,8,9,11,14,17,19]
    idx = [6, 8, 9, 11, 17, 19]
    li = []
    for i in range(1, 22):
        li.append((float(item['P%dX' % i]), float(item['P%dY' % i])))
    q = []
    for x in idx:
        q.append(li[x])

    #-----
    ref = []
    ref.append((50.0694, 68.3160))
    ref.append((68.3604, 68.3318))
    ref.append((88.3886, 68.1872))
    ref.append((106.9256, 67.6126))
    # ref.append((78.7128,86.0086))
    ref.append((62.2908, 106.0550))
    ref.append((95.7594, 105.5998))

    coeffs = find_coeffs_similarity(q, ref)

    # x0 = -0.8762
    # x1 = 157.1373
    # y0 = -7.3191
    # y1 = 151.0860

    # swami btas
    x0 = 17.1238
    x1 = 140.1373
    y0 = -7.3191
    y1 = 151.086

    T = matrix([[coeffs[0], coeffs[1], coeffs[2] - x0],
                [coeffs[3], coeffs[4], coeffs[5] - y0], [0, 0, 1]])
    Tinv = linalg.inv(T)
    Tinvtuple = (Tinv[0, 0], Tinv[0, 1], Tinv[0, 2],
                 Tinv[1, 0], Tinv[1, 1], Tinv[1, 2])

    im = im.transform((200, 200), Image.AFFINE,
                      Tinvtuple, resample=Image.BILINEAR)
    # im = im.crop((8,8,151,151))
    # swami btas
    im = im.crop((0, 0, int(round(x1 - x0)), int(round(y1 - y0))))
    im = im.resize((256, 256), resample=Image.BILINEAR)
    # im = im.crop((14,14,241, 241))

    #-----

    return im


def main():

    (data, titles) = load_csv('lfwall_ultraface.csv')

    print("=> using pre-trained model '{}'".format('resnet18'))
    model = models.__dict__['resnet18'](pretrained=True)
    model.avgpool = RajeevNet()
    model.fc = torch.nn.Linear(512, 8277)

    model = torch.nn.DataParallel(model).cuda()
    print("=> loading checkpoint '{}'".format('model_best.pth.tar'))
    checkpoint = torch.load('model_best.pth.tar')
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded")

    model.fc = NoOp()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    fout = open('lfwall_resnet18_deep_features.csv', 'w')
    header = ['FILE']
    for i in range(1, 513):
        header.append('DEEPFEATURE_%d' % i)

    fout.write(','.join(header) + '\n')
    lines = []
    for it, item in enumerate(data):
        vs = []
        printProgress(it, len(data))

        im = process(item)
        vs.append(item['FILE'])

        model.eval()

        t = transforms.Compose([
            transforms.RandomCrop(227),
            transforms.ToTensor(),
            normalize
        ])

        input = Variable(t(im).float(),
                         volatile=True).squeeze(0).cuda()
        x = torch.unsqueeze(input, 0)
        output = model(x)
        # l = []
        for i in range(512):

            vs.append(str(output[0][i].data[0]))

        lines.append(','.join(vs))
    fout.write('\n'.join(lines))


if __name__ == "__main__":
    main()
