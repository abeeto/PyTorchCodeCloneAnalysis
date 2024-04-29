import argparse
import os
import shutil
import time
import random
from sklearn.metrics import roc_auc_score
import torch
import options
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision

from utility import *
import math

from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable

from networks import *
 
from torchvision.utils import save_image

from dataset import set_data


def main(opt):
    args = opt
    state = {k: v for k, v in args._get_kwargs()}
    exp=args.exp_name
    checkpoint = args.chechpoint
    if not os.path.isdir(checkpoint):
        utility.mkdir_p(checkpoint)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    if args.manualSeed is None:
        args.manualSeed = random.randint(1,10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.data.manual_seed_all(args.manualSeed)

    best_acc = 1000

    dataloading = set_data(args)
    num_classes = 2
    Tensor = torch.cuda.FloatTensor
    enc = get_encoder().cuda()
    dec = get_decoder().cuda()

    checkpoint_en = torch.load(os.path.join(checkpoint, enc_model.pth.tar))
    enc.load_state_dict(checkpoint_en['state_dict'])

    checkpoint_de = torch.load(os.path.join(checkpoint, dec_model.pth.tar))
    dec.load_state_dict(checkpoint_de['state_dict'])

    for batch_idx, (inputs, targets) in enumerate(dataloading['test']):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        recon = dec(enc(inputs))



        scores = torch.mean(torch.pow((inputs-recon), 2), dim=[1,2,3])
        prec1 = roc_auc_score(targets.cpu().detach().numpy(), -score.cpu().detach().numpy())
        print('\nBatch: {0:d} ==== auc: {1:.2e}' .format(batch_idx, prec1))
        pic = recon.cpu().data()
        img = inputs.cpu().data()
        fake_name = 'test_dc_fake0{}.png'.format(batch_idx)
        real_name = 'test_dc_real0{}.png'.format(batch_idx)
        save_image(pic,os.path.join(".",result,exp,fake_name))
        save_image(img, os.path.join(".",result,exp,real_name))
    print('saving pic...')

if __name__ == '__main__':
    opt = options.test_options()
    best_acc = 1000
    main(opt)