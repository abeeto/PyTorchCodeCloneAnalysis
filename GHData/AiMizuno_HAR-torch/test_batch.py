#!/usr/bin/env python

import os, sys
import collections
import numpy as np
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix

import datasets
import argparse
import models
import video_transforms
from scripts import VideoSpatialPrediction
from scripts import VideoTemporalPrediction

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['rgb', 'flow', 'RGBDiff'])
parser.add_argument('test_list', type=str) # ce shi lu jing
parser.add_argument('weights', type=str)#.pth
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

parser.add_argument('-b', '--batch-size', default=25, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')


def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():
    global args
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_categories = 101
    elif args.dataset == 'hmdb51':
        num_categories = 51
    elif args.dataset == 'kinetics':
        num_categories = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    start_frame = 0

    model_start_time = time.time()
    params = torch.load(args.weights)

    #hard code
    net = models.__dict__[args.arch](pretrained = False, num_classes = num_categories)
    net.load_state_dict(params['state_dict'])
    net.cuda()
    net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))
    ###

    if args.modality == "rgb":
        new_length = 1
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * new_length
        clip_std = [0.229, 0.224, 0.225] * new_length
    elif args.modality == "flow":
        new_length = 10
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * new_length
        clip_std = [0.226, 0.226] * new_length
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.GroupCenterCrop(net.input_size),
            video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])    

    dataset = datasets.load_clip(root=args.data,
                                            source=args.test_list,
                                            phase="val",
                                            modality=args.modality,
                                            is_color=is_color,
                                            new_length=new_length,
                                            new_width=args.new_width,
                                            new_height=args.new_height,
                                            video_transform=val_transform)     
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True) 

    data_gen = enumerate(data_loader)

    total_num = len(data_loader.dataset)
    output = []   

    def eval_video(video_data):
        i, data, label = video_data
        num_crop = 1

        if args.modality == 'rgb':
            length = 3
        elif args.modality == 'flow':
            length = 10
        elif args.modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+args.modality)

        input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                            volatile=True)
        input_var = input_var.type(torch.FloatTensor).cuda()         
        rst = net(input_var).data.cpu().numpy().copy()
        return i, rst.reshape((num_crop, args.test_segments, num_categories)).mean(axis=0).reshape(
            (args.test_segments, 1, num_categories)
        ), label[0]

    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

    for i, (data, label) in data_gen:
        data = data.float().cuda(async=True)
        label = label.cuda(async=True)
        input_var = torch.autograd.Variable(data, volatile=True)
        target_var = torch.autograd.Variable(label, volatile=True)

        rst = net(input_var).data.cpu().numpy()
        #avg_pred_fc8 = np.mean(rst, axis=1)
        # print(avg_spatial_pred_fc8.shape)
        #result_list.append(avg_pred_fc8)
        # avg_spatial_pred = softmax(avg_spatial_pred_fc8)
        pred_index = np.argmax(rst)
        #print (label.cpu().numpy())
        #print (pred_index)
        # print(rst)
        # if i >= max_num:
        #     break
        # rst = eval_video((i, data, label))
        output.append(rst)
        if label.cpu().numpy()[0] == pred_index:
            match_count += 1
        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))
    print(match_count)
    print(total_num)
    print("Accuracy is %4.4f" % (float(match_count)/float(total_num)))

    np.save("{}_sX_{}_{}.npy".format(args.dataset, args.modality, args.arch), np.array(output))#hard code
    
if __name__ == "__main__":
    main()