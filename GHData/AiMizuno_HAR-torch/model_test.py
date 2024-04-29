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
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
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

    f_val = open(args.test_list, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    line_id = 1
    match_count = 0
    result_list = []
    for line in val_list:
        line_info = line.split(" ")
        clip_path = line_info[0]
        input_video_label = int(line_info[2])# - 1 

        if args.modality == "RGB":
            prediction = VideoSpatialPrediction(
                clip_path,
                net,
                num_categories,
                start_frame
            )
        
        elif args.modality == "Flow":
            prediction = VideoTemporalPrediction(
                clip_path,
                net,
                num_categories,
                start_frame
            )

        avg_pred_fc8 = np.mean(prediction, axis=1)
        # print(avg_spatial_pred_fc8.shape)
        result_list.append(avg_pred_fc8)
        # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

        pred_index = np.argmax(avg_pred_fc8)
        print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, pred_index))

        if pred_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(val_list))
    print("Accuracy is %4.4f" % (float(match_count)/len(val_list)))
    np.save("{}_sX_{}_{}.npy".format(args.dataset, args.modality, args.arch), np.array(result_list))#hard code

if __name__ == "__main__":
    main()