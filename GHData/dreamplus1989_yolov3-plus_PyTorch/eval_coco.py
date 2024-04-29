from __future__ import division

from utils.cocoapi_evaluator import COCOAPIEvaluator
from data import *
from utils.augmentations import SSDAugmentation
from data.cocodataset import *
import tools

import os
import random
import argparse
import time
import math
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo_v3_plus',
                    help='yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_half, yolo_v3_plus_medium, yolo_v3_plus_small, \
                            yolo_v3_slim, yolo_v3_slim_csp, yolo_v3_slim_csp2, \
                            yolo_v3_spp.')
parser.add_argument('-t', '--testset', action='store_true', default=False,
                    help='COCO_val, COCO_test-dev dataset')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='Batch size for training')
parser.add_argument('--dataset_root', default='/home/k545/object-detection/dataset/COCO/', 
                    help='Location of COCO root directory')
parser.add_argument('--trained_model', default='weights/coco/', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--diou_nms', action='store_true', default=False,
                    help='Use DIoU NMS')
parser.add_argument('--num_classes', default=80, type=int, 
                    help='The number of dataset classes')
parser.add_argument('--n_cpu', default=8, type=int, 
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')


args = parser.parse_args()

def test(model, input_size, device):
    if args.testset:
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=True,
                        transform=BaseTransform(input_size)
                        )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=coco_root,
                        img_size=input_size,
                        device=device,
                        testset=False,
                        transform=BaseTransform(input_size)
                        )

    # COCO evaluation
    ap50_95, ap50 = evaluator.evaluate(model)
    print('ap50 : ', ap50)
    print('ap50_95 : ', ap50_95)


if __name__ == '__main__':

    input_size = [args.input_size, args.input_size]
    num_classes = 80
    if args.cuda:
        print('use cuda')
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # build model
    # # yolo_v3_plus series: yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small
    if args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-53'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_large':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-l'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_large on the COCO dataset ......')

    elif args.version == 'yolo_v3_plus_half':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-h'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_half on the COCO dataset ......')

    elif args.version == 'yolo_v3_plus_medium':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-m'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_medium on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_small':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-s'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_small on the COCO dataset ......')
    
    # # yolo_v3_slim series: 
    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-slim'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim_csp on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp2':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim_csp2 on the COCO dataset ......')

    # # yolo_v3_spp
    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        anchor_size = MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-53'
        
        yolo_net = YOLOv3SPP(device, input_size=input_size, num_classes=num_classes, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo-v3-spp on the COCO dataset ......')

    else:
        print('Unknown version !!!')
        exit()

    # load model
    yolo_net.load_state_dict(torch.load(args.trained_model, map_location=device))
    yolo_net.eval().to(device)
    print('Finished loading model!')

    test(yolo_net, input_size, device)
