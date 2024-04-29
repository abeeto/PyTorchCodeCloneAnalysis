import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import VOC_ROOT, VOC_CLASSES
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import config
import numpy as np
import cv2
import tools
import time
from decimal import *


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo_v3_spp',
                    help='yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_half, yolo_v3_plus_medium, yolo_v3_plus_small, \
                            yolo_v3_slim, yolo_v3_slim_csp, yolo_v3_slim_csp2, \
                            yolo_v3_spp.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='Batch size for training')
parser.add_argument('--trained_model', default='weight/voc/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--diou_nms', action='store_true', default=False, 
                    help='use diou_nms.')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--voc_root', default=VOC_ROOT, 
                    help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()

def test_net(net, device, testset, input_size, thresh, mode='voc'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img_raw = testset.pull_image(index)

        img_tensor, _, h, w, offset, scale = testset.pull_item(index)
        # img_id, annotation = testset.pull_anno(i)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        t0 = time.clock()
        bboxes, scores, cls_inds = net(img_tensor)
        print("detection time used ", Decimal(time.clock()) - Decimal(t0), "s")
        # scale each detection back up to the image
        max_line = max(h, w)
        # map the boxes to input image with zero padding
        bboxes *= max_line
        # map to the image without zero padding
        bboxes -= (offset * max_line)

        CLASSES = VOC_CLASSES
        class_color = tools.CLASS_COLOR
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img_raw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                cv2.rectangle(img_raw, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                mess = '%s' % (CLASSES[int(cls_indx)])
                cv2.putText(img_raw, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('detection', img_raw)
        cv2.waitKey(0)


def test():
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    num_classes = len(VOC_CLASSES)
    input_size = [args.input_size, args.input_size]
    testset = VOCDetection(args.voc_root, [('2007', 'test')], BaseTransform(input_size))

    # build model
    # # yolo_v3_plus series: yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small
    if args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'd-53'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_plus on the VOC dataset ......')
    
    elif args.version == 'yolo_v3_plus_large':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'csp-l'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_plus_large on the VOC dataset ......')
   
    elif args.version == 'yolo_v3_plus_half':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'csp-h'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_plus_half on the VOC dataset ......')
    
    elif args.version == 'yolo_v3_plus_medium':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'csp-m'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_plus_medium on the VOC dataset ......')
    
    elif args.version == 'yolo_v3_plus_small':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'csp-s'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_plus_small on the VOC dataset ......')
    
    # # yolo_v3_slim series: 
    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'd-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_slim on the VOC dataset ......')

    elif args.version == 'yolo_v3_slim_csp':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'csp-slim'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_slim_csp on the VOC dataset ......')

    elif args.version == 'yolo_v3_slim_csp2':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'csp-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo_v3_slim_csp2 on the VOC dataset ......')
       
    # # yolo_v3_spp
    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        anchor_size = config.MULTI_ANCHOR_SIZE
        backbone = 'd-53'
        
        yolo_net = YOLOv3SPP(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, diou_nms=args.diou_nms, anchor_size=anchor_size, backbone=backbone)
        print('Let us test yolo-v3-spp on the VOC dataset ......')

    else:
        print('Unknown version !!!')
        exit()


    yolo_net.load_state_dict(torch.load(args.trained_model, map_location=device))
    yolo_net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(yolo_net, device, testset, input_size,
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()