import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data.cocodataset import *
from data import config, BaseTransform, VOCAnnotationTransform, VOCDetection, VOC_ROOT, VOC_CLASSES
import numpy as np
import cv2
import time
from decimal import *


parser = argparse.ArgumentParser(description='YOLO-v2 Detection')
parser.add_argument('-v', '--version', default='yolo_v3_plus',
                    help='yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_half, yolo_v3_plus_medium, yolo_v3_plus_small, \
                            yolo_v3_slim, yolo_v3_slim_csp, yolo_v3_slim_csp2, \
                            yolo_v3_spp.')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='we use VOC-test or COCO-val to test.')
parser.add_argument('-size', '--input_size', default=416, type=int, 
                    help='Batch size for training')
parser.add_argument('--trained_model', default='weights/yolo_v2_72.2.pth',
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
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')


args = parser.parse_args()

coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def test_net(net, device, testset, thresh, mode='voc'):
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    num_images = len(testset)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        if args.dataset == 'COCO':
            img, _ = testset.pull_image(index)
            img_tensor, _, h, w, offset, scale = testset.pull_item(index)
        elif args.dataset == 'VOC':
            img = testset.pull_image(index)
            img_tensor, _, h, w, offset, scale = testset.pull_item(index)

        x = img_tensor.unsqueeze(0).to(device)

        t0 = time.clock()
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", Decimal(time.clock()) - Decimal(t0), "s")
        # scale each detection back up to the image
        max_line = max(h, w)
        # map the boxes to input image with zero padding
        bboxes *= max_line
        # map to the image without zero padding
        bboxes -= (offset * max_line)

        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                cls_id = coco_class_index[int(cls_indx)]
                cls_name = coco_class_labels[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.imshow('detection', img)
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
    num_classes = 80
    input_size = [args.input_size, args.input_size]

    if args.dataset == 'COCO':
        testset = COCODataset(
                    data_dir=coco_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=input_size[0],
                    transform=BaseTransform(input_size),
                    debug=args.debug)
    elif args.dataset == 'VOC':
        testset = VOCDetection(VOC_ROOT, [('2007', 'test')], BaseTransform(input_size))
    

    # build model
    # # yolo_v3_plus series: yolo_v3_plus, yolo_v3_plus_large, yolo_v3_plus_medium, yolo_v3_plus_small
    if args.version == 'yolo_v3_plus':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-53'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_large':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-l'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_large on the COCO dataset ......')

    elif args.version == 'yolo_v3_plus_half':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-h'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_half on the COCO dataset ......')

    elif args.version == 'yolo_v3_plus_medium':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-m'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_medium on the COCO dataset ......')
    
    elif args.version == 'yolo_v3_plus_small':
        from models.yolo_v3_plus import YOLOv3Plus
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-s'
        
        yolo_net = YOLOv3Plus(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_plus_small on the COCO dataset ......')
    
    # # yolo_v3_slim series: 
    elif args.version == 'yolo_v3_slim':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-slim'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim_csp on the COCO dataset ......')

    elif args.version == 'yolo_v3_slim_csp2':
        from models.yolo_v3_slim import YOLOv3Slim
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'csp-tiny'
        
        yolo_net = YOLOv3Slim(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo_v3_slim_csp2 on the COCO dataset ......')

    # # yolo_v3_spp
    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import YOLOv3SPP
        anchor_size = config.MULTI_ANCHOR_SIZE_COCO
        backbone = 'd-53'
        
        yolo_net = YOLOv3SPP(device, input_size=input_size, num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, anchor_size=anchor_size, backbone=backbone, diou_nms=args.diou_nms)
        print('Let us test yolo-v3-spp on the COCO dataset ......')

    else:
        print('Unknown version !!!')
        exit()


    yolo_net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    yolo_net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(yolo_net, device, testset, thresh=args.visual_threshold)

if __name__ == '__main__':
    test()