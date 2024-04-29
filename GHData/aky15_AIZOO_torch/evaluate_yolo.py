from __future__ import division

from yolo_utils.models import *
from yolo_utils.utils import *
from yolo_utils.datasets import *
from yolo_utils.parse_config import *
from yolo_utils.test import evaluate

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    weights_path = 'checkpoints_yolo/yolov3_ckpt_99.pth'
    test_img_path = './test/JPEGImages/'
    test_label_path = './test/Annotations/'
    test_list_path = './test/test.txt'    
    class_names = ['background','face','face_mask']

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))
    ap_face=[]
    ap_facemask=[]
    nms_thres = 0.5
    print("Compute mAP...")
    for iou_thres in np.arange(0.5,1,0.05):
        pascal_voc_map_face,pascal_voc_map_facemask,precision, recall, AP, f1, ap_class = evaluate(
            model,
            img_path = test_img_path,
            label_path = test_label_path,
            list_path = test_list_path,                        
            iou_thres=iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=nms_thres,
            img_size=opt.img_size,
            batch_size=8,
        )
        print("AP_face @ {}".format(iou_thres),pascal_voc_map_face)
        ap_face.append(pascal_voc_map_face)
        print("AP_facemask @ {}".format(iou_thres),pascal_voc_map_facemask)
        ap_facemask.append(pascal_voc_map_facemask)
    ap_face = np.array(ap_face)
    ap_facemask = np.array(ap_facemask)
    print("AP face:",ap_face.mean())
    print("AP facemask:",ap_facemask.mean())

