from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

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

import numpy as np

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (img_names, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # 测试结果写文件
    core_pred = ""
    coreless_pred = ""
    for i in range(len(img_names)):
        img_name = img_names[i]
        pre_outs = outputs[i]
        for pre_out in pre_outs:
            pre_out_class = pre_out[6]
            pre_out_line = img_name \
                           + " " + "{:.3f}".format(pre_out[5]) \
                           + " " + "{:.1f}".format(pre_out[0]) \
                           + " " + "{:.1f}".format(pre_out[1]) \
                           + " " + "{:.1f}".format(pre_out[2]) \
                           + " " + "{:.1f}".format(pre_out[3]) \
                           + "\n"
            if(pre_out_class == 0):
                core_pred += pre_out_line
            else:
                coreless_pred += pre_out_line

    core_pred_path = path[3] + "det_test_带电芯充电宝.txt"
    coreless_pred_path = path[3] + "det_test_不带电芯充电宝.txt"

    with open(core_pred_path, "w") as file:
        file.writelines(core_pred)
    with open(coreless_pred_path, "w") as file:
        file.writelines(coreless_pred)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/st19.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="D:/mine/save/PyTorch-YOLOv3-Data/weights/yolov3_ckpt_54.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/st19.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--img_path", type=str, default="D:/mine/save/PyTorch-YOLOv3-Data/Image_test/", help="image path for test")
    parser.add_argument("--anno_path", type=str, default="D:/mine/save/PyTorch-YOLOv3-Data/Anno_test/", help="label path for test")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    
    # P modify start 2019-12-07
    """
    valid_path = data_config["valid"]
    """
    """
    以下代码从配置文件读测试集路径
    valid_path = []
    valid_path.append(data_config["valid"])
    valid_path.append(data_config["validImagesPath"])
    valid_path.append(data_config["validLabelsPath"])
    """
    # P modify end 2019-12-07
    
    # P modify start 2019-12-30
    """
    以下代码从调用参数读测试集路径
    """
    valid_path = []
    # 0 测试集文件名文件路径
    parent_path = os.path.dirname(os.path.dirname(opt.img_path))
    valid_path.append(parent_path + "/core_coreless_test.txt")
    # 1 测试集照片目录
    valid_path.append(opt.img_path)
    # 2 测试集标签目录
    valid_path.append(opt.anno_path)
    # 3 测试结果目录
    parent_path = os.path.dirname(os.getcwd())
    valid_path.append(parent_path + "/predicted_file/")
    
    # P modify end 2019-12-30
    
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
