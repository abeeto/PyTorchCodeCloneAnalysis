from __future__ import division

from model import *
from util import *
from datasets import *

import os
import sys
import time
import datetime
import argparse
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import colorsys
from PIL import Image, ImageFont, ImageDraw
import cv2


def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_in", type=str, default="D:\BaiduNetdiskDownload\kalete.avi", help="path to dataset")
    parser.add_argument("--video_out", type=str, default="kalete_out.avi", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/kalete/ep893-map80.55-loss0.00.weights",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/kalete/dnf_classes.txt",
                        help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def).cuda()

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_state_dict(torch.load(opt.weights_path))
    else:
        print('无检测模型')

    model.eval()  # Set in evaluation mode

    # 加载类名
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    # 为每个类名配置不同的颜色
    hsv_tuples = [(x / len(classes), 1., 1.) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

    prev_time = time.time()
    vid = cv2.VideoCapture(opt.video_in)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if opt.video_out != "" else False
    if isOutput:
        print("!!! TYPE:", type(opt.video_out), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(opt.video_out, video_FourCC, video_fps, video_size)
    while True:
        return_value, frame = vid.read()
        h, w, c = frame.shape
        PIL_img = Image.fromarray(frame[:, :, ::-1])
        tensor_img = transforms.ToTensor()(PIL_img)
        img, _ = pad_to_square(tensor_img, 0)
        # Resize
        img = resize(img, (opt.img_size,opt.img_size)).cuda().unsqueeze(0)
        with torch.no_grad():
            detections = model(img)
            detections = NMS(detections, opt.conf_thres, opt.nms_thres)

        # current_time = time.time()
        # inference_time = current_time - prev_time
        # prev_time = current_time
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * h + 0.5).astype('int32'))
        thickness = (w + h) // 300
        if detections[0] is not None:
            # Rescale boxes to original image
            detections = xywh2xyxy(detections[0])
            # 先将在320*320标准下的xyxy坐标转换成max(600,800)下的坐标 再将x向或y向坐标减一下就行
            detections[:, :4] *= (max(h, w) / opt.img_size)
            if max(h - w, 0) == 0:
                detections[:, 1:4:2] -= (w - h) / 2
            else:
                detections[:, 0:3:2] -= (h - w) / 2
            # 随机取一个颜色
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                label = '{} {:.2f}'.format(classes[int(cls_pred)], cls_conf.item())
                draw = ImageDraw.Draw(PIL_img)
                # 获取文字区域的宽高
                label_w, label_h = draw.textsize(label, font)
                # 画出物体框 顺便加粗一些边框
                for i in range(thickness):
                    draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=colors[int(cls_pred)])
                # 画出label框
                draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=colors[int(cls_pred)])
                draw.text((x1, y1 - label_h), label, fill=(0, 0, 0), font=font)
            cv_img = np.array(PIL_img)[..., ::-1]
        cv2.imshow('result', cv_img)
        # cv2.waitKey(300)
        out.write(cv_img)
        cv2.waitKey(1)

