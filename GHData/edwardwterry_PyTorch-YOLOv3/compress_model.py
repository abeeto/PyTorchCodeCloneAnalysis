from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import argparse

import torch
from torch2trt import torch2trt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_35.pth", help="path to weights file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading model...')
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    print('Loading weights...')
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print('Setting model to eval mode')
    model.eval().cuda()

    x = torch.ones((1,3,416,416)).cuda()

    print('Converting model to TensorRT format')
    model_trt = torch2trt(model, [x])

    torch.save(model_trt.state_dict(), opt.weights_path.split('.')[0] + '_trt.pth')
