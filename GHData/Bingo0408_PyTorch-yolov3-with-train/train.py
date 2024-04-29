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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image-folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch-size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model-config-path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data-config-path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights-path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class-path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf-thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms-thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n-cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img-size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint-interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint-dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use-cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# 读取训练数据的图片绝对地址列表
# trainvalno5k.txt与trainvalno5k.part的区别在于前者被转化成了绝对地址
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
# model.load_weights(opt.weights_path)
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):

    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        """ _:      输出的是形如('${image_path}/image1.jpg', '${image_path}/image2.jpg')的Tuple, 长度为batch_size
            imgs:   输出的是Tensor形式的图片矩阵, 维度为[2, 3, 416, 416], 2为batch_size
            targets:输出的是Tensor形式的Labels, 维度为[2,50,5], 参考data/coco_images_sample/labels/train2014下的label定义(中心坐标以及w,h)
                    2为batch_size, 50为检测的最大目标数量, 5表示坐标以及类别。如果不满50行填全0表示。 """
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
