# author: Jintao Huang
# date: 2020-5-14
# pred_image for train.py

import json
from models.efficientnet import efficientnet_b0, config_dict
import torch
from utils.display import resize_pad
from utils.utils import processing
import cv2 as cv


def pred_transform(image, target):
    """

    :param image: ndarray[H, W, C] RGB
    :param target: None
    :return: ndarray[H, W, C] RGB 0-255, None"""
    image = resize_pad(image, image_size, False, 32, False, 114)[0]
    return image, target


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
efficientnet = efficientnet_b0
image_size = config_dict[efficientnet.__name__][2]

# read images
image_fname = "images/1.jpg"
x = cv.imread(image_fname, cv.IMREAD_COLOR)
x = processing(x, pred_transform)[0].to(device)[None] / 255

# read labels
with open('imagenet_labels/labels_list.txt') as f:
    labels = json.load(f)

# pred
model = efficientnet(pretrained=True).to(device)
model.eval()
with torch.no_grad():
    pred = torch.softmax(model(x), dim=1)

# only show max
# print("-------------------------------------")
# value, idx = torch.max(pred[0], dim=-1)
# value, idx = value.item(), idx.item()
# print("Image Pred: %s" % image_fname)
# print("%-75s%.2f%%" % (labels[idx], value * 100))

# show top5
values, indices = torch.topk(pred, k=5)
print("Image Pred: %s" % image_fname)
print("-------------------------------------")
for value, idx in zip(values[0], indices[0]):
    value, idx = value.item(), idx.item()
    print("%-75s%.2f%%" % (labels[idx], value * 100))
