# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:16:08 2020

@author: cj
"""

import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

IMG_SIZE = 480
THRESHOLD = 0.95


model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

import os
d_list='D:/image/'
for ls in os.listdir(d_list):
    print(ls)
    img = Image.open(d_list+ls)
    img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))
    
    plt.figure(figsize=(16, 16))    
    
    trf = T.Compose([T.ToTensor()])
    
    input_img = trf(img)
    out = model([input_img])[0]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO]    
    fig, ax = plt.subplots(1, figsize=(16, 16))    
    ax.imshow(img)
    
    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().numpy()
    
        if score < THRESHOLD:
            continue
    
        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()[:, :2]
    
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    
        # 17 keypoints
        for k in keypoints:
            circle = patches.Circle((k[0], k[1]), radius=2, facecolor='r')
            ax.add_patch(circle)
        
        # draw path
        # left arm
        path = Path(keypoints[5:10:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
        
        # right arm
        path = Path(keypoints[6:11:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
        
        # left leg
        path = Path(keypoints[11:16:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
        
        # right leg
        path = Path(keypoints[12:17:2], codes)
        line = patches.PathPatch(path, linewidth=2, facecolor='none', edgecolor='r')
        ax.add_patch(line)
    plt.savefig('D:/model_output/'+ls)