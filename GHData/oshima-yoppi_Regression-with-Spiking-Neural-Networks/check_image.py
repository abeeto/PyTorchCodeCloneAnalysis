import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from data import LoadDataset
import os 
from tqdm import tqdm
from model import snu_layer
from model import network
from model import loss
from tqdm import tqdm
#from mp4_rec import record, rectangle_record
import pandas as pd
# import scipy.io
# from torchsummary import summary
import argparse
import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--number', '-n', type=int, default=7)
args = parser.parse_args()
r = args.number


train_dataset = LoadDataset(dir = 'C:/Users/oosim/Desktop/snn/v2e/output/', which = "train" ,time = 100)
data = train_dataset.__getitem__(r)
for avi_name in glob.glob(f'C:/Users/oosim/Desktop/avi/{r}_*.avi'):
    print(avi_name)
# cap = cv2.VideoCapture(avi_name)
# #動画の表示
# while (cap.isOpened()):
#     #フレーム画像の取得
#     ret, frame = cap.read()
#     #画像の表示
#     cv2.imshow("Image", frame)
#     #キー入力で終了
#     if cv2.waitKey(10) != -1:
#         break


    
# cap.release()
# cv2.destroyAllWindows()

print(data[1])
# print(data[0].shape)
fig = plt.figure(figsize=(10, 4.5))
ax_frames = fig.add_subplot(1, 3, 1)
ax_events_p = fig.add_subplot(1, 3, 2)
ax_events_n = fig.add_subplot(1, 3, 3)

def anime(i):
    im = [ax_events_p.imshow(data[0][1,i,:,:],cmap='gray'),
            ax_events_n.imshow(data[0][0,i,:,:],)]
    return im 
def init():
    im = []

    ax_frames.set_title("Conventional camera view", fontweight='bold', fontsize=15)
    # ax_frames.set_xticks(np.arange(0, 129, 16))
    # ax_frames.set_yticks(np.arange(0, 129, 16))
    # ax_frames.xaxis.set_tick_params(labelsize=15)
    # ax_frames.yaxis.set_tick_params(labelsize=15)
    # ax_frames.set_xlim([0, frameWidth])
    # ax_frames.set_ylim([0, frameHeight])

    ax_events_p.set_title("Positive Events", fontweight='bold', fontsize=15)
    ax_events_n.set_title("Negative Events", fontweight='bold', fontsize=15)
    # ax_events.set_xlim([0, frameWidth])
    # ax_events.set_ylim([0, frameHeight])

    # im.append(ax_frames.imshow(frames[:, :, 0].T, cmap='gray'))
    # im.append(ax_events.imshow(np.zeros([frameWidth, frameHeight]), cmap='seismic', ))
    im.append(ax_frames.imshow())
    im.append(ax_events_p.imshow())
    im.append(ax_events_n.imshow())
    return im

ani = animation.FuncAnimation(fig, anime, interval=0.01, frames=range(99),)
plt.show()