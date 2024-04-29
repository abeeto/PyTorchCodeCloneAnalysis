# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 03:08:12 2021

@author: taebe
"""

import torch
import torchvision

import torchvision.models.detection as tvd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MAXIMUM_OBJ = 10
FLG = 0

#model = tvd.ssd300_vgg16(pretrained=True)
#model = tvd.ssdlite320_mobilenet_v3_large(pretrained=True)
model = tvd.retinanet_resnet50_fpn(pretrained=True)
#model = tvd.maskrcnn_resnet50_fpn(pretrained=True)

#
# Faster R-CNN
#
# 느림
#model = tvd.fasterrcnn_resnet50_fpn(pretrained=True)
# 빠름
#model = tvd.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
# 느림
#model = tvd.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# 정확
#model = tvd.maskrcnn_resnet50_fpn(pretrained=True)

MODEL = "retinanet_resnet50_fpn"

if torch.cuda.is_available():
    model.cuda()

model.eval()

# We will now get a list of class names for this model, i will link the notebook ni the description.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

from PIL import Image
import numpy as np
from io import BytesIO # For url images
import requests
from torchvision import transforms as T
import matplotlib.pyplot as plt
import cv2
import random
from urllib.request import urlopen
import os
import time

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' # For retina displays
def get_prediction(img, threshold=0.7, gpu=False):
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    if gpu:
        img = img.to(device)
    pred = model([img]) # We have to pass in a list of images
    if gpu:
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # If using GPU, you would have to add .cpu()
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding Boxes
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    else:
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]  # Bounding Boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
    #pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    except:
        pred_t = 0

    pred_box = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_box, pred_class, pred_score

def object_detection(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3, type='img', gpu=False):
    if type=='url':
        img = url_to_image(img_path)  # If on the internet.
        # Not all images will work though.
    elif (type == 'img'):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # If Local
    elif (type == 'video'): # blackbox_2 영상에서 본네트 부분 제거하기 위해 crop 부분 추가
        img = img_path
        #print(img_path.shape)
        roi = img_path[250:550, ].copy()
    else:
        img = img_path


    #
    # Object detection
    #
    #boxes, pred_clas, pred_score = get_prediction(img, threshold=threshold, gpu=gpu)
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    boxes, pred_clas, pred_score = get_prediction(roi, threshold=threshold, gpu=gpu)
    pred_score = [str(round(a,4)) for a in pred_score]
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    import PIL
    #img_pil = Image.fromarray(img)
    #draw = PIL.ImageDraw.Draw(img_pil)
    #font = PIL.ImageFont.truetype("fonts/gulim.ttc", 20)

    for i in range(len(boxes)):
        if i > MAXIMUM_OBJ:
            break
        r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random Color
        #print(boxes)
        #cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
        #              color=(r, g, b), thickness=rect_th)  # Draw Rectangle with the coordinates
        #draw.text((int(boxes[i][0][0]), int(boxes[i][0][1]) + 150), pred_clas[i], font=font, fill=(b, g, r, 0))
        #img = np.array(img_pil)
        if float(pred_score[i]) > 0.75:
            cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])+250),
                          (int(boxes[i][1][0]), int(boxes[i][1][1])+250),
                          color=(255, 0, 0), thickness=rect_th, lineType=cv2.LINE_AA)  # Draw Rectangle with the coordinates

            #cv2.putText(img, pred_clas[i], (int(boxes[i][0][0]), int(boxes[i][0][1])+150), cv2.FONT_HERSHEY_SIMPLEX, text_size,
            #            (r, g, b), thickness=text_th)
            #cv2.putText(img, pred_score[i], (int(boxes[i][0][0]), int(boxes[i][0][1])+250), cv2.FONT_HERSHEY_SIMPLEX, text_size,
            #            (r, g, b), thickness=text_th)
            cv2.putText(img, pred_clas[i], (int(boxes[i][0][0]), int(boxes[i][0][1])+250),
                        color=(0,255,0),thickness=1, fontFace=0, fontScale=0.5)
            #cv2.putText(img, pred_clas[i] + ' ' + pred_score[i], (int(boxes[i][0][0]), int(boxes[i][0][1])+250),
            #            cv2.FONT_HERSHEY_SIMPLEX, text_size,
            #            (0, 255, 0), thickness=text_th)
        #print(f'pred_{i} score: {pred_score[i]}')
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # plt.figure(figsize=(15, 15))
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    #cv2.imshow('img', img)
    return img

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
  resp = urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, readFlag)
  return image

#
# data path
#
file_path = 'D:/dataset/vehicle_pedestrian_detect_video/Training/img_ggd_FLRR_1_02/20201126_ggd_front_5_0002'
file_names = os.listdir(file_path)
f_names=[]
for i in file_names:
    nm = i.split('.png')[0]
    f_names.append(nm)
f_names = list(map(int, f_names))
f_names.sort()

#
# convert the names of training or test data
#
def hangul_to_index(file_path):
    file_names = os.listdir(file_path)
    i = 1
    for name in file_names:
        src = os.path.join(file_path, name)
        dst = str(i) + '.png'
        dst = os.path.join(file_path, dst)
        os.rename(src, dst)
        i += 1

#
# image 파일 실행
#
# for file in f_names:
#     #print(file_path+'/'+str(file)+'.png')
#     img_name = file_path+'/'+str(file)+'.png'
#     res = object_detection(img_name, rect_th=2, text_size=1, text_th=2, type='img', gpu=True)
#     cv2.imshow('frame', res)
#     cv2.waitKey(1)

#
# video 파일 실행
#
#cap = cv2.VideoCapture('test_video.mp4')
#cap = cv2.VideoCapture('bts_Trim.mp4')
cap = cv2.VideoCapture('blackbox_2.mp4')

tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fps = 20
width = int(cap.get(3))
height = int(cap.get(4))
fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('output/'+MODEL+'.avi', fcc, fps, (width, height))
cnt = 0
if cap.isOpened():
    while True:
        t1 = time.time()
        ret, img = cap.read()
        if ret:
            res = object_detection(img, rect_th=1, text_size=1, text_th=1, type='video', gpu=True)
            t2 = time.time()
            cv2.putText(img, 'Model: '+MODEL, (25, 25),
                        color=(255, 255, 0), thickness=1, fontFace=0, fontScale=.5)
            cv2.putText(img, 'FPS: '+str(round(1/(t2-t1), 2)), (25, 50),
                        color=(255, 255, 0), thickness=1, fontFace=0, fontScale=.5)
            #cv2.imshow('frame', res)
            out.write(res)

            #
            # count frames
            #
            cnt += 1
            print(f'{cnt}/{tot_frames}')

            #print(f'Done. ({t2 - t1:.3f}s)')
            cv2.waitKey(1)
        else:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
