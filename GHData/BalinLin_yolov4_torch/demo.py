# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True


cfgfile_RGB='./cfg/yolo-1031_obj.cfg'
weightfile_RGB='./weight/yolo-1031_obj_final.weights'
cfgfile_TRM='./cfg/yolo-1031_obj.cfg'
weightfile_TRM='./weight/yolo-1031_trm_obj_final.weights'

imgfile_RGB='./data/RGB/RGB_costco4.avi'
imgfile_TRM='./data/TRM/Thermal_costco4.avi'
imgfile_fusion='./data/fusion/fusion_costco4.avi'

import cv2

RGB = Darknet(cfgfile_RGB)
RGB.print_network()
RGB.load_weights(weightfile_RGB)
print('Loading RGB weights from %s... Done!' % (weightfile_RGB))

if use_cuda:
    RGB.cuda()
num_classes = RGB.num_classes
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/1104_obj.names'
class_names = load_class_names(namesfile)


TRM = Darknet(cfgfile_TRM)
TRM.print_network()
TRM.load_weights(weightfile_TRM)
print('Loading TRM weights from %s... Done!' % (weightfile_TRM))
if use_cuda:
    TRM.cuda()
num_classes = TRM.num_classes
if num_classes == 20:
    namesfile = 'data/voc.names'
elif num_classes == 80:
    namesfile = 'data/coco.names'
else:
    namesfile = 'data/1104_obj.names'
class_names = load_class_names(namesfile)



RGB_test = cv2.VideoCapture(imgfile_RGB)
T_test = cv2.VideoCapture(imgfile_TRM)
fusion_test = cv2.VideoCapture(imgfile_fusion)

while RGB_test.isOpened():
    t1 = time.time()
    RGB_ret, RGB_frame = RGB_test.read()
    T_ret, T_frame = T_test.read()
    f_ret, f_frame = fusion_test.read() 
    if not RGB_ret or not T_ret or not f_ret:
        print('no video')
        break
    sized_RGB = cv2.resize(RGB_frame, (RGB.width, RGB.height))
    sized_RGB = cv2.cvtColor(sized_RGB, cv2.COLOR_BGR2RGB)
    sized_TRM = cv2.resize(T_frame , (TRM.width, TRM.height))
    sized_TRM = cv2.cvtColor(sized_TRM, cv2.COLOR_BGR2RGB)
    boxes_fusion = do_detect_ye(TRM, RGB, sized_TRM, sized_RGB, 0.25, 0.4, use_cuda)

    result_fusion = draw_bbox(f_frame, boxes_fusion[0], class_names=class_names, show_label=True)
    t2 = time.time()
    print('-----------------------------------')
    print('       max and argmax : %f' % (1 / (t2 - t1)))
    print('-----------------------------------')
    cv2.imshow('Yolo demo', result_fusion)
  #  cv2.imwrite("result_fusion.jpg",final_img)
    cv2.waitKey(1)
