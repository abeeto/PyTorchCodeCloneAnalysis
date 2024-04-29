import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import os.path
import os
import glob
import re
import shutil
from torch import torch
from os import path
from PIL import Image
from datetime import datetime, timedelta, timezone
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
from detecto.utils import read_image, filter_top_predictions
from torch.functional import Tensor
from datetime import datetime
from paddleocr import PaddleOCR

print(torch.cuda.is_available())
#===============================================
pnow = datetime.now()
pcurrent_time = pnow.strftime("%d_%m_%Y_%H_%M_%S")
print(pcurrent_time)
#===============================================

def convertTuple(tup):
    str = ''.join(tup)
    return str

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
model = core.Model.load('models/cnn-plate-4.dll', ['plateb', 'platey'])
while True:
    folderkosong = False
    hasFrame = False
    save_path = 'images'
    list_of_files = glob.glob('images\*.jpg') # * means all if need specific format then *.csv
    #======== read file ====================
    if os.path.isfile("images/img_01.jpg"):
        time.sleep(0.5)
                
    if (list_of_files != []):
        im = mpimg.imread("images/img_01.jpg")
        print('start..')
        image = im
        mshow = plt.imshow(image)
        plt.title('image')
        plt.show()
        
        #=============================
        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:%S")
        print(current_time)
        #======== process prediktion ====================

        predictions = model.predict_top(image)
        labels, boxes, scores = predictions
                
        big_score = 0
        cnt = 0
        area =0
        big_area =0
        boxes2 = []
        for box in boxes:            
            np_arr = box.cpu().detach().numpy()
            box_x1 = np_arr.item(0)
            box_y1 = np_arr.item(1)
            box_x2 = np_arr.item(2)
            box_y2 = np_arr.item(3)
            area = (box_x2 - box_x1)*(box_y2 - box_y1)
            if area > big_area:
                cnt = cnt+ 1
                big_area= area
                boxes2 = boxes[cnt-1]
                
        cropped = image
        try:
            np_arr = boxes2.cpu().detach().numpy()
            box_x1 = np_arr.item(0)
            box_y1 = np_arr.item(1)
            box_x2 = np_arr.item(2)
            box_y2 = np_arr.item(3)
            roundbox_x1 = int(box_x1) - 20
            roundbox_x2 = int(box_x2) + 15
            roundbox_y1 = int(box_y1)
            roundbox_y2 = int(box_y2)
            roundbox_y1 = roundbox_y1 - 5
        except:
            roundbox_x1 = 1720
            roundbox_x2 = 1920
            roundbox_y1 = 0
            roundbox_y2 = 100
                
        panjang = 3 * (roundbox_y2-roundbox_y1 )
        selisih = panjang-(roundbox_x2-roundbox_x1)
        pertambahan = int((panjang-(roundbox_x2-roundbox_x1))/2)
        roundbox_x1 = roundbox_x1 - pertambahan
        roundbox_x2 = roundbox_x2 + pertambahan        
        cropped = cropped[roundbox_y1:roundbox_y2,roundbox_x1:roundbox_x2]
        mshow = plt.imshow(cropped)
        plt.title('cropped')
        plt.show()
                
        number_plate = '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        plate = 'A', 'AA', 'AB', 'AD', 'AE', 'AG', 'B', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP', 'D', 'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DR', 'DT', 'E', 'EA', 'EB', 'ED', 'F', 'G', 'H', 'K', 'KB', 'KH', 'KT', 'KU', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z'
                
        merger_text = ''
        try:
            result = ocr.ocr(cropped, det=True, cls=False)
            txts = [line[1][0] for line in result]   
            if txts !=[]:
                strxsss = convertTuple(txts)
                upcase = strxsss.upper()
                katas = re.sub("/", "4", upcase)
                texts = re.findall("[A-Z0-9]" ,katas)
                merger = ''.join(map(str, texts))
                if (merger[0] in plate):
                    if (merger[1] in plate):
                        if (merger[2] in number_plate):
                            merger_text = merger[0:8] + ''
                        else:
                            merger_text = merger[0:2] + merger[3:8] + ''
                    else:
                        merger_text = merger[0:8] + ''
                elif (merger[0] in number_plate):
                    merger_text = merger[0:6]
                else:
                    merger_text = merger[1:8] + ''
        except:
            merger_text = ''
        print(merger_text)
        #===============================================
        pnow = datetime.now()
        pcurrent_time = pnow.strftime("%d-%m-%Y %H:%M:%S")
        print('finish..')
        print(pcurrent_time)
        #===============================================
                
        filelist = glob.glob(os.path.join(save_path, "*"))
        for f in filelist:
            os.remove(f)
        print("file deleted \n-------------")