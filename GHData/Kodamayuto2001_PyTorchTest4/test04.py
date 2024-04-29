#region IMPORT
from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import cv2
from skimage import io, transform
from PIL import Image
from pathlib import Path
#endregion

#region データセットの準備
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,dirPath,tagName):
        self.__MakeData(dirPath)
        self.__Rescale()
        #self.__ToTensor()
        pass
    
    def __MakeData(self,dirPath):
        try:
            os.makedirs(dirPath)
        except FileExistsError:
            pass
        cascadePath = "opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascadePath)#detectorを作成
        cap = cv2.VideoCapture(0)
        cap.set(3,340)
        cap.set(4,200)
        cnt = 0
        while True:
            success,img = cap.read()
            #pathImage = dirPath + str(cnt) + ".jpg"
            pathImgTrim = dirPath + str(cnt) + ".jpg"
            self.__dirImgTrim = dirPath
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            imgResults = img.copy()
            #カスケード分類気の特徴量を取得する
            cascade = cv2.CascadeClassifier(cascadePath)
            facerect = cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))
            color = (255,0,255)
            self.__imgTrim = []
            #検出した場合
            if len(facerect) > 0:
                for rect in facerect:
                    cv2.rectangle(imgResults,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),color,thickness=2)
                    x = rect[0]
                    y = rect[2]
                    xw,yh = tuple(rect[0:2]+rect[2:4])
                    imgTrim = img[y:yh,x:xw]
                    self.__imgTrim.append(imgTrim)
                    #保存
                    cv2.imwrite(pathImgTrim,imgTrim)
                    #表示
                    cv2.imshow("Results",imgResults)
                    cv2.imshow("ImageTrim",imgTrim)
            cv2.imshow("Original",img)
            if cnt >= 99:
                cnt = 0
            else:
                cnt += 1
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        pass
    
    def __Rescale(self):
        self.__myFileName = os.listdir(self.__dirImgTrim) 
        #print(self.__myFileName)
        self.__myFilePath = []
        self.__ndArrayImg = []
        self.__h = []
        self.__w = []
        self.__c = []
        for i in range(0,len(self.__myFileName)):
            self.__myFilePath.append(self.__dirImgTrim + str(self.__myFileName[i]))
            self.__ndArrayImg.append(np.array(Image.open(self.__myFilePath[i]),np.float))
            self.__h.append(self.__ndArrayImg[i].shape[0])
            self.__w.append(self.__ndArrayImg[i].shape[1])
            self.__c.append(self.__ndArrayImg[i].shape[2])
        self.__defH = 0
        self.__defW = 0
        self.__defC = 0
        for i in range(0,len(self.__myFileName)):
            self.__defH += self.__h[i]
            self.__defW += self.__w[i]
            self.__defC += self.__c[i]
        self.__defH /= len(self.__h)
        self.__defW /= len(self.__w)
        self.__defC /= len(self.__c)
        self.__defH = int(self.__defH)
        self.__defW = int(self.__defW)
        self.__defC = int(self.__defC)
        self.__resImg = []
        for i in range(0,len(self.__myFileName)):
            self.__resImg.append(cv2.resize(self.__ndArrayImg[i],dsize=(self.__defW,self.__defH)))
            pass
        pass
    
    def showRescale(self):
        for i in range(0,len(self.__myFileName)):
            print("h    :{}".format(self.__resImg[i].shape[0]))
            print("w    :{}".format(self.__resImg[i].shape[1]))
            print("c    :{}".format(self.__resImg[i].shape[2]))

#endregion

dataSet = MyDataset(dirPath="Resources/Image/",tagName="Kodama")


#ニューラルネットワークモデルの引数はtorch.tensorを使ったtensor型