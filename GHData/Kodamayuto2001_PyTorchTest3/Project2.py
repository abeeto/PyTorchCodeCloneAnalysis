from __future__ import division, print_function

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

class MyDataset:
    def __init__(self,pathImgTrim,tagName):
        self.__setPath(pathImgTrim)
        self.__setTag(tagName)
        self.__ndarrayTransform()
        self.__CSV_transform()
        #self.__toTensor_transform()

    def __setPath(self,pathImgTrim):
        self.pathImgTrim = pathImgTrim
        self.myFileName = os.listdir(self.pathImgTrim)
        print(type(self.myFileName))
        print(len(self.myFileName))

    def __setTag(self,tagName="Kodama"):
        self.tag = tagName

    def __ndarrayTransform(self):
        self.myFilePath = self.pathImgTrim+str(self.myFileName[0])
        self.ndAraayImg = np.array(Image.open(self.myFilePath),np.float64)
        print(type(self.ndAraayImg))
        print(self.ndAraayImg)
        print(self.ndAraayImg.shape)
        cv2.imshow("Image",self.ndAraayImg)
        cv2.imshow("Org",cv2.imread("Resources/ImgTrim/0.jpg"))
        cv2.waitKey(10000)

    def __BinaryTransform(self):
        pass

    def __CSV_transform(self):
        self.cnt = 0
        self.ListFN = []
        self.ListTag = []
        while self.cnt < len(self.myFileName):
            #print(self.myFileName[self.cnt])
            self.ListFN.append(self.myFileName[self.cnt])
            self.ListTag.append(self.tag)

            self.cnt += 1
        #print(self.ListFN)
        #print(self.ListTag)
        self.CSV_Data = pd.DataFrame(data=self.ListTag,index=self.ListFN)
        print(self.CSV_Data)
        self.osPathCSV_1 = "Data/"
        try:
            os.makedirs(self.osPathCSV_1)
        except FileExistsError:
            pass
        
        self.pathCSV = self.osPathCSV_1+"data_"+str(self.tag)+".csv"
        self.CSV_Data.to_csv(self.pathCSV)
    
    def __toTensor_transform(self):
        self.sample = {"FileName":self.ListFN,"label":self.ListTag}
        #画像読み込み
        im = []
        #for i in range(0,len(self.))
        
        #画像サイズ(幅、高さ)を取得


        #多次元配列に変更


    def showFileName(self):
        print(self.myFileName)

    def showCSV_File(self):
        print(self.CSV_Data)

x = MyDataset(pathImgTrim="Resources/ImgTrim/",tagName="Kodama")
x.showCSV_File()
