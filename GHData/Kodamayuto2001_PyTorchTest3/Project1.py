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


class FaceLandmarkDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset"""
    def __init__(self,csv_file,root_dir,transform=None):
        self.landmark_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform 
    
    def __len__(self):
        return len(self.landmark_frame)
    
    def __getitem__(self,idx):
        pass
        

class MyDataset:
    def __init__(self,pathImgTrim,tagName):
        self.__setPath(pathImgTrim)
        self.__setTag(tagName)
        self.__CSV_transform()
        self.__readCSV()

    def __setPath(self,pathImgTrim=""):
        self.pathImgTrim = pathImgTrim
        self.myFileName = os.listdir(self.pathImgTrim)

    def __setTag(self,tagName="Kodama"):
        self.tag = tagName

    def showFileName(self):
        print(self.myFileName)
        print("{0}のディレクトリの中にあるファイル数は{1}です。".format(self.pathImgTrim,len(self.myFileName)))
        print(type(self.myFileName))
    
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
        self.CSV_Data = pd.DataFrame(data=self.ListFN,index=self.ListTag)
        print(self.CSV_Data)
        self.osPathCSV_1 = "Data/"
        try:
            os.makedirs(self.osPathCSV_1)
        except FileExistsError:
            pass
        
        self.pathCSV = self.osPathCSV_1+"data_"+str(self.tag)+".csv"
        #self.CSV_Data.to_csv(self.pathCSV)
    
    def __readCSV(self):
        self.pdCSV_frame = pd.read_csv(self.pathCSV)
        print("pd.read_csv={}".format(type(self.pdCSV_frame)))
        self.pdCSV_frame = np.asarray(self.pdCSV_frame)
        print("np.asarray={}".format(type(self.pdCSV_frame)))
        #self.pdCSV_frame = self.pdCSV_frame.astype("float64")
        #print("astype(float64)={}".format(self.pdCSV_frame))
        self.pdCSV_frame = torch.from_numpy(self.pdCSV_frame)
        print("torch.from_numpy={}".format(self.pdCSV_frame))
        #face_dataset = FaceLandmarkDataset(csv_file=self.pathCSV,root_dir=self.osPathCSV_1)
        
        #for i in range(len())
        pass
    def showCSV_File(self):
        print(self.pdCSV_frame)
        print(type(self.pdCSV_frame))



x = MyDataset(pathImgTrim="Resources/ImgTrim",tagName="Kodama")

#x.showFileName()
#x.showCSV_File()