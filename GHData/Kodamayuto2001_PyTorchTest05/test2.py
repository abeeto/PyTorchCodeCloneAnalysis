import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import os
import cv2

import torchvision.datasets as dset
import torchvision
import torchvision.transforms as transforms


class MyDataset:
    def __init__(self,dirImgPath,tagName):
        self.MyMakeData(dirImgPath)
        pass

    def MyMakeData(self,dirPath):
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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        pass
    pass

class bsds_dataset(torch.utils.data.Dataset):
    def __init__(self, ds_main, ds_energy):
        self.dataset1 = ds_main
        self.dataset2 = ds_energy

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]

        return x1, x2

    def __len__(self):
        return len(self.dataset1)

a = MyDataset(dirImgPath="./images/whole",tagName="Kodama")
b = MyDataset(dirImgPath="./results/whole",tagName="Yuto")

original_imagefolder = './images/whole'
target_imagefolder = './results/whole'

original_ds = torchvision.datasets.ImageFolder(original_imagefolder, 
transform=transforms.ToTensor())
energy_ds = torchvision.datasets.ImageFolder(target_imagefolder, transform=transforms.ToTensor())

dataset = bsds_dataset(original_ds, energy_ds)
loader = DataLoader(dataset, batch_size=16)