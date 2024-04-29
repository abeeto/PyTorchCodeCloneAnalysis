import os
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import cv2
import sys
import time
from tqdm import tqdm

class MyDataset:
    r"""
        Please turn on the camera.
        And download Cascade Classifier.

        dirImgPath = "path/"
        tagName = "tagName"
        dataNum = 1000
    """
    def __init__(self,dirImgPath,tagName,dataNum):
        self.MyMakeDataSplit(dirImgPath,dataNum)
        pass

    def MyMakeDataSplit(self,dirPath,dataNum):
        dirTrain = dirPath + "train/"
        dirTest = dirPath + "test/"
        try:
            os.makedirs(dirTrain)
            os.makedirs(dirTest)
        except FileExistsError:
            pass
        cascadePath = "opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascadePath)
        cap = cv2.VideoCapture(0)
        cap.set(3,340)
        cap.set(4,200)
        cnt = 0
        XXX = 0
        #train : test = 8 : 2
        nTrain = int(dataNum*0.8)
        nTest  = int(dataNum-nTrain)
        #print(nTrain)
        #print(nTest)
        while True:
            success,img = cap.read()
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            imgResults = img.copy()
            cascade = cv2.CascadeClassifier(cascadePath)
            facerect = cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))
            color = (255,0,255)
            self.__imgTrim = []
            if len(facerect) > 0:
                for rect in facerect:
                    cv2.rectangle(imgResults,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),color,thickness=2)
                    x = rect[0]
                    y = rect[2]
                    xw,yh = tuple(rect[0:2]+rect[2:4])
                    imgTrim = img[y:yh,x:xw]
                    self.__imgTrim.append(imgTrim)
                    if cnt < nTrain:
                        cv2.imwrite(dirTrain+str(cnt)+".jpg",imgTrim)
                    elif cnt >= nTrain and cnt < dataNum:
                        cv2.imwrite(dirTest+str(cnt)+".jpg",imgTrim)
                    else:
                        pass
                    if (cnt % int(dataNum / 100)) == 0:
                        if XXX == 100:
                            pass
                        else:
                            XXX += 1 
                    cv2.imshow("Results",imgResults)
                    cv2.imshow("ImageTrim",imgTrim)
                    if cnt >= dataNum:
                        cnt = 0
                        break
                    else:
                        cnt += 1
                    sys.stdout.write("\r data {:<3d}{}".format(XXX,"%"))
                    sys.stdout.flush()
            cv2.imshow("Original",img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            pass
        pass

    pass


class Net(nn.Module):
    def __init__(self,imgSize):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(imgSize*imgSize,400)
        self.fc2 = nn.Linear(400,2)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class MyDataLoader:
    def __init__(self,trainDir,testDir,batchSize,imgSize,numWorkers):
        self.dataSet(trainDir,testDir,imgSize)
        self.dataLoader(batchSize,numWorkers)
        pass
    def dataSet(self,trainDir,testDir,imgSize):
        self.trainData = torchvision.datasets.ImageFolder(
            root=trainDir,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(imgSize),
                transforms.ToTensor(),
            ])
        )
        self.testData = torchvision.datasets.ImageFolder(
            root=testDir,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(225),
                transforms.ToTensor(),
            ])
        )
    def dataLoader(self,batchSize,numWorkers):
        self.trainDataLoader = torch.utils.data.DataLoader(self.trainData,batch_size=batchSize,shuffle=True,num_workers=numWorkers)
        self.testDataLoader = torch.utils.data.DataLoader(self.testData,batch_size=batchSize,shuffle=True,num_workers=numWorkers)

mDataset = MyDataset(dirImgPath="Resources/",tagName="Kodama",dataNum=1000)

