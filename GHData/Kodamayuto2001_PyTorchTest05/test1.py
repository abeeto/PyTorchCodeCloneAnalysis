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
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28*28,400)
        self.fc2 = nn.Linear(400,2)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class MyDataLoader:
    def __init__(self,dirPath,batch_size,imgSize):
        self.dataSet(dirPath,imgSize)
        self.dataLoader(batch_size)
    
    def dataSet(self,dirPath,imgSize):
        self.ImageFolder_data = torchvision.datasets.ImageFolder(dirPath,transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(imgSize),
            transforms.ToTensor(),
        ]))

    def dataLoader(self,batch_size):
        dataloader = torch.utils.data.DataLoader(self.ImageFolder_data,batch_size=batch_size,shuffle=True,num_workers=2)


#mX1 = MyDataset(dirImgPath="Resources/train/Kodama/",tagName="Kodama")
#mX2 = MyDataset(dirImgPath="Resources/train/Kodama2/",tagName="Kodama2")
#mY1 = MyDataset(dirImgPath="Resources/test/amadok1/",tagName="amadok1")
#mY2 = MyDataset(dirImgPath="Resources/test/amadok2/",tagName="amadok2")


net = Net()
#print(net)

mDataLoader = MyDataLoader(dirPath="Resources/train/",batch_size=32,imgSize=28)