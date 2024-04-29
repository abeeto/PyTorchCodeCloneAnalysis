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
class MyDataset:
    def __init__(self,dirImgPath,tagName,nTrain,nVal):
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
    
    def DataSplit(self):
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


net = Net(imgSize=225)
mDL = MyDataLoader(trainDir="./train",testDir="./test",batchSize=4,imgSize=225,numWorkers=2)
optimizer = optim.Adam(net.parameters(),lr=0.01)
criterion = nn.MSELoss()

def train(epoch,name):
    if not os.mkdir("model/"+name):
        os.mkdir("models/"+name)
    train_loss = np.array([])
    test_loss = np.array([])

print(type(np.array([])))




