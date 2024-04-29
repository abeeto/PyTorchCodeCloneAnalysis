#region __IMPORT__
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
import torchvision.utils as vutils
import matplotlib.pyplot as plt
#endregion !__IMPORT__
class MyDataset:
    r"""
        Please turn on the camera.
        And download Cascade Classifier.

        dirImgPath = "path/"
        tagName = "tagName"
        dataNum = 1000
    """
    def __init__(self,dirImgPath,tagName,dataNum):
        self.MyMakeDataSplit(dirImgPath,dataNum,tagName)
        pass

    def MyMakeDataSplit(self,dirPath,dataNum,tagName):
        dirTrain = dirPath + "train/" + tagName + "/"
        self.dirTrain = dirTrain
        dirTest = dirPath + "test/" + tagName + "/"
        self.dirTest = dirTest
        try:
            os.makedirs(dirTrain)
            os.makedirs(dirTest)
        except FileExistsError:
            if len(os.listdir(dirTrain)) + len(os.listdir(dirTest)) >= int(dataNum-1):
                return 0
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
   
class MyDataLoader:
    r"""
    trainRootDir = "path/to/train/"
    testRootDir = "path/to/test/"
    imgSize = 28
    batchSize = 1
    numWorkers = 3
        --numWorkers = the number of tagNames
    """
    def __init__(self,trainRootDir,testRootDir,imgSize,batchSize,numWorkers):
        #self.__my_collate_fn(batch=batchSize)
        self.__dataSet(trainRootDir=trainRootDir,testRootDir=testRootDir,imgSize=imgSize)
        self.__dataLoader(batchSize=batchSize,numWorkers=numWorkers)
        pass
    def __my_collate_fn(self,batch):
        pass 
    def __dataSet(self,trainRootDir,testRootDir,imgSize):
        self.trainData = torchvision.datasets.ImageFolder(root=trainRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),]))
        self.testData = torchvision.datasets.ImageFolder(root=testRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),]))
        pass
    def __dataLoader(self,batchSize,numWorkers):
        self.trainDataLoaders = torch.utils.data.DataLoader(self.trainData,batch_size=batchSize,shuffle=True,num_workers=numWorkers)
        self.testDataLoaders = torch.utils.data.DataLoader(self.testData,batch_size=batchSize,shuffle=True,num_workers=numWorkers)
        pass
    def imshow(self):
       print(type(self.trainDataLoaders))
       dataiter = iter(self.trainDataLoaders)
       img,labels = dataiter.next()
       img = torchvision.utils.make_grid(img)
       img = img /2 + 0.5
       npimg = img.numpy()
       plt.imshow(np.transpose(npimg,(1,2,0)))
       plt.show()
       pass
    def getDataLoaders(self):
        r"""
        RETURN
            --trainDataLoaders,
            --testDataLoaders
        """
        return self.trainDataLoaders,self.testDataLoaders
    pass
"""
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,4)
        pass
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        pass
    pass
"""
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #1 input image channel,6 output channels,3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # an affine operation: y=Wx+b
        self.fc1 = nn.Linear(16*6*6,120)#6*6 from image dimension
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,3)
    def forward(self,x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        #If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)),1)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    dNum = 100
    Person1 = MyDataset(dirImgPath="Resources/",tagName="Yuto",dataNum=dNum)
    Person2 = MyDataset(dirImgPath="Resources/",tagName="Shouta",dataNum=dNum)
    Person3 = MyDataset(dirImgPath="Resources/",tagName="Ouki",dataNum=dNum)
    mDataLoader = MyDataLoader(trainRootDir="Resources/train/",testRootDir="Resources/test/",imgSize=32,batchSize=1,numWorkers=3)
    mDataLoader.imshow()

    trainLoader,testLoader = mDataLoader.getDataLoaders()

    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.01)

    for epoch in tqdm(range(2)):
        for i,data in tqdm(enumerate(trainLoader,0)):
            inputs,labels = data
            #print("{}--{}--{}".format(type(inputs),labels,inputs.size()))
            inputs = inputs.view(1, -1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
    sys.stdout.write("\r {}".format("Finished Training"))

    

