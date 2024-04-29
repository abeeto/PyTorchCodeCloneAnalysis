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
import pandas as pd 
#from tqdm import tqdm
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
    def __init__(self,dirImgPath,dataNum):
        print("tagName >",end="")
        tagName = input()
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
        cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascadePath)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920/10)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080/10)
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
            #cascade = cv2.CascadeClassifier(cascadePath)
            facerect = cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))
            color = (255,0,255)
            self.__imgTrim = []
            if len(facerect) > 0:
                for (x,y,w,h) in facerect:
                    #左上(原点)=(x,y)
                    #左上が(0,0)で、右下が(1920,1080)となっている
                    #顔を囲む
                    #rectangle(img,leftTop,rightBottom,color)
                    cv2.rectangle(imgResults,(x,y),(x+w,y+h),color,thickness=2)
                    imgTrim = img[y:y+h,x:x+w]
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
                        cv2.destroyAllWindows()
                        print("--------------------------------------------")
                        return 0
                    else:
                        cnt += 1
                    sys.stdout.write("\r data {:<3d}{}".format(XXX,"%"))
                    sys.stdout.flush()
            cv2.imshow("Original",img)
        
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            pass
        pass
    pass

class MyDataLoader:
    r"""
    trainRootDir = "path/to/train/"
    testRootDir = "path/to/test/"
    imgSize = 28
    batchSize = 128
    numWorkers = 3
        --numWorkers = the number of tagNames
    """
    def __init__(self,trainRootDir,testRootDir,imgSize,batchSize):
        #self.__my_collate_fn(batch=batchSize)
        self.__dataSet(trainRootDir=trainRootDir,testRootDir=testRootDir,imgSize=imgSize)
        self.__dataLoader(batchSize=batchSize)
        pass
    def __my_collate_fn(self,batch):
        pass 
    def __dataSet(self,trainRootDir,testRootDir,imgSize):
        self.trainData = torchvision.datasets.ImageFolder(root=trainRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),]))
        self.testData = torchvision.datasets.ImageFolder(root=testRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),]))
        pass
    def __dataLoader(self,batchSize):
        self.trainDataLoaders = torch.utils.data.DataLoader(self.trainData,batch_size=1,shuffle=True)
        self.testDataLoaders = torch.utils.data.DataLoader(self.testData,batch_size=1,shuffle=True)
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
        return {"train":self.trainDataLoaders,"test":self.testDataLoaders}
    pass

class Net(nn.Module):
    def __init__(self,num):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 1000)
        self.fc2 = torch.nn.Linear(1000,1000)
        self.fc3 = torch.nn.Linear(1000, num)
 
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    print("撮影する(y):撮影しない(n):",end="")
    isMakeData = input()

    #学習回数
    epoch = 20

    #学習結果保存用
    history = {
        "trainLoss":[],
        "testLoss":[],
        "testAcc":[],
    }

    #データセット
    Person = []
    cnt = 0
    print("Class num : ",end="")
    num = input()
    #print(num)
    if isMakeData == "y":
        while True:
            Person.append(MyDataset(dirImgPath="Resources/",dataNum=100))
            cnt += 1
            if cnt == int(num):
                break
    else:
        pass

    #ネットワークを構築
    net: torch.nn.Module = Net(num=int(num))

    #データローダーを取得
    mDataLoader = MyDataLoader(trainRootDir="Resources/train/",testRootDir="Resources/test/",imgSize=28,batchSize=1)
    #mDataLoader.imshow()
    print(mDataLoader.trainData)

    loaders = mDataLoader.getDataLoaders()
    print(loaders)
    print(loaders["train"])
    #print(loaders["train"].shape())
    optimizer = torch.optim.Adam(params=net.parameters(),lr=0.001)

    for e in range(epoch):
        for i,data in enumerate(loaders["train"],0):
            inputs,label = data 
            #print(i)
            #print(inputs)
            print(label)
            inputs = inputs.view(-1,28*28)
            optimizer.zero_grad()
            output = net(inputs)
            loss = F.nll_loss(output,label)
            loss.backward()
            optimizer.step()
            
            print("loss : {}".format(loss.item()))
        history["trainLoss"].append(loss)
        """ Test Part """
        #学習のストップ
        net.eval()
        testLoss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loaders["test"]:
                #print(i)
                #print(data)
                inputs,label = data
                #print(inputs)
                print(label)
                inputs = inputs.view(-1,28*28)
                output = net(inputs)
                #testLoss += F.nll_loss(output,label,reduction="sum").item()
                #pred = output.argmax(dim=1,keepdim=True)
                #correct += pred.eq(label.view_as(pred)).sum().item()
                #print(correct)
                _,predicted = torch.max(output.data,1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        #testLoss /= 100
        print("Test Loss (avg): {}".format(testLoss))
        print("Accuracy of the network on the 100 test images: %d %%" % (100*correct/total))

        history['testLoss'].append(testLoss)
        history['testAcc'].append(100*correct/total)

    # 結果の出力と描画
    print(history)
    df = pd.DataFrame(history)
    df.to_excel('0.001relu.xlsx')
    plt.figure()
    plt.plot(range(1, epoch+1), history['trainLoss'], label='trainLoss')
    plt.plot(range(1, epoch+1), history['testLoss'], label='testLoss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss0.001relu.png')
 
    plt.figure()
    plt.plot(range(1, epoch+1), history['testAcc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.savefig('test_acc0.001relu.png')


    #Save
    PATH = "model.pt"
    torch.save(net.state_dict(), PATH)






