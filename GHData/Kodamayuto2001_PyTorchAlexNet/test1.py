"""今回はGrayScale変換なしでAlexNetに入力3チャネルでやろうと思う"""

# region __IMPORT__
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
# from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
#from torchviz import make_dot 
# endregion !__IMPORT__

class MyDataset:
    r"""
        Please turn on the camera.
        And download Cascade Classifier.
        dirImgPath = "path/"
        tagName = "tagName"
        dataNum = 1000
    """

    def __init__(self, dirImgPath, dataNum):
        print("tagName >", end="")
        tagName = input()
        self.MyMakeDataSplit(dirImgPath, dataNum, tagName)
        pass

    def MyMakeDataSplit(self, dirPath, dataNum, tagName):
        dirTrain = dirPath + "train/" + tagName + "/"
        dirTest = dirPath + "test/" + tagName + "/"
        try:
            os.makedirs(dirTrain)
            os.makedirs(dirTest)
        except FileExistsError:
            if len(os.listdir(dirTrain)) + len(os.listdir(dirTest)) >= int(
                    dataNum - 1):
                return 0
            pass
        cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascadePath)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 10)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 10)
        cnt = 0
        XXX = 0
        # train : test = 8 : 2
        nTrain = int(dataNum * 0.8)
        nTest = int(dataNum - nTrain)
        # print(nTrain)
        # print(nTest)
        while True:
            success, img = cap.read()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgResults = img.copy()
            # cascade = cv2.CascadeClassifier(cascadePath)
            facerect = cascade.detectMultiScale(imgGray, scaleFactor=1.1,
                                                minNeighbors=2,
                                                minSize=(10, 10))
            color = (255, 0, 255)
            self.__imgTrim = []
            if len(facerect) > 0:
                for (x, y, w, h) in facerect:
                    # 左上(原点)=(x,y)
                    # 左上が(0,0)で、右下が(1920,1080)となっている
                    # 顔を囲む
                    # rectangle(img,leftTop,rightBottom,color)
                    cv2.rectangle(imgResults, (x, y), (x + w, y + h), color,
                                  thickness=2)
                    imgTrim = img[y:y + h, x:x + w]
                    self.__imgTrim.append(imgTrim)
                    if cnt < nTrain:
                        cv2.imwrite(dirTrain + str(cnt) + ".jpg", imgTrim)
                    elif cnt >= nTrain and cnt < dataNum:
                        cv2.imwrite(dirTest + str(cnt) + ".jpg", imgTrim)
                    else:
                        pass
                    if (cnt % int(dataNum / 100)) == 0:
                        if XXX == 100:
                            pass
                        else:
                            XXX += 1
                    cv2.imshow("Results", imgResults)
                    cv2.imshow("ImageTrim", imgTrim)
                    if cnt >= dataNum:
                        cnt = 0
                        cv2.destroyAllWindows()
                        print("--------------------------------------------")
                        return 0
                    else:
                        cnt += 1
                    sys.stdout.write("\r data {:<3d}{}".format(XXX, "%"))
                    sys.stdout.flush()
            cv2.imshow("Original", img)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            pass
        pass

    pass

class MyGTA5Dataset:
    """
    mp4Path = "path/1.mp4"
    """
    def __init__(self,mp4Path,tagName,dirImgPath,dataNum):
        self.MyMakeDataSplit(mp4Path,tagName,dirImgPath,dataNum)
        pass 
    
    def MyMakeDataSplit(self,mp4Path,tagName,dirPath,dataNum):
        dirTrain = dirPath + "train/" + str(tagName) + "/"
        dirTest = dirPath + "test/" + str(tagName) + "/"
        try:
            os.makedirs(dirTrain)
            os.makedirs(dirTest)
        except FileExistsError:
            if len(os.listdir(dirTrain)) + len(os.listdir(dirTest)) >= int(dataNum -1):
                return 0
            pass 
        cap = cv2.VideoCapture(mp4Path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cnt = 0
        XXX = 0
        # train : test = 8 : 2
        nTrain = int(dataNum * 0.8)
        nTest = int(dataNum -nTrain)
        while True:
            success,img = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES,6*cnt)
            if cnt < nTrain:
                cv2.imwrite(dirTrain + str(cnt) + ".jpg",img)
            elif cnt >= nTrain and cnt < dataNum:
                cv2.imwrite(dirTest + str(cnt) + ".jpg",img)
            else:
                pass 
            cv2.imshow("Image",img)
            if (cnt % int(dataNum / 100)) == 0:
                if XXX == 100:
                    pass
                else:
                    XXX += 1
            if cnt >= dataNum:
                cnt = 0
                cv2.destroyAllWindows()
                return 0
            else:
                cnt += 1
            sys.stdout.write("\r data {:<3d}{}".format(XXX, "%"))
            sys.stdout.flush()
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

    def __init__(self, trainRootDir, testRootDir, imgSize, batchSize,num_workers):
        # self.__my_collate_fn(batch=batchSize)
        self.__dataSet(trainRootDir=trainRootDir, testRootDir=testRootDir,
                       imgSize=imgSize)
        self.__dataLoader(batchSize=batchSize,num_workers=num_workers)
        pass

    def __my_collate_fn(self, batch):
        pass

    """
    def __dataSet(self, trainRootDir, testRootDir, imgSize):
        self.trainData = torchvision.datasets.ImageFolder(root=trainRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),]))
        self.testData = torchvision.datasets.ImageFolder(root=testRootDir,transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),]))
        pass
    """

    def __dataSet(self,trainRootDir,testRootDir,imgSize):
        self.trainData = torchvision.datasets.ImageFolder(root=trainRootDir,transform=transforms.Compose([transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5,)),]))
        self.testData = torchvision.datasets.ImageFolder(root=testRootDir,transform=transforms.Compose([transforms.Resize((imgSize,imgSize)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5,)),]))
        pass 

    def __dataLoader(self, batchSize,num_workers):
        self.trainDataLoaders = torch.utils.data.DataLoader(self.trainData,
                                                            batch_size=1,
                                                            shuffle=True,num_workers=num_workers)
        self.testDataLoaders = torch.utils.data.DataLoader(self.testData,
                                                           batch_size=1,
                                                           shuffle=True,num_workers=num_workers)
        pass

    def imshow(self):
        print(type(self.trainDataLoaders))
        dataiter = iter(self.trainDataLoaders)
        img, labels = dataiter.next()
        img = torchvision.utils.make_grid(img)
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        pass

    def getDataLoaders(self):
        r"""
        RETURN
            --trainDataLoaders,
            --testDataLoaders
        """
        return {"train": self.trainDataLoaders, "test": self.testDataLoaders}

    pass

class AlexNet(nn.Module):
    """
    5つの畳み込み層と3つの全結合層
    """
    def __init__(self,num):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 =  nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(384)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.fc1 = nn.Linear(384*6*6,30)
        self.fc2 = nn.Linear(30,9)
        self.fc3 = nn.Linear(9,4)

        
    
    def forward(self,x):
        #print(x.size())
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        #print(x.size())
        x = self.pool1(x)
        #print(x.size())
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        #print(x.size())
        x = self.pool2(x)
        #print(x.size())
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.bn5(x)
        x = self.pool3(x)
        #print(x.size())
        x = x.view(-1,384*6*6)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
        


if __name__ == "__main__":
    #print("撮影する(y):撮影しない(n):",end="")
    #isMakeData = input()

    #学習回数

    epoch = 10

    #学習結果保存用
    history = {
        "trainLoss":[],
        "testLoss":[],
        "testAcc":[],
    }

    #データセット
    Person = []
    cnt = 0
    # print("Class num : ",end="")
    # num = input()
    #print(num)
    # if isMakeData == "y":
    #     while True:
    #         Person.append(MyDataset(dirImgPath="Resources/",dataNum=100))
    #         cnt += 1
    #         if cnt == int(num):
    #             break
    # else:
    #     pass

    data1 = MyGTA5Dataset(mp4Path="GTA5/1.mp4",tagName=1,dirImgPath="Resources/",dataNum=300)
    data2 = MyGTA5Dataset(mp4Path="GTA5/2.mp4",tagName=2,dirImgPath="Resources/",dataNum=300)
    data3 = MyGTA5Dataset(mp4Path="GTA5/3.mp4",tagName=3,dirImgPath="Resources/",dataNum=300)
    data4 = MyGTA5Dataset(mp4Path="GTA5/4.mp4",tagName=4,dirImgPath="Resources/",dataNum=300)
    num = 4


    #ネットワークを構築
    net = AlexNet(num=int(num))

    #データローダーを取得
    #DataLoaderのnum_workers(CPUのコア数)を設定する
    num_workers = os.cpu_count()
    mDataLoader = MyDataLoader(trainRootDir="Resources/train/",testRootDir="Resources/test/",imgSize=512,batchSize=1,num_workers=num_workers)
    #mDataLoader.imshow()
    #print(mDataLoader.trainData)

    loaders = mDataLoader.getDataLoaders()
    #print(loaders)
    #print(loaders["train"])
    #print(loaders["train"].shape())
    optimizer = torch.optim.Adam(params=net.parameters(),lr=0.00005)

    for e in range(epoch):
        for i,data in enumerate(loaders["train"],0):
            inputs,label = data 
            #print(i)
            #print(inputs[0][0][0][0])
            #print(len(inputs[0][0][0]))
            #print(len(inputs[0][0]))
            #print(inputs.size())
            #torch.Size([1, 1, 28, 28])
            #print(label)
            #inputs = inputs.view(-1,28*28)
            #print(inputs.size())
            optimizer.zero_grad()
            output = net(inputs)
            loss = F.nll_loss(output,label)
            loss.backward()
            optimizer.step()
            
            print("loss : {}".format(loss.item()))
        history["trainLoss"].append(loss)

        
        
        ### Test Part ###
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
                #inputs = inputs.view(-1,28*28)
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
    df.to_excel('0.00005relu.xlsx')
    plt.figure()
    plt.plot(range(1, epoch+1), history['trainLoss'], label='trainLoss')
    plt.plot(range(1, epoch+1), history['testLoss'], label='testLoss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss0.00005relu.png')
 
    plt.figure()
    plt.plot(range(1, epoch+1), history['testAcc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.savefig('test_acc0.00005relu.png')


    #Save
    PATH = "model5.pt"
    torch.save(net.state_dict(), PATH)




