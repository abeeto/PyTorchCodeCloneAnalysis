import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torchvision
import numpy as np
import torch 
import cv2 
import os 

class Net(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)

    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

class CNN(torch.nn.Module):
    def __init__(self,num,inputSize,hidden1):
        super(CNN,self).__init__()
        self.iSize  = inputSize
        self.conv1  = torch.nn.Conv2d(1,4,3)
        self.bn1    = torch.nn.BatchNorm2d(4)
        self.pool   = torch.nn.MaxPool2d(2,2)
        self.conv2  = torch.nn.Conv2d(4,16,3)
        self.bn2    = torch.nn.BatchNorm2d(16)
        self.fc1    = torch.nn.Linear(16*38*38,hidden1)
        self.fc2    = torch.nn.Linear(hidden1,num)

    def forward(self,x):
        x = self.conv1(x)
        # print(x.size())
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.pool(x)    
        # print(x.size())
        x = x.view(-1,16*38*38)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


class AI:
    def __init__(
        self,
        EPOCH       =   40,
        IMAGE_SIZE  =   160,
        HIDDEN_1    =   320,
        LR          =   0.000005,
        model_num   =   20,
        TRAIN_DIR   =   "train-dataset",
        # TEST_DIR    =   "test-dataset",
        PT_NAME     =   "nn.pt",
        LOSS_PNG    =   "loss.png",
        ACC_PNG     =   "acc.png"
        ):
        self.EPOCH      =   EPOCH
        self.IMAGE_SIZE =   IMAGE_SIZE
        self.HIDDEN_1   =   HIDDEN_1
        self.LR         =   LR 
        self.MODEL      =   CNN(num=20,inputSize=IMAGE_SIZE,hidden1=HIDDEN_1)
        self.OPTIMIZER  =   torch.optim.Adam(params=self.MODEL.parameters(),lr=self.LR)
        self.TRAIN_DIR  =   TRAIN_DIR
        # self.TEST_DIR   =   TEST_DIR
        self.PT_NAME    =   PT_NAME
        self.LOSS_PNG   =   LOSS_PNG
        self.ACC_PNG    =   ACC_PNG

        ######------学習用データローダー-----#####
        train_data  =   torchvision.datasets.ImageFolder(
            root=self.TRAIN_DIR,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((self.IMAGE_SIZE,self.IMAGE_SIZE)),
                torchvision.transforms.ToTensor(),
            ])
        )
        self.train_data =   torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=True
        )

    def train(self):
        for data in tqdm(self.train_data):
            x,target    =   data 
            # print(target)
            self.OPTIMIZER.zero_grad()
            output      =   self.MODEL(x)
            ######------損失関数-----#####
            loss = F.nll_loss(output,target)
            loss.backward()
            self.OPTIMIZER.step()
        return loss 

    def test(self,test_dir,name,label):
        self.TEST_DIR   =   test_dir
        #   学習停止
        self.MODEL.eval()

        total   =   0
        correct =   0
        with torch.no_grad():
            for f in tqdm(os.listdir(self.TEST_DIR)):
                path    =   self.TEST_DIR+"/"+f
                x,target=   self.maesyori(path,label)
                output  =   self.MODEL(x)
                _,p     =   torch.max(output.data,1)
                total   +=  target.size(0)
                correct +=  (p == target).sum().item()
        percent =   100*correct/total
        print("{}{:>10f}".format(name,percent))
        return percent

    def maesyori(self,path,label):
        img     =   cv2.imread(path)
        img     =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img     =   cv2.resize(img,(self.IMAGE_SIZE,self.IMAGE_SIZE))
        img     =   np.reshape(img,(1,self.IMAGE_SIZE,self.IMAGE_SIZE))
        img     =   np.transpose(img,(1,2,0))
        img     =   torchvision.transforms.ToTensor()(img)
        img     =   img.unsqueeze(0)
        label   =   torch.Tensor([label]).long()
        return img,label

    def save_loss_png(self,loss):
        plt.figure()
        plt.plot(range(1,self.EPOCH+1),loss,label="trainLoss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(self.LOSS_PNG)
        plt.close()
    
    def save_acc_png(self,acc,name):
        plt.figure()
        plt.plot(range(1,self.EPOCH+1),acc,label=str(name))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(str(name)+"_"+self.ACC_PNG)
        plt.close()

    def save_model(self):
        torch.save(self.MODEL.state_dict(),self.PT_NAME)

class AI_30Classes:
    def __init__(self,root_train_dir,root_test_dir,PT_NAME="cnn_da_60000.pt",LOSS_PNG="loss_da_60000.png",ACC_PNG="acc_da_60000.png"):
        ai  =   AI(TRAIN_DIR=root_train_dir,PT_NAME=PT_NAME,LOSS_PNG=LOSS_PNG,ACC_PNG=ACC_PNG)

        denjyo_classes  =   [
            [

                [],[],[]
            ]  for i in os.listdir(root_test_dir)
        ]

        loss    =   []
        for e in range(ai.EPOCH):
            loss.append(ai.train())
            for i,name in enumerate(os.listdir(root_test_dir)):
                test_path               =   root_test_dir+name+"/"
                denjyo_classes[i][0]    =   name 
                denjyo_classes[i][1]    =   i
                denjyo_classes[i][2].append(ai.test(test_path,name,i))
        

        ai.save_loss_png(loss)

        for x in denjyo_classes:
            # print(x)
            ai.save_acc_png(x[2],x[0]) 

        #   保存
        ai.save_model()


if __name__ == "__main__":
    ai_30classes    =   AI_30Classes(
        root_train_dir="../DataAugmentation2/data_augmentation_10000×6_/",
        root_test_dir="test_6_DataAugmentation/"
    )