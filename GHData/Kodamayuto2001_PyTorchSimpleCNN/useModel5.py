import os
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self,num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num)
 
    def forward(self, x):
        x = self.conv1(x)
        #print(x.size())
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.pool(x)    
        #print(x.size())
        x = x.view(-1,16*5*5)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
 
if __name__ == "__main__":
    model = Net(6)
    PATH = "model5.pt"
    model.load_state_dict(torch.load(PATH))
    model.eval()

    testData = torchvision.datasets.ImageFolder(root="Resources/test/",transform=transforms.Compose([transforms.Grayscale(),transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),]))

    testData = torch.utils.data.DataLoader(testData,batch_size=1,shuffle=False,num_workers=os.cpu_count())

    andou = 0
    higashi = 0
    kataoka = 0
    kodama = 0
    masuda = 0
    suetomo = 0
    for data in testData:
        test,label = data 
        #print(test)
        output = model(test)
        _,predicted = torch.max(output.data,1)

        npimg = test.numpy()
        #print(npimg.shape) #(1, 1, 28, 28)
        npimg = npimg.reshape([1,28,28])
        #print(npimg.shape)
        #plt.imshow(np.transpose(npimg,(1,2,0)))
        #plt.show()
        print("予測値：{}     正解ラベル：{}".format(predicted,label))
        if int(label) == int(0):
            if predicted == int(0):
                andou += 1
        if int(label) == int(1):
            if predicted == int(1):
                higashi += 1
        if int(label) == int(2):
            if predicted == int(2):
                kataoka += 1
        if int(label) == int(3):
            if predicted == int(3):
                kodama += 1
        if int(label) == int(4):
            if predicted == int(4):
                masuda += 1
        if int(label) == int(5):
            if predicted == int(5):
                suetomo += 1
    print("andou : {} %".format((andou/20)*100))
    print("higashi : {} %".format((higashi/20)*100))
    print("kataoka : {} %".format((kataoka/20)*100))
    print("kodama : {} %".format((kodama/20)*100))
    print("masuda : {} %".format((masuda/20)*100))
    print("suetomo : {} %".format((suetomo/20)*100))
        
    