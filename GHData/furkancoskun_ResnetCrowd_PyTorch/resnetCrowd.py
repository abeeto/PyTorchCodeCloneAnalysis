import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/resnetCrowd')

class BehaviourClassificationHead(nn.Module):
    def __init__(self):
        super(BehaviourClassificationHead, self).__init__()
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc.weight)

class DensityLevelClassificationHead(nn.Module):
    def __init__(self):
        super(DensityLevelClassificationHead, self).__init__()
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        x = self.fc(x)
        output = F.softmax(x, dim=1)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc.weight)

class CountingHead(nn.Module):
    def __init__(self):
        super(CountingHead, self).__init__()
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc(x)
        output = F.relu(x)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc.weight)

class HeatMapHead(nn.Module):
    def __init__(self):
        super(HeatMapHead, self).__init__()
        self.conv = nn.Conv2d(64, 1, 3, stride=1, padding=1)

    def forward(self, x):
        output = self.conv(x)
        return output

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.conv.weight)

class ResnetCrowd(nn.Module):
    def __init__(self):
        resnet18 = models.resnet18(pretrained = True)
        self.backbone = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.layer1
        )
        self.averagePool = nn.AvgPool2d(7, stride=1)
        self.fc32 = nn.Linear(154*84, 32)
        self.behavClsHead = BehaviourClassificationHead()
        self.densLevClsHead = DensityLevelClassificationHead()
        self.countingHead = CountingHead()
        self.heatMapHead = HeatMapHead()
        self.behavClsHeadLoss = nn.BCELoss()
        self.densLevClsHeadLoss = nn.CrossEntropyLoss()
        self.countingHeadLoss = nn.MSELoss()
        self.heatMapHeadLoss = nn.BCELoss()

    def forward(self, x):
        x = self.backbone(x)
        heatmap = self.countingHead(x)
        x = self.averagePool(x)
        x = x.view(64*154*84)
        x = self.fc32(x)
        behavCls = self.behavClsHead(x)
        densLevelCls = self.densLevClsHead(x)
        count = self.countingHead(x)
        return behavCls, densLevelCls, count, heatmap

    def initWeights(self):
        torch.nn.init.xavier_uniform(self.fc32.weight)
        self.behavClsHead.initWeights()
        self.densLevClsHead.initWeights()
        self.countingHead.initWeights()
        self.heatMapHead.initWeights()

    def calculateLoss(self, result, groundTruth):
        behavClsResult, densLevelClsResult, countResult, heatmapResult = result
        behavClsGt, densLevelClsGt, countGt, heatmapGt = groundTruth
        behavClsLoss = self.behavClsHeadLoss(behavClsResult, behavClsGt)
        densLevelClsLoss = self.densLevClsHeadLoss(densLevelClsResult, densLevelClsGt)
        countLoss = self.countingHeadLoss(countResult, countGt)
        heatmapLoss = self.heatMapHeadLoss(heatmapResult, heatmapGt)
        loss = behavClsLoss + densLevelClsLoss + countLoss + heatmapLoss
        return loss

def checkDensityClass(count):
    densityClass = 3
    if int(count) < 20 :
        densityClass = 1
    elif int(count) < 35 :
        densityClass = 2        
    return densityClass

class ResnetCrowdDataset(Dataset):
    def __init__(self):
        self.images = []
        self.heatMaps = []
        self.countValues = []
        self.densityClasses = []
        with open("./dataset/neutralTrain.txt") as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                imageName = line[0]
                self.countValues.append(int(line[1]))
                self.densityClasses.append(checkDensityClass(line[1]))
                imagePath = "./dataset/images/" + str(imageName) + ".png"
                image = Image.open(imagePath)
                image = TF.to_tensor(image)
                image.unsqueeze_(0)
                self.images.append(image)
                heatMapPath = "./dataset/heatmaps/" + str(imageName) + "_densityMap.jpg"
                heatMap = Image.open(heatMapPath)
                heatMap = TF.to_tensor(heatMap)
                heatMap.unsqueeze_(0)
                self.heatMaps.append(heatMap)
        with open("./dataset/panicTrain.txt") as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                imageName = line[0]
                self.countValues.append(int(line[1]))
                self.densityClasses.append(checkDensityClass(line[1]))
                imagePath = "./dataset/images/" + str(imageName) + ".png"
                image = Image.open(imagePath)
                image = TF.to_tensor(image)
                image.unsqueeze_(0)
                self.images.append(image)
                heatMapPath = "./dataset/heatmaps/" + str(imageName) + "_densityMap.jpg"
                heatMap = Image.open(heatMapPath)
                heatMap = TF.to_tensor(heatMap)
                heatMap.unsqueeze_(0)
                self.heatMaps.append(heatMap)

    def __getitem__(self, index):
        return self.images[index], self.heatMaps[index], self.countValues[index], self.densityClasses[index], 
    
    def __len__(self):
        len(self.images)

def train(model, device, train_loader, optimizer, epoch):
    model.initWeights()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.calculateLoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def main():
    dataset = ResnetCrowdDataset()
    firstData = dataset[0]
    print (firstData)

def main_1():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 32}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    model = ResnetCrowd().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=1.0, weight_decay=0.0001)
    epoch = 20
    train(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), "resnetCrowd.pth")

if __name__ == '__main__':
    main()

