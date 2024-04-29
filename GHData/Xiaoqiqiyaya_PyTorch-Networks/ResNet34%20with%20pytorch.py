import torch
import torch.nn as nn
import torchvision
from  torchvision import transforms



class block(nn.Module):
    def __init__(self,input,output,downsample =None):
        super(block, self).__init__()
        self.layer1 = nn.Conv2d(input,output,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
        self.bn = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(inplace=True)

        self.layer2 = nn.Conv2d(output,output,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
        self.bn2 = nn.BatchNorm2d(output)
        self.downsample = downsample


    def  forward(self,x):
            reout = x
            out = self.layer1(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.layer2(out)
            out = self.bn2(out)
            if self.downsample is not None:
              reout = self.downsample(x)
            out+=reout
            out = self.relu(out)
            return out

class Resnet(nn.Module):
    def __init__(self,block):
        super(Resnet, self).__init__()
        self.input_c =64
        self.conv = nn.Conv2d(3,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.relu= nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.layer1 = self.Layer_add(64,64,block,3)
        self.layer2 = self.Layer_add(64,128,block,4)
        self.layer3 = self.Layer_add(128,256,block,6)
        self.layer4 = self.Layer_add(256,512,block,3)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, 10)


    def forward(self,x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


    def Layer_add(self,input_c,output_c,block,num,stride=1):
        downsample = nn.Sequential(
                nn.Conv2d(input_c,output_c,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(output_c)
        )
        layer = []
        layer.append(block(input_c,output_c,downsample))
        for i in range(1,num):
            layer.append(block(output_c,output_c))
        return nn.Sequential(*layer)

if  __name__ == '__main__':
    bathc_size = 100
    epoch_num = 1
    learning_rate = 0.1
    num_class = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),  # 防止过拟合，随机反转
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.45, 0.45, 0.40), (0.2, 0.2, 0.2))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.49, 0.4, 0.4), (0.23, 0.2, 0.2))
    ])

    train_data = torchvision.datasets.CIFAR10(root='../../data',
                                              train=True,
                                              download=True,
                                              transform=train_transform)

    test_data = torchvision.datasets.CIFAR10(root='../../data', train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bathc_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bathc_size)


    model = Resnet(block).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

    total_step = len(train_loader)
    for epoch in range(epoch_num):
        for i ,(image,label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epoch_num, i + 1, total_step, loss.item()))
    length = len(test_loader)
    with torch.no_grad():
        correct = 0
        total = 0
        for (image,label) in test_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _,prediction = torch.max(output,1)
            total+=label.size(0)
            correct +=(prediction==label).sum().item()
            print('Test Accuracy of the model on the {} test images: {} %'.format(length, 100 * correct / total))```
