import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

class Lenet(nn.Module):
    def __init__(self,num_class):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,kernel_size=(5,5))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 =nn.Conv2d(6,16,kernel_size=(5,5))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_class)

    def forward(self,x):
        out = self.relu(self.conv1(x))
        out = self.maxpool(out)
        out = self.relu2(self.conv2(out))
        out = self.maxpool2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out




if __name__ =="__main__":
    bathc_size = 100
    epoch_num = 80
    learning_rate = 0.001
    num_class = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),  # 防止过拟合，随机反转
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
    ])

    train_data = torchvision.datasets.CIFAR10(root='../../data',
                                              train=True,
                                              download=True,
                                              transform=train_transform)

    test_data = torchvision.datasets.CIFAR10(root='../../data', train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bathc_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bathc_size)

    model = Lenet(num_class).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(model.parameters(),lr = learning_rate)

    total_step = len(train_loader)
    for epoch in range(epoch_num):
        for i,(image,label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss  =  criterion(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
               print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                     .format(epoch + 1, epoch_num, i + 1, total_step, loss.item()))
    length = len(train_loader)
    with torch.no_grad():
        correct = 0
        total = 0
        for  (image,label) in train_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _,prediction = torch.max(output,1)
            total+=label.size(0)
            correct +=(prediction==label).sum().item()
        print('Test Accuracy of the model on the {} test images: {} %'.format(length, 100 * correct / total))
