import torchvision
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size= 32, shuffle = False)

device = torch.device("cuda")
print(device)

#always look at your datasets
print(trainset)
print(testset)

# batch1 = iter(testloader)
# imgs, labels = batch1.__next__()
# print(labels)
# print(imgs.size()) # just making sure it looks ok

class fashNet(nn.Module):

    def __init__(self):
        super(vgg16, self).__init__()

        ## note that vgg always does same padding on convolutions
        ## dec img size by pooling and inc channels using kernels
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding = 1),
            # nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 3, padding = 1),
            # nn.BatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding = 1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # out = 14x14 img
        )

        self.fc_block = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            # nn.Softmax(dim = 1) 
        )

    def forward(self, x):
        x = self.cnn_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:] # exclude batch dimension
    #     num_features = 1
    #     for i in size:
    #         num_features *= i
    #     return num_features

net = fashNet().to(device)
loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(net.parameters(), lr = 0.0001)

# # output looks ok
# dat = iter(testloader).__next__()[0].to(device)
# print(dat.size())
# out = net(dat)
# print(out)
# dat = iter(testloader).__next__()[0].to(device)
# print(dat.size())
# out = net(dat)
# print(out)
# print("hehehehehehhehehehehe")
# print(out.data)
# _, preds = torch.max(out.data, 1)
# print("heheheheheh22222")
# print(preds)

# calculate accuracy using GPU
def evalaluate(dataLoader):
    total = 0
    correct = 0
    net.eval() # put model in eval mode

    for data in dataLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predictions = torch.max(outputs.data, 1)
        total += labels.size()
        correct += (predictions == labels).sum().item()
    return (correct/total) * 100


## training loop
loss_arr = []
epochs = 10
c = 0
for epoch in range(epochs):
    print("----------------EPOCH ", c, " ---------------------")
    c += 1
    for data in trainloader:
        # print(c)
        # c+= 1
        net.train() # put net in train mode
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        opt.step()
    loss_arr.append(loss.item())




