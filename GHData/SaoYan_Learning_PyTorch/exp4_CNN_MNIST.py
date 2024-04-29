import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

class Net(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=7*7*64, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., 0.01)
                nn.init.constant_(m.bias, 0.)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    # dataset
    def _worker_init_fn_():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32-1
        random.seed(torch_seed)
        np.random.seed(np_seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(root='MNIST', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, 
        shuffle=True, num_workers=4, worker_init_fn=_worker_init_fn_())
    testset = datasets.MNIST(root='MNIST', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)

    # define the network
    net = Net(in_features=1, num_classes=10).to(device)

    # define loss
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    # training
    epochs = 10
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        correct = 0.
        total = 0.
        for i, data in enumerate(trainloader, 0):
            # data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # prepare
            net.train()
            net.zero_grad()
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # check accuracy on test set
            if i % 100 == 0:
                net.eval()
                with torch.no_grad():
                    for __, data_test in enumerate(testloader, 0):
                        images_test, labels_test = data_test
                        images_test, labels_test = images_test.to(device), labels_test.to(device)
                        outputs_test = net(images_test)
                        predict = torch.argmax(outputs_test, 1)
                        total += labels_test.size(0)
                        correct += torch.eq(predict, labels_test).sum().double().item()
                print('[eppch %d/%d][iter %d/%d] loss: %.4f test accuracy: %.2f%%'
                    % (epoch+1, epochs, i+1, len(trainloader), loss.item(), 100*correct/total))
                correct = 0.
                total = 0.
