import torch
import torchvision

from torch.autograd import Variable
import torch.optim as optim


# Hyperparameters
batch_size = 128
learning_rate = 0.0001
load_pretrained_model = False
epochs = 100
pretrained_epoch = 10 

# CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Preparing the training set and testing set
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)



def save_model(net, epoch):
    PATH = "./DwoG_pretrained_models/DwoG_model_epoch" + str(epoch+1) + ".model"
    torch.save(net.state_dict(),PATH)

def load_model(net, pretrained_epoch):
    PATH = "./DwoG_pretrained_models/DwoG_model_epoch" + str(pretrained_epoch) + ".model"
    net.load_state_dict(torch.load(PATH))
    net.eval()


# define the test accuracy function
def test_accuracy(net, testset_loader, epoch):
    # Test the model
    net.eval()
    correct = 0
    total = 0

    for data in testset_loader:
        images, labels = data
        images, labels = Variable(images).to(device), labels.to(device)
        _, output = net(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy Test -- Epoch '+str(epoch+1)+': ' + str(100 * correct / total))


import math
import torch.nn as nn
import torch.nn.functional as F
import copy

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=2)
        self.conv3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=2)
        self.conv5 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv7 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv8 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=2)
        self.conv1_ln = nn.LayerNorm([196,32,32])
        self.conv2_ln = nn.LayerNorm([196,16,16])
        self.conv3_ln = nn.LayerNorm([196,16,16])
        self.conv4_ln = nn.LayerNorm([196,8,8])
        self.conv5_ln = nn.LayerNorm([196,8,8])
        self.conv6_ln = nn.LayerNorm([196,8,8])
        self.conv7_ln = nn.LayerNorm([196,8,8])
        self.conv8_ln = nn.LayerNorm([196,4,4])
        self.pool = nn.MaxPool2d(4,4)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(in_features=196, out_features=1)
        self.fc10 = nn.Linear(in_features=196, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_ln(self.conv1(x)))
        x = F.leaky_relu(self.conv2_ln(self.conv2(x)))
        x = F.leaky_relu(self.conv3_ln(self.conv3(x)))
        x = F.leaky_relu(self.conv4_ln(self.conv4(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv5_ln(self.conv5(x)))
        x = F.leaky_relu(self.conv6_ln(self.conv6(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv7_ln(self.conv7(x)))
        x = F.leaky_relu(self.conv8_ln(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 196) # reshape x
        out1 = self.fc1(x)
        out2 = self.fc10(x)
        return out1, out2


import torch.optim as optim

model =  Discriminator()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_epoch = 0
if load_pretrained_model == True:
    load_model(model, pretrained_epoch)
    start_epoch = pretrained_epoch

model.to(device)


print('Start Training!')
for epoch in range(start_epoch,epochs):  # loop over the dataset multiple times
    if(epoch+1==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch+1==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0

    for group in optimizer.param_groups: 
        for p in group['params']:
            state = optimizer.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000

    running_loss = 0.0
    tmp_loss = 0.0
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
        X_train_batch = Variable(X_train_batch).to(device)
        Y_train_batch = Variable(Y_train_batch).to(device)
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data.item()
        tmp_loss += loss.data.item()
        if batch_idx % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, tmp_loss / 1000))
            tmp_loss = 0.0
            
    # print the loss after every epoch
    print('Epoch ' + str(epoch + 1) + ': loss = ' + str(running_loss / 50000))    
    if (epoch + 1)%2 == 0:
        # Test for accuracy after every 5 epochs
        test_accuracy(model, testloader, epoch)
        # Save model after every 5 epochs
        if (epoch + 1)%5 == 0:
            save_model(model, epoch)
    elif epoch == epochs - 1:
        test_accuracy(model, testloader, epoch)
        save_model(model, epoch)

print('Training finished and final models saved!')