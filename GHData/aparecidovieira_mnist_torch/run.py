from __future__ import print_function
import argparse
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data

batch_size = 128
BATCH_SIZE = 10
epochs = 10
momentum = 0.9
torch.manual_seed(1)
lr = 0.001
LR = 0.001


filter1_size = 32
filter2_size = 64
hide1_num = int(filter2_size*((28-2*2)/2)**2) #9216
hide2_num =128
DOWNLOAD_MNIST = True
device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, filter1_size, kernel_size=3), nn.BatchNorm2d(filter1_size))
        self.conv2 = nn.Sequential(nn.Conv2d(filter1_size, filter2_size, kernel_size=3), nn.BatchNorm2d(filter2_size))
        self.conv2d_drop = nn.Dropout(0.25)
        self.layer3 = nn.Linear(hide1_num, hide2_num)
        self.layer3_drop = nn.Dropout2d(0.5)
        self.layer4 = nn.Linear(hide2_num, 10)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2d_drop(self.conv2(x)), 2))
        x = x.view(-1, hide1_num)
        x = F.relu(self.layer3_drop(self.layer3(x)))
        x = self.layer4(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_, optimizer, epoch):
    model.train()
    ave_loss = 0
    for batch_idx, (data, target) in enumerate(train_):
        data, target = Variable(data), Variable(target)
        
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        #ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            #test_out, last_layer = model(test_x)
            #pred_y = torch.max(test_out, 1)[1].data.squeeze()
            #accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Train Epoch: {}[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_.dataset),
                100.*batch_idx/len(train_), loss.data[0])
            )
            #print('accuracy {}'.format(accuracy))

def test(model, device, test_, optimizer, epoch):
    model.eval()
    correct, ave_loss = 0, 0
    test_loss = 0
    total_cnt = 0
    for batch_idx, (data, target) in enumerate(test_):
        loss = F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        #total_cnt += data.data.size()[0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
        test_loss /= len(test_.dataset)
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_):
             print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

use_cuda = not True and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
"""
train_ = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,),(0.3081))
                                                ])),
                                                batch_size=batch_size, shuffle=True,**kwargs)

test_ = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                train=False,
                                                download=True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,),(0.3081))
                                                ])),
                                                batch_size=batch_size, shuffle=True, **kwargs)

"""
root = './data'
if not os.path.exists(root):
    os.mkdir(root)

"""
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_ = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_ = dset.MNIST(root=root, train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_,
                 batch_size=batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_,
                batch_size=batch_size,
                shuffle=False)
#print 'train data : {}'.format(len(train_loader))
#print('test data : {}'.format(len(test_loader))


train_data = torchvision.datasets.MNIST(
        root='./data/', 
        train=True,
        transform=torchvision.transforms.ToTensor,
        download=DOWNLOAD_MNIST
        )
print(train_data.train_data.size())
print(train_data.train_labels.size())



train_loader = torch.utils.data.DataLoader(
        dataset=train_data, 
        batch_size=BATCH_SIZE,
        shuffle=True
        )
""" 
  
kwargs = {'num_workers': 1, 'pin_memory': True} if 0 else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)


#test_data = torchvision.datasets.MNIST(root='./data/', train=False)
#test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255
#test_y = test_data.test_labels[:2000]


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 5 + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader, optimizer, epoch)
    
   
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        optimizer.zero_grad()
        output = model(data)