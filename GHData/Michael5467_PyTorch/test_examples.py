#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(args, net, device, train_data_set, optimizer, criterion, epoch,   train_losses, train_counter):
    net.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_data_set):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        y_pred = net(x_batch.view(-1, 28*28))
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(x_batch), 
                len(train_data_set.dataset),
                100. * batch_idx / len(train_data_set), 
                loss.item())
            )
            train_losses.append(loss.item())
            train_counter.append( (batch_idx * DEF_BATCH_SIZE) + ((epoch)*len(train_data_set.dataset)) )
            # torch.save(net.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test(args, net, device, test_data_set, criterion,   test_losses):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in test_data_set:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = net(x_batch.view(-1, 28*28))
            test_loss += criterion(y_pred, y_batch, size_average=False).item()
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(y_batch.data.view_as(pred)).sum()
            test_loss /= len(test_data_set.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, 
                correct, 
                len(test_data_set.dataset),
                100. * correct / len(test_data_set.dataset))
            )

DEF_BATCH_SIZE = 10
LEARNING_RATE = 1e-3
MOMENTUM = 0.5
EPOCHS = 3
LOG_INTERVAL = 10

# # fig, axs = plt.subplots(2, 2)
# # for i in range(2):
# #     for j in range(2):
# #         Z = np.random.rand(6, 10)
# #         axs[i, j].pcolor(Z)
# #         axs[i, j].set_title('%d:%d' %(i, j))
# # plt.show()

# train = datasets.MNIST("D:\\PyTorch\\", train=True,  transform = transforms.Compose([transforms.ToTensor()]), download=True)
# test  = datasets.MNIST("D:\\PyTorch\\", train=False, transform = transforms.Compose([transforms.ToTensor()]), download=True)

# # train_data_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
# # test_data_set  = torch.utils.data.DataLoader(test,  batch_size=10, shuffle=False)
# train_data_set = DataLoader(train, batch_size=DEF_BATCH_SIZE, shuffle=True)
# test_data_set  = DataLoader(test,  batch_size=DEF_BATCH_SIZE, shuffle=False)

# 1.
# for data_ndx, data in enumerate(train_data_set):
#     # print (data)
#     print ("data")    
#     break
#     # if (data_ndx > 0):
#         # break

# 2.
# print(len(train_data_set))
# dataiter = iter(train_data_set)
# for i in range(4):
#     print("\ni=", i)
#     print(dataiter.next())

# 3.
# dataiter = iter(train_data_set)
# data = dataiter.next()
# # print(data)
# # print(data[0][0].shape)
# # print(data[1][0])
# plt.imshow(data[0][0].view(28,28))
# plt.show()


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# fig, axs = plt.subplots(2, 2)
# for i in range(2):
#     for j in range(2):
#         Z = np.random.rand(6, 10)
#         axs[i, j].pcolor(Z)
#         axs[i, j].set_title('%d:%d' %(i, j))
# plt.show()

train_loader = datasets.MNIST('D:\\PyTorch\\', train=True,  transform = transforms.Compose([transforms.ToTensor()]), download=True)
test_loader  = datasets.MNIST('D:\\PyTorch\\', train=False, transform = transforms.Compose([transforms.ToTensor()]), download=True)

train_data_set = DataLoader(train_loader, batch_size=DEF_BATCH_SIZE, shuffle=True)
test_data_set  = DataLoader(test_loader,  batch_size=DEF_BATCH_SIZE, shuffle=False)

net = Model().to(device)
print(net)

# x = torch.rand((28, 28))
# x = x.view(-1, 28*28)
# output = net(x)
# print (output)

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

criterion = F.nll_loss
# criterion = F.mse_loss

train_losses  = []
train_counter = []
test_losses   = []
test_counter  = [i*len(train_data_set.dataset) for i in range(EPOCHS + 1)]

for batch_idx, (x_batch, y_batch) in enumerate(train_data_set):
    print(batch_idx) 
    # print(x_batch) 
    # print(y_batch) 
    print(x_batch.size())
    print(y_batch.size())

    x_batch_linear = x_batch.view(-1, 28*28)
    print(x_batch_linear) 
    print(x_batch_linear.size())

    break


# test("", net, device, test_data_set, criterion, test_losses)
# for epoch in range(EPOCHS):
#     train('', net, device, train_data_set, optimizer, criterion, epoch, train_losses, train_counter)
#     test ('', net, device, test_data_set,  criterion, test_losses)

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()

# ones_arr = torch.ones(28, 28)
# ones_dot_five = nn.init.constant_(torch.ones(28, 28), 0.5)
# # print (ones_dot_five)

# for epoch in range(EPOCHS):
#     print("epoch %d" %(epoch))
#     for x_batch, y_batch in train_data_set:
#         # y_batch = y_batch.float()
#         # x, y = data
#         # net.zero_grad()
#         y_pred = net(x_batch.view(-1, 28*28))

#         # # Debug print:
#         # fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
#         # for i in range(2):
#         #     for j in range(5):
#         #         # print(x_batch[2*i+j])
#         #         axs[i, j].imshow(ones_arr - x_batch[2*i+j][0], cmap='gray')
#         #         # axs[i, j].imshow(ones_arr - x_batch[2*i+j][0])
#         #         axs[i, j].set_title('%d' %(y_batch[2*i+j]))
#         # # print(ones_dot_five - x_batch[0][0])
#         # plt.show()
#         # break

#         # loss = F.nll_loss(y_pred, y_batch)
#         # print(x_batch.dtype)
#         # print(y_batch.dtype)
#         # print(y_batch_1.dtype)
#         # print(y_pred.dtype)
#         # # print("loss:")
#         # print(loss.dtype)

#         # break

#         print(x_batch.size())
#         print(y_batch.size())
#         print(y_pred.size())
#         print(y_pred)

#         break

#         # loss = criterion(y_pred.to(torch.float), y_batch.to(torch.float))
#         loss = criterion(y_pred, y_batch.float())

#         # # net.zero_grad()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # break
#     # print("loss:", loss)




# # # create classifier and optimizer objects
# # clf = CNNClassifier()
# # opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)

# # loss_history = []
# # acc_history = []

# # def train(epoch):
# #     clf.train() # set model in training mode (need this because of dropout)
    
# #     # dataset API gives us pythonic batching 
# #     for batch_id, (data, label) in enumerate(train_loader):
# #         data = Variable(data)
# #         target = Variable(label)
        
# #         # forward pass, calculate loss and backprop!
# #         opt.zero_grad()
# #         preds = clf(data)
# #         loss = F.nll_loss(preds, target)
# #         loss.backward()
# #         loss_history.append(loss.data[0])
# #         opt.step()
        
# #         if batch_id % 100 == 0:
# #             print(loss.data[0])
