# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 22:16:17 2021

@author: narut
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


stride = 1
kernel_size = 3
classes = ('plane', 'car', 'bird', 'cat', 'deer',
      'dog', 'frog', 'horse', 'ship', 'truck')
                                                            
class CIFAR10_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size, stride, 'same')
    self.conv2 = nn.Conv2d(32, 64, kernel_size, stride, 'same')
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 8 * 8, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    #print(x.shape)
    #x = self.conv1(x)
    #print(x.shape)
    #x = F.relu(x)
    #print(x.shape)
    #x = self.pool(x)
    #print(x.shape)
    x = self.pool(F.relu(self.conv2(x)))
    #x = self.conv2(x)
    #print(x.shape)
    #x = F.relu(x)
    #print(x.shape)
    #x = self.pool(x)
    #print(x.shape)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    #print(x.shape)
    x = F.relu(self.fc1(x))
    #print(x.shape)
    x = F.relu(self.fc2(x))
    #print(x.shape)
    x = self.dropout2(x)
    x = F.relu(self.fc3(x))
    output = F.log_softmax(x, dim=1)
    #print(output.shape)
    return output

class CIFAR10_CNN_mod(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size, stride, 'same')
    self.conv2 = nn.Conv2d(32, 64, kernel_size, stride, 'same')
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 8 * 8, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 10)

  def forward(self, x):
    #x = self.pool(F.relu(self.conv1(x)))
    #print(x.shape)
    x1 = self.conv1(x)
    #print(x.shape)
    x = F.relu(x1)
    #print(x.shape)
    x = self.pool(x)
    #print(x.shape)
    #x = self.pool(F.relu(self.conv2(x)))
    #print(x.shape)
    x2 = self.conv2(x)
    #print(x.shape)
    x = F.relu(x2)
    #print(x.shape)
    x = self.pool(x)
    x = torch.flatten(x, 1)
    #print(x.shape)
    x = F.relu(self.fc1(x))
    #print(x.shape)
    x = F.relu(self.fc2(x))
    #print(x.shape)
    x = F.relu(self.fc3(x))
    output = F.log_softmax(x, dim=1)
    #print(output.shape)
    return x1, x2, output


def train(net, train_loader, optimizer, criterion, epoch, device=None):
  for epoch_idx in range(epoch):
    net.train()   # Set the flag
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
    
      optimizer.zero_grad()   # zero the parameter gradients
      outputs = net(inputs)   # Forward
      loss = criterion(outputs, labels)
      loss.backward()   # Backprop
      optimizer.step()  # Update parameters

      if batch_idx % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
          epoch_idx, batch_idx * len(inputs), len(train_loader.dataset),      
          100. * batch_idx / len(train_loader), loss.item()))    

  print('Finished Training')  


def train_one_epoch(net, train_loader, optimizer, criterion, epoch_idx, device=None):
  net.train()
  running_loss = 0.0
  batch_cnt = 0
  for batch_idx, (inputs, labels) in enumerate(train_loader):
    if device != None:
      inputs, labels = inputs.to(device), labels.to(device)
  
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = net(inputs)   # Forward
    loss = criterion(outputs, labels)
    loss.backward()   # Backprop
    optimizer.step()  # Update parameters

    running_loss += loss.item()
    batch_cnt = batch_idx

    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
        epoch_idx, batch_idx * len(inputs), len(train_loader.dataset),      
        100. * batch_idx / len(train_loader), loss.item()))
      
  return (running_loss / batch_cnt)


def test(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  batch_cnt = 0
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      c = criterion(outputs, labels)
      print(c.item())
      test_loss += c.item()  # sum up batch loss
      pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(labels.view_as(pred)).sum().item()
      batch_cnt = batch_idx

  test_loss /= batch_cnt

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def test_one_epoch(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  batch_cnt = 0
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(labels.view_as(pred)).sum().item()
      batch_cnt = batch_idx
      

  test_loss /= batch_cnt

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  return (100. * correct / len(test_loader.dataset))


def test_return_incorrect(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  batch_cnt = 0
  incorrect_list = []
  
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(labels.view_as(pred)).sum().item()
      
      #print(pred.flatten())
      #print(labels)
      #incorrect_idx = ((torch.flatten(pred.eq(labels.view_as(pred)))==False).nonzero(as_tuple=False)).flatten()
      incorrect_idx = ((torch.flatten(pred.eq(labels.view_as(pred)))==True).nonzero(as_tuple=False)).flatten()
      incorrect_pred = torch.index_select(pred.flatten(), 0, incorrect_idx).tolist()
      incorrect_label = torch.index_select(labels.flatten(), 0, incorrect_idx).tolist()
      incorrect_idx = incorrect_idx.tolist()
      incorrect_list.extend([(i + (batch_idx * test_loader.batch_size), p, l) for i, p, l in zip(incorrect_idx, incorrect_pred, incorrect_label)])
      
      batch_cnt = batch_idx

  test_loss /= batch_cnt

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  return incorrect_list

def test_return_feature(net, test_loader, criterion, device=None):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  batch_cnt = 0
  feature_map1__list = []
  feature_map2__list = []
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      fm1, fm2, outputs = net(inputs)
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(labels.view_as(pred)).sum().item()
      batch_cnt = batch_idx
      
      for idx in range(inputs.shape[0]):
        feature_map1__list.append(fm1[idx, :, :, :])
        feature_map2__list.append(fm2[idx, :, :, :])
      

  test_loss /= batch_cnt

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  return feature_map1__list, feature_map2__list

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def main():

  # Load data
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
  )


  train_set = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
  test_set = datasets.CIFAR10('../data', train=False, download=True, transform=transform)
  train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
  train_loader_ev = DataLoader(train_set, batch_size=1000, shuffle=False)
  test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = CIFAR10_CNN().to(device)
  #net = MNIST_CNN_mod().to(device)
  #optimizer = optim.Adagrad(net.parameters(), lr = 0.001)

  optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
  # Regularization
  #optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-3)
  #optimizer = optim.SGD(net.parameters(), lr=0.003)
  criterion = nn.CrossEntropyLoss()
  epoch_num = 50
  #epoch_num = 30
  """
  train(net, train_loader, optimizer, criterion, epoch_num, device)
  test(net, test_loader, criterion, device)
  """
  
  # Training
  loss_list = []
  train_acc_list = []
  test_acc_list = []
  for epoch in range(1, epoch_num + 1):
    loss = train_one_epoch(net, train_loader, optimizer, criterion, epoch, device)
    loss_list.append(loss)

    train_acc = test_one_epoch(net, train_loader_ev, criterion, device)
    train_acc_list.append(train_acc)
    test_acc = test_one_epoch(net, test_loader, criterion, device)
    test_acc_list.append(test_acc)

  # Save parameters of the model
  torch.save(net.state_dict(), 'net_cifar_2.pt')
  """
  # Load the parameters of model
  net.load_state_dict(torch.load('net_cifar_1.pt'))
  """
  
  # Plot Accuracy
  
  print("=== Show accuracy plot ===>>")
  fig , ax = plt.subplots()
  #plt.rcParams["figure.figsize"] = (8, 3.5)
  plt.plot(range(len(train_acc_list)), train_acc_list, label = "training accuracy", color = "blue", linewidth = 0.5)
  plt.plot(range(len(test_acc_list)), test_acc_list, label = "testing accuracy", color = "orange", linewidth = 0.5)
  plt.title("Accuracy of the Model")
  plt.ylabel("Accuracy Rate(%)")
  plt.xlabel("Epoch")
  leg = ax.legend(loc='lower right') 
  plt.savefig('cifar_acc_2.png')
  plt.show()
  
  print(" ")
  
  # Plot learning curve
  print("=== Show learning plot ===>>")
  plt.plot(loss_list)
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.title("Learning Curve")
  plt.savefig('cifar_lc_2.png')
  plt.show()
  
  print(" ")
  
  # Plot weight distribution
  filter1 = net.conv1.weight.data
  filter1_data = torch.flatten(filter1).cpu().data.numpy()
  #print(filter1_data.shape)
  plt.hist(filter1_data)
  plt.title("Hist of Conv Layer1")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('conv1_w_2.png')
  plt.show()
  
  filter2 = net.conv2.weight.data
  filter2_data = torch.flatten(filter2).cpu().data.numpy()
  #print(filter2_data.shape)
  plt.hist(filter2_data)
  plt.title("Hist of Conv Layer2")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('conv2_w_2.png')
  plt.show()
  
  dense1 = net.fc1.weight.data
  dense_data = torch.flatten(dense1).cpu().data.numpy()
  #print(dense_data.shape)
  plt.hist(dense_data)
  plt.title("Hist of Dense Layer")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('dense1_w_2.png')
  plt.show()
  
  dense2 = net.fc2.weight.data
  dense_data = torch.flatten(dense2).cpu().data.numpy()
  #print(dense_data.shape)
  plt.hist(dense_data)
  plt.title("Hist of Dense Layer")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('dense2_w_2.png')
  plt.show()
  
  out = net.fc3.weight.data
  out_data = torch.flatten(out).cpu().data.numpy()
  #print(dense_data.shape)
  plt.hist(out_data)
  plt.title("Hist of Output Layer")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('out_w_2.png')
  plt.show()
  
  
  
  # Check miss-classified images

  incorrect_list = test_return_incorrect(net, test_loader, criterion, device)
  print(incorrect_list)
  for idx, pred, label in incorrect_list:
    
    if idx % 100 == 0:    
      plt.title("label: {}, pred: {}".format(classes[label], classes[pred]))
      plt.imshow(imshow(test_set[idx][0].data.squeeze()))
      #plt.savefig("miss-id: {}.png".format(idx))
      plt.savefig("hit-id: {}.png".format(idx))
      plt.show()

     
  
  
  
  # Extract feature map

  fm1, fm2 = test_return_feature(net, test_loader, criterion, device)
  lst = [7300, 9300]
  print(fm1[449][0].shape)
  for i in lst:
    for j in range(32):
      plt.imshow(fm1[i][j].data.squeeze().cpu().data.numpy(), cmap='gray')
      plt.savefig("fm1:{}-{}.png".format(i, j))
      plt.show()

    for k in range(64):
      plt.imshow(fm2[i][k].data.squeeze().cpu().data.numpy(), cmap='gray')
      plt.savefig("fm2:{}-{}.png".format(i, k))
      plt.show()

  
if __name__=="__main__":
  main()
