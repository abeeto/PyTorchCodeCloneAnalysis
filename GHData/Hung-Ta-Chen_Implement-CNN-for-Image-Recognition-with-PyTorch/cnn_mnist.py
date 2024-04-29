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


stride = 1
kernel_size = 3

class MNIST_CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size, stride)
    self.conv2 = nn.Conv2d(16, 32, kernel_size, stride)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(32 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 10)

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
    #print(x.shape)
    x = torch.flatten(x, 1)
    #print(x.shape)
    x = F.relu(self.fc1(x))
    #print(x.shape)
    x = F.relu(self.fc2(x))
    #print(x.shape)
    output = F.log_softmax(x, dim=1)
    #print(output.shape)
    return output

class MNIST_CNN_mod(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size, stride)
    self.conv2 = nn.Conv2d(16, 32, kernel_size, stride)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(32 * 5 * 5, 128)
    self.fc2 = nn.Linear(128, 10)

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
      incorrect_idx = ((torch.flatten(pred.eq(labels.view_as(pred)))==False).nonzero(as_tuple=False)).flatten()
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


def main():

  # Load data
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
  )
  train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
  test_set = datasets.MNIST('../data', train=False, download=True, transform=transform)
  train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
  train_loader_ev = DataLoader(train_set, batch_size=1000, shuffle=False)
  test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net = MNIST_CNN().to(device)
  #net = MNIST_CNN_mod().to(device)
  #optimizer = optim.Adagrad(net.parameters(), lr = 0.001)

  optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
  # Regularization
  #optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-2)
  #optimizer = optim.SGD(net.parameters(), lr=0.003)
  criterion = nn.CrossEntropyLoss()
  epoch_num = 35
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
  torch.save(net.state_dict(), 'net.pt')
  """
  # Load the parameters of model
  net.load_state_dict(torch.load('net.pt'))
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
  plt.savefig('acc.png')
  plt.show()
  
  print(" ")
  
  # Plot learning curve
  print("=== Show learning plot ===>>")
  plt.plot(loss_list)
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.title("Learning Curve")
  plt.savefig('lc.png')
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
  plt.savefig('w1.png')
  plt.show()
  
  filter2 = net.conv2.weight.data
  filter2_data = torch.flatten(filter2).cpu().data.numpy()
  #print(filter2_data.shape)
  plt.hist(filter2_data)
  plt.title("Hist of Conv Layer2")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('w2.png')
  plt.show()
  
  dense = net.fc1.weight.data
  dense_data = torch.flatten(dense).cpu().data.numpy()
  #print(dense_data.shape)
  plt.hist(dense_data)
  plt.title("Hist of Dense Layer")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('w3.png')
  plt.show()
  
  out = net.fc2.weight.data
  out_data = torch.flatten(out).cpu().data.numpy()
  #print(dense_data.shape)
  plt.hist(out_data)
  plt.title("Hist of Output Layer")
  plt.ylabel("Number")
  plt.xlabel("Value")
  plt.savefig('w4.png')
  plt.show()
  
  
  
  # Check miss-classified images
  incorrect_list = test_return_incorrect(net, test_loader, criterion, device)
  print(incorrect_list)
  for idx, pred, label in incorrect_list:
    plt.title("label: {}, pred: {}".format(label, pred))
    plt.imshow(test_set[idx][0].data.squeeze(), cmap='gray')
    plt.savefig("miss-id: {}.png".format(idx))
    plt.show()

  
  
  # Extract feature map
  fm1, fm2 = test_return_feature(net, test_loader, criterion, device)
  lst = [449, 9400, 3767, 3600, 9792, 3200]
  print(fm1[449][0].shape)
  for i in lst:
    for j in range(16):
      plt.imshow(fm1[i][j].data.squeeze().cpu().data.numpy(), cmap='gray')
      plt.savefig("fm1:{}-{}.png".format(i, j))
      plt.show()

    for k in range(32):
      plt.imshow(fm2[i][k].data.squeeze().cpu().data.numpy(), cmap='gray')
      plt.savefig("fm2:{}-{}.png".format(i, k))
      plt.show()

  
if __name__=="__main__":
  main()
