#!/usr/bin/env python3

# Based on the code at https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

# The code is very similar to the sample code given, creating a for loop at line 123. 

import torch

print(f'Pytorch {torch.__version__} is installed')

import numpy as np

import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

import os.path

#
# Read the data from mnist
#
def read_data(data_dir):
   transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
                              
                              
   train_set = datasets.MNIST(data_dir, download=True, train=True, transform=transform)
   test_set = datasets.MNIST('test/MNIST/processed/', download=True, train=False, transform=transform)
   
   train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)
   test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
   
   return train_loader, test_loader
   
def make_network(input_size, hidden_sizes, output_size):
   neural_network = nn.Sequential(
                      nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1),
                      )
   return neural_network

#
# Train the Neural Network
#
def train_neural_network(epoch_input, neural_network, train_loader, learning_rate):
   optimizer = optim.SGD(neural_network.parameters(), lr = learning_rate)
   time0 = time()
   loss_function = nn.NLLLoss()

   epochs = epoch_input
   for e in range(epochs):
       running_loss = 0
       for images, labels in train_loader:
           # Flatten MNIST images into a 784 long vector
           images = images.view(images.shape[0], -1)
       
           # Training pass
           optimizer.zero_grad()
           
           output = neural_network(images)
           loss = loss_function(output, labels)
           
           #This is where the model learns by backpropagating
           loss.backward()
           
           #And update the weights using the derivatives (or gradient)
           optimizer.step()
           
           running_loss += loss.item()
           
       print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_loader)))
           
   print("\nTraining Time (in minutes) =", (time()-time0)/60)

def test(neural_network, test_loader):
   correct_count, all_count = 0, 0
   for images,labels in test_loader:
     for i in range(len(labels)):
       img = images[i].view(1, 784)
       with torch.no_grad():
           logps = neural_network(img)
       ps = torch.exp(logps)
       probab = list(ps.numpy()[0])
       pred_label = probab.index(max(probab))
       true_label = labels.numpy()[i]
       if(true_label == pred_label):
          correct_count += 1
       elif False:
          print("is " + str(true_label) + ", thought " + str(pred_label))
          plt.imshow(images[i].numpy().squeeze(), cmap='gray_r');
          plt.show()
       all_count += 1

   print("Number Of Images Tested =", all_count)
   print("\nModel Accuracy =", (correct_count/all_count), )

def main():
   # For testing, best to ensure that we are deterministic
   torch.manual_seed(0)
   
   # Read in the data (training and testing
   train_loader, test_loader = read_data('test/MNIST/processed/')

   # make the network
   input_size = 784
   hidden_sizes = [128, 64]
   output_size = 10

   neural_network = make_network(input_size, hidden_sizes, output_size)

   print(neural_network)
   
   learning_rate = 0.1

   epoch_values = [5, 15, 30]
   for i in range(len(epoch_values)):

      if os.path.exists('models/' + 'nn128_64_' + str(learning_rate) + '.nn'):
         neural_network.load_state_dict(torch.load('models/' + 'nn128_64_' + str(learning_rate) + '.nn'))
      else:
         train_neural_network(epoch_values[i], neural_network, train_loader, learning_rate)
   
   epoch_values = [5, 15, 30]
   for i in range(len(epoch_values)):
      test(neural_network, test_loader)

main()
