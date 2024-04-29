# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:58:43 2018

@author: Neeti
"""

# Install the library using conda install packageName in the Anaconda Prompt
#for eg conda install plotly
#conda install numpy
#conda install pandas
#conda install matplotlib
#conda install torch

#Importing the required Library
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
import plotly.plotly as py
import plotly
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from plotly.graph_objs import *

#Made class to reuse code for reading and reshaping data 
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)
        self.Y = np.array(data.iloc[:, 0])
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        
        if self.transform:
            item = self.transform(item)
        
        return (item, label)


train_dataset = FashionMNISTDataset(csv_file ='fashion-mnist_train1.csv')
test_dataset = FashionMNISTDataset(csv_file ='fashion-mnist_test.csv')
data = pd.read_csv('fashion-mnist_train1.csv');

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


num_epochs = 3
learning_rate = 0.001
#instance of the Conv Net
cnn = CNN();
#loss function and optimizer
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);


losses = [];
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.data[0])
        
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))

losses_in_epochs = losses[0::600]
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.plot(losses_in_epochs);
plt.show();

print(losses_in_epochs)

py.sign_in('neeti.jain', 'VMLBAbaEpnj2CQyryrVS')
plotly.tools.set_credentials_file(username='neeti.jain', api_key='VMLBAbaEpnj2CQyryrVS')



trace1 = {
  "x": ["3", "23", "24", "18", "22", "1", "1", "12", "2", "6", "2", "12", "23", "2", "3", "13", "11", "15", "20", "2", "4", "10", "0", "19", "5", "7", "0", "11", "6", "21", "18", "19", "21", "7", "0", "0", "4", "14", "15", "10", "5", "19", "2", "13", "7", "16", "12", "8", "15", "11", "21", "14", "10", "6", "18", "11", "16", "14", "16", "18", "4", "2", "16", "14", "23", "0", "21", "16", "10", "14", "17", "1", "15", "18", "4", "16", "12", "19", "18", "12", "18", "10", "1", "5", "21", "18", "10", "10", "22", "4", "19", "11", "12", "1", "10", "5", "20", "0", "20", "11", "23", "14", "0", "23", "10", "7", "24", "5", "15", "20", "20", "19", "6", "20", "19", "3", "24", "4", "12", "10", "8", "8", "6", "21", "13", "2", "20", "1", "19", "14", "19", "18", "11", "17", "20", "21", "3", "4", "23", "7", "24", "16", "21", "20", "23", "8", "4", "22", "7", "1", "19", "15", "15", "4", "14", "14", "23", "2", "11", "18", "1", "8", "3", "5", "11", "14", "18", "14", "5", "5", "22", "10", "2", "13", "3", "16", "22", "18", "6", "16", "19", "22", "19", "6", "21", "20", "13", "17", "10", "24", "23", "15", "18", "15", "3", "3", "0", "0", "1", "7", "21", "19", "19", "2", "10", "20", "19", "14", "8", "11", "14", "17", "23", "17", "11", "24", "15", "18", "18", "3", "13", "0", "2", "22", "18", "15", "6", "1", "10", "17", "1", "15", "18", "23", "23", "13", "0", "11", "6", "3", "16", "20", "18", "0", "6", "1", "24", "16", "4", "19", "14", "1", "17", "20", "2", "7", "14", "3", "2", "21", "0", "22", "18", "0", "10", "24", "16", "13", "0", "14", "20", "6", "16"], 
  "y": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246", "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262", "263", "264", "265", "266", "267", "268", "269", "270", "271", "272", "273"], 
  "mode": "markers", 
  "type": "scatter", 
  "xsrc": "neeti.jain:0:593987", 
  "ysrc": "neeti.jain:0:9cc9d9"
}
data = Data([trace1])
layout = {
  "autosize": True, 
  "margin": {"t": 92}, 
  "plot_bgcolor": "rgb(72, 240, 204)", 
  "showlegend": False, 
  "title": "<b>Scatter Plot for test data wrt labels</b>", 
  "xaxis": {
    "automargin": False, 
    "autorange": False, 
    "gridwidth": 3, 
    "range": [-1.36027892562, 24.3602789256], 
    "rangeslider": {
      "autorange": True, 
      "bgcolor": "rgb(146, 242, 158)", 
      "borderwidth": 1, 
      "range": [-1.36027892562, 24.3602789256], 
      "thickness": 0.2, 
      "visible": True, 
      "yaxis": {"rangemode": "match"}
    }, 
    "showspikes": False, 
    "side": "bottom", 
    "tickfont": {"size": 14}, 
    "ticklen": 5, 
    "ticks": "inside", 
    "tickwidth": 4, 
    "title": "Labels&nbsp;", 
    "titlefont": {"size": 19}, 
    "type": "category"
  }, 
  "yaxis": {
    "automargin": False, 
    "autorange": True, 
    "range": [-17.6417445483, 291.641744548], 
    "side": "left", 
    "ticklen": 5, 
    "ticks": "inside", 
    "tickwidth": 4, 
    "title": "Iteration", 
    "titlefont": {"size": 19}, 
    "type": "linear"
  }
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)

