# egclass.py
# Author: Alejandro Sanchez
# Created: Jun 25, 2018
#
# First iteration of classifying between photons and electrons using
# supercluster rechit images as input into a convolutional neural net

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils_data
from sklearn.metrics import roc_auc_score, roc_curve, auc

electron_input_file = "SingleElectronFlatPt10To160_2016_25ns_Moriond17MC_PoissonOOTPU_IMG_RH1_n225k.hdf5"
photon_input_file = "SinglePhotonFlatPt10To160_2016_25ns_Moriond17MC_PoissonOOTPU_IMG_RH1_n225k.hdf5"
validation_fraction = 0.15
use_gpu = True

print("Fetching Data...")

f_electron = h5py.File(electron_input_file, 'r')   # denoted as y = 0.0
ele_data = np.squeeze(f_electron['X_crop0'])
ele_data = torch.Tensor(ele_data)
ele_data = ele_data.view(-1, 1, 32, 32)
ele_targets = torch.zeros(len(ele_data))
num_ele_val = int(len(ele_data) * validation_fraction)

f_photon = h5py.File(photon_input_file, 'r')       # denoted as y = 1.0
pho_data = np.squeeze(f_photon['X_crop0'])
pho_data = torch.Tensor(pho_data)
pho_data = pho_data.view(-1, 1, 32, 32)
pho_targets = torch.ones(len(pho_data))
num_pho_val = int(len(pho_data) * validation_fraction)

training_data = utils_data.ConcatDataset((utils_data.TensorDataset(
                    ele_data[num_ele_val:], ele_targets[num_ele_val:]),
                utils_data.TensorDataset(
                    pho_data[num_pho_val:], pho_targets[num_pho_val:])))

validation_data = utils_data.ConcatDataset((utils_data.TensorDataset(
                      ele_data[:num_ele_val], ele_targets[:num_ele_val]),
                  utils_data.TensorDataset(
                      pho_data[:num_pho_val], pho_targets[:num_pho_val])))

dataloader = utils_data.DataLoader(training_data, batch_size=100, shuffle=True, num_workers=10)
testloader = utils_data.DataLoader(validation_data, batch_size=100, shuffle=True, num_workers=10)

print("Initializing Classifyer...")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( 1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        #x = x.view(-1, self.num_flat_features(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def accuracy(self, testloader):
        correct, total = 0.0, 0.0
        values = np.array([])
        answer = np.array([])

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
        
                value, predicted = torch.max(outputs.data, 1)
                values = np.concatenate((values, value.cpu().numpy()))
                answer = np.concatenate((answer, labels.cpu().numpy()))
                total += labels.size(0)
                correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum().item()

            acc = correct/total
            auc = roc_auc_score(answer,values)
        return acc,auc

net = Net()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2,
#                                                 verbose=True, min_lr=1e-6,
#                                                 threshold=1e-4, factor=0.2)
loss = 0.0

print("Begin Training")

device = torch.device("cuda:0" if use_gpu else "cpu")
net.to(device)

for epoch in range(20):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs[:,1], labels).to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    #scheduler.step(loss)
    
    net.eval()
    acc, auc = net.accuracy(testloader)
    
    print("[epoch %2d] loss: %.6f" %
                  (epoch + 1, running_loss / len(dataloader)))
    print("Acc: %.5f auc score: %.5f" % (acc, auc))
    running_loss = 0.0

print("Fininshed Training")

#correct = 0
#total = 0
#values = np.array([])
#answer = np.array([])
#with torch.no_grad():
#    for data in testloader:
#        inputs, labels = data
#        inputs, labels = inputs.to(device), labels.to(device)
#        outputs = net(inputs)
#  
#        value, predicted = torch.max(outputs.data, 1)
#        values = np.concatenate((values, value.cpu().numpy()))
#        answer = np.concatenate((answer, labels.cpu().numpy()))
#        #total += labels.size(0)
#        #correct += (predicted == labels).sum().item()
#
#fpr,tpr,_ = roc_curve(answer,values)
#roc_auc = auc(fpr,tpr)
#print("roc auc score: %.3f" % (roc_auc))

