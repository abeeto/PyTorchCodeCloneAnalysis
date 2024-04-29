#All imports
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler

#Particularity of TORCH
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

#Loading my Local Data
alldata = np.loadtxt('./datasetonlyvalues.csv', delimiter=",")

#Dictionaries for better compreension
mydata ={}
mydata['group0'] = alldata[0:20]
mydata['group1'] = alldata[20:40]
mydata['group2'] = alldata[40:60]
mydata['group3'] = alldata[60:80]
mydata['group4'] = alldata[80:100]

mylabel = len(mydata['group0']) * [0]
mylabel.extend(len(mydata['group1']) * [1])
mylabel.extend(len(mydata['group2']) * [2])
mylabel.extend(len(mydata['group3']) * [3])
mylabel.extend(len(mydata['group4']) * [4])

#My Dataset Class Creation
class MyData(Dataset):
    def __init__(self, alldata, mylabel):
        self.alldata = alldata
        self.mylabel  = mylabel

    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, idx):
        return self.alldata[idx], self.mylabel[idx]


data = MyData(alldata, mylabel)

train_size = int(0.75*len(data))
idx  = torch.randperm(len(data))
train_sampler = SubsetRandomSampler(idx[0:train_size]) 
test_sampler = SubsetRandomSampler(idx[train_size:])

train_loader = DataLoader(data, sampler=train_sampler,
                          batch_size=100, num_workers=0)

test_loader  = DataLoader(data, sampler=test_sampler,
                          batch_size=100, num_workers=0)

#Neural Network Class
class MyNN(nn.Module):

    def __init__(self, tam_entrada):

        super(MyNN, self).__init__()

        # Definir a arquitetura
        self.rede = nn.Sequential(
            nn.Linear(tam_entrada, 100),
            nn.ReLU(),
            nn.Linear(100, 5)
        )

    def forward(self, dado):

        # Fluxo de passagem do dado
        saida = self.rede(dado)
        return saida

tam_entrada = 78
rede = MyNN(tam_entrada).to(device).double()

print(rede)

#Some Optmizations
optimizer = optim.Adam(rede.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss().to(device)

#Training Flow
from sklearn import metrics 

def forward(mode, loader):

  if mode == 'train':
    rede.train()
  else:
    rede.eval()
  
  epoch_loss = []
  pred_list, label_list = [], []

  for data, label in loader:

    data = data.to(device)
    label = label.to(device)

    optimizer.zero_grad()
    out = rede(data)
    loss = criterion(out, label)
    epoch_loss.append(loss.cpu().data)
  
    pred = out.data.max(dim=1)[1].cpu().numpy()
    pred_list.append(pred)
    label_list.append(label.cpu().numpy())

    if mode == 'train':
      loss.backward()
      optimizer.step()

  epoch_loss = np.asarray(epoch_loss)
  pred_list = np.asarray(pred_list).ravel()
  label_list = np.asarray(label_list).ravel()
  acc = metrics.accuracy_score(pred_list, label_list)

  print(mode, 'Loss:', epoch_loss.mean(), '+/-', epoch_loss.std(), 'Accuracy:', acc)

#Training and Testing together
num_epochs = 5000
for i in range(num_epochs):
    forward('train', train_loader)
    forward('test', test_loader)
    print('--------------------------------')

#Making Predictions
#Predição
def predict(fromalldata):
  data = torch.Tensor(fromalldata).double().to(device)
  out = rede(data)
  print(out)

individual1 = alldata[idx[-1]]
print(mylabel[idx[-1]])
predict(individual1)

individual2 = alldata[idx[-75]]
print(mylabel[idx[-75]])
predict(individual2)