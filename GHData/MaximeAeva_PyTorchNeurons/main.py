from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import random


print("PyTorch version: ", torch.__version__)

random.seed(1)

a = 2*np.random.rand()-1
b = 2*np.random.rand()-1
c = 2*np.random.rand()-1
d = 2*np.random.rand()-1

class Data(Dataset):
    #   Build Data to train and validate

    # Constructor
    def __init__(self, a, b, train = True):
            self.x = torch.zeros(400, 2)
            self.y = torch.zeros(400, 1)
            self.x[:, 0] = torch.arange(-1, 1, 0.005)
            self.x[:, 1] = torch.arange(-1, 1, 0.005)
            if train == True:
                r0=torch.randperm(400)
                self.x[:, 0] = self.x[r0, 0] 
                r1=torch.randperm(400)
                self.x[:, 1] = self.x[r1, 1] 
            else :
                r0=torch.randperm(400)
                self.x[:, 0] = self.x[r0, 0] 
                r1=torch.randperm(400)
                self.x[:, 1] = self.x[r1, 1]

            
            c = 0.3

            self.f1 = (np.sqrt((self.x[:, 0]-a)**2 + (self.x[:, 1]-b)**2)) < c
            self.f2 = (np.sqrt((self.x[:, 0]-c)**2 + (self.x[:, 1]-d)**2)) < c
            self.y[:, 0] = (self.f1 +self.f2).float() 
            self.len = self.x.shape[0]
      
    # Getter
    def __getitem__(self, index):    
        return self.x[index, :], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

class Neuron(nn.Module):
    
    # Constructor
    def __init__(self, sequence):
        super(Neuron, self).__init__()
        self.Sequence = nn.Sequential(sequence)
        
    # Prediction
    def forward(self, x):
        return self.Sequence(x)

def function_train(epochs, batches, model, costFunc, optimizer):
    LOSS = []
    for i in range(epochs):
        for x, y in batches:
            yhat = model(x)
            loss = costFunc(yhat, y)
            LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def plot_decision_regions_2class(model,data_set, title=None, save=None):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1 , X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1 , X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    yhat = np.logical_not((model(XX)[:, 0] > 0.5).numpy()).reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], 'o', label='y=0')
    plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], 'ro', label='y=1')
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig("res/"+ save)
    plt.show()

#def crossEntropy(x):

# Build the Data
train_data = Data(a, b)
val_data = Data(a, b, train = False)
print(train_data.x.shape, " ", train_data.y.shape)

#Choose cost function
costFunc = nn.BCELoss()

#Divide datas in batches
batches = DataLoader(dataset=train_data, batch_size=1)

for i in range(9, 11, 2):
    # Build the model
    seq1 = OrderedDict()
    seq1['linear1'] = nn.Linear(2, i)
    seq1['Tanh1'] = nn.Tanh()
    seq1['linear2'] = nn.Linear(i, 1)
    seq1['Sigmoid'] = nn.Sigmoid()

    seq2 = OrderedDict()
    seq2['linear1'] = nn.Linear(2, i)
    seq2['Tanh1'] = nn.Tanh()
    seq2['linear2'] = nn.Linear(i, i)
    seq2['Tanh2'] = nn.Tanh()
    seq2['linear3'] = nn.Linear(i, 1)
    seq2['Sigmoid'] = nn.Sigmoid()

    seq3 = OrderedDict()
    seq3['linear1'] = nn.Linear(2, i)
    seq3['Tanh1'] = nn.Tanh()
    seq3['linear2'] = nn.Linear(i, i)
    seq3['Tanh2'] = nn.Tanh()
    seq3['linear3'] = nn.Linear(i, i)
    seq3['Tanh3'] = nn.Tanh()
    seq3['linear4'] = nn.Linear(i, 1)
    seq3['Sigmoid'] = nn.Sigmoid()


    model = Neuron(seq1)
    #Choose otpimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    function_train(900, batches, model, costFunc, optimizer)
    strSave = "fig"+str(i)+"_seq1.png"
    plot_decision_regions_2class(model, train_data, title=(seq1.keys()), save=strSave)

    model = Neuron(seq2)
    #Choose otpimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    function_train(900, batches, model, costFunc, optimizer)
    strSave = "fig"+str(i)+"_seq2.png"
    plot_decision_regions_2class(model, train_data, title=(seq2.keys()), save=strSave)

    model = Neuron(seq3)
    #Choose otpimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    function_train(900, batches, model, costFunc, optimizer)
    strSave = "fig"+str(i)+"_seq3.png"
    plot_decision_regions_2class(model, train_data, title=(seq3.keys()), save=strSave)