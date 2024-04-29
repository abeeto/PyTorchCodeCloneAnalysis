from torch.autograd import Variable
from torch.nn import Module, Linear, ReLU, Sigmoid, Tanh, MSELoss, BatchNorm1d, CrossEntropyLoss
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.optim import Adam
import time
import sys




dataset = pd.read_csv("mushrooms.csv")

cols = list(range(4,23))

dataset.drop(dataset.columns[cols],axis=1,inplace=True)

cdf = pd.DataFrame()

for i in list(dataset.columns):
    dataset[i] = pd.Categorical(dataset[i])
    cdf[i] = dataset[i].cat.codes

batch_size = 1000

# Neural network coding begins here
y_one_idx = np.where(np.array(cdf['class']==1))[0][:batch_size]


y_zero_idx = np.where(np.array(cdf['class']==0))[0][:batch_size]

y_idx = np.hstack((y_one_idx, y_zero_idx))
np.random.shuffle(y_idx)

X = torch.from_numpy(np.column_stack((np.array(cdf['cap-shape'][y_idx], dtype=np.float32),
                            np.array(cdf['cap-surface'][y_idx], dtype=np.float32),
                            np.array(cdf['cap-color'][y_idx], dtype=np.float))))


# X = torch.from_numpy(np.column_stack((np.array(cdf['cap-shape'][:batch_size], dtype=np.float), np.array(cdf['cap-surface'][:batch_size], dtype=np.float), np.array(cdf['cap-color'][:batch_size], dtype=np.float))))
# Y = torch.from_numpy(np.array(cdf['class'][:batch_size], dtype=np.long))
Y = torch.from_numpy(np.array(cdf['class'][y_idx], dtype=np.int64))




# X = (X - torch.min(X))/(torch.max(X) - torch.min(X))




x = Variable(X, requires_grad=True).float()
y = Variable(Y)



# print(y.data[:3000])
# exit()
# torch.manual_seed(123)
# x = Variable(torch.rand(1000,3))
# y = Variable((torch.rand(1000)*2).long())



num_classes = 2
input_size = 3
hidden_size = 8
num_epochs = 5

#
# x = Variable(torch.rand(4000, 3))
# y = Variable((torch.rand(4000)*num_classes).long())

class Net(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.l1 = Linear(input_size, 7)

        self.l2 = Linear(7,hidden_size)

        self.l3 = Linear(hidden_size, 7)

        self.l4 = Linear(7,4)

        self.l5 = Linear(4,num_classes)

        self.sig = Tanh()

        self.bn1 = BatchNorm1d(7)
        self.bn2 = BatchNorm1d(hidden_size)
        self.bn3 = BatchNorm1d(7)
        self.bn4 = BatchNorm1d(4)

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(self.sig(out))
        out = self.l2(out)
        out = self.bn2(self.sig(out))
        out = self.l3(out)
        out = self.bn3(self.sig(out))
        out = self.l4(out)
        out = self.bn4(self.sig(out))
        out =  self.l5(out)
        return out



net = Net(input_size, hidden_size, num_classes)



optimizer = optim.Adam(net.parameters(), lr=0.0001)

criterion = CrossEntropyLoss()
# optimizer = optim.SGD(nn.parameters(), lr = 0.01, momentum=0.9)

for epoch in range(9000):

    output = net(x)
    _, y_pred = output.max(1)


    # sys.stdout.write('Target :'+str(y[:4].view(1,-1)).split('\n')[1]+"\n"+'\r')
    # sys.stdout.flush()
    # sys.stdout.write('Predct :'+str(y_pred[:4].view(1,-1)).split('\n')[1]+"\n"+'\r')
    # sys.stdout.flush()

    optimizer.zero_grad()

    loss = criterion(output, y)
    sys.stdout.write('Loss: '+str(loss.data[0])[:5]+'\r')
    sys.stdout.flush()


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Loss: ",loss.data[0])

correct = 0

for i,j in zip(y,y_pred):
    if int(i.data[0]) == int(j.data[0]):
        correct += 1
    # print(i.data[0], j.data[0])

print("Accuracy",int((correct/len(y))*100), "%")
