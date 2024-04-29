import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

class Softmax(nn.Module):
    def __init__(self,in_size,out_size):
        super(Softmax,self).__init__()
        self.linear = nn.Linear(in_size,out_size)
    def forward(self,x):
        out = self.linear(x)
        return out

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
val_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

input_dim = 28*28
output_dim = 10
model = Softmax(input_dim,output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)
n_epochs = 100
accuracy_list = []

trainloader = DataLoader(dataset=train_dataset,batch_size=100)
valloader = DataLoader(dataset=val_dataset,batch_size=5000)

for epoch in range(n_epochs):
    for x,y in trainloader:
        optimizer.zero_grad()
        z = model(x.view(-1,input_dim))
        loss = criterion(z,y)
        loss.backward()
        optimizer.step()

    correct = 0
    for x_test,y_test in valloader:
        z = model(x_test.view(-1,input_dim))
        _,yhat = torch.max(z.data,1)
        correct = correct + (yhat == y_test).sum().item()
    accuracy = correct / 5000

    accuracy_list.append(accuracy)
    print(accuracy_list)
