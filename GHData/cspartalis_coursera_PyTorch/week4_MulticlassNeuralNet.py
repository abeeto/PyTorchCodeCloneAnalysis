import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(D_in,H)
        self.linear2 = nn.Linear(H,D_out)

    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))
        x=self.linear2(x)
        return x

def train(model,criterion,train_loader,val_loader,optimizer,epochs=100):
    i=0
    useful_stuf = {'training_loss':[],'validation_accuracy':[]}
    for epoch in range(epochs):
        for i,(x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            z=model(x.view(-1,28*28))
            loss=criterion(z,y)
            loss.backward()
            optimizer.step()
            useful_stuf['training_loss'].append(loss.data.item())
        correct=0
        for x,y in val_loader:
            z = model(x.view(-1,28*28))
            _,label = torch.max(z,1)
            correct+= (label==y).sum().item()
        accuracy = 100 * (correct / len(val_dataset))
        useful_stuf['validation_accuracy'].append(accuracy)
    return useful_stuf

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
val_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,batch_size=2000)
val_loader = DataLoader(dataset=val_dataset,batch_size=5000)
criterion = nn.CrossEntropyLoss()
model = Net(784,100,10)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
training_results = train(model,criterion,train_loader,val_loader,optimizer,epochs=30)
print(training_results)