import torch
import torchvision
from visdom import Visdom
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np

class Mnet(nn.Module):
    
    def __init__(self):
        super(Mnet,self).__init__()
        self.lin1 = nn.Linear(28*28,500)
        self.lin2 = nn.Linear(500,250)
        self.lin3 = nn.Linear(250,100)
        self.final_lin = nn.Linear(100,10)
        self.relu = nn.ReLU()
        
    def forward(self, images):
        s = images.view(-1, 28*28)
        x = self.relu(self.lin1(s))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.final_lin(x)
        
        return(x)


T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_data = torchvision.datasets.MNIST('mnist_Data', transform = T, download = True)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size = 128)

model = Mnet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params = params,lr = 0.01)

n_epoch = 15
n_iterations = 0

vis = Visdom(use_incoming_socket = False)
vis_window = vis.line(np.array([0]), np.array([0]))

for e in range(n_epoch):
    for i,(images,labels) in enumerate(mnist_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)
        
        model.zero_grad()
        loss = cec_loss(output,labels)
        loss.backward()
        
        optimizer.step()
        
        n_iterations+=1
        
        vis.line(np.array([loss.item()]),np.array([n_iterations]),win = vis_window, update = 'append')
        break
        
