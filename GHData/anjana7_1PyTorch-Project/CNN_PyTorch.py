import torch
import torchvision
from visdom import Visdom
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Mnet(nn.Module):
    "ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"
    def __init__(self):
        super(Mnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return (F.log_softmax(x))



T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_data = torchvision.datasets.MNIST('mnist_Data', transform = T, download = True)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size = 128)

model = Mnet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params = params,lr = 0.01)

n_epoch = 300
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
        
