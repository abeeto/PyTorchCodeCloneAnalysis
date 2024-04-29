#save a mode
import torch
import numpy as np 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

#fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) #x data(tensor), shape(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) #this give y a noise with x's size

#save func
def save():
    #create model
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    #train net work
    optimizer = torch.optim.SGD(net.parameters(), 0.2)
    loss_func = torch.nn.MSELoss()
    for i in range(200):
        prediction = net(x)
        loss =  loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    #save network
    torch.save(net, 'model.pkl')
    torch.save(net.state_dict(), 'params.pkl')

#load func
def load():
    net2 = torch.load('model.pkl')
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)   

#load only parameters
def loadPara():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load('params.pkl'))
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()   

save()
load()
loadPara()

