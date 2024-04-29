import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST('mnist', download = True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle=True, num_workers=0)


#See for later on PyTorch: https://jacobgil.github.io/deeplearning/pruning-deep-learning
# https://cs231n.github.io/convolutional-networks/
conv = torch.nn.Sequential(
    #nn.Dropout(),
    nn.Conv2d(1, 20, 5, 1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2)
)

fullc = torch.nn.Sequential(
    nn.Linear(4*4*50, 500),
    nn.ReLU(),
    nn.Linear(500, 10))

loss_fn = torch.nn.CrossEntropyLoss()
parameterlist = list(conv.parameters()) + list(fullc.parameters())

loss_val = np.zeros(len(trainloader))
optimizer = optim.Adagrad(parameterlist, lr=1e-2)
for i, (x, y) in enumerate(trainloader, 0):
    optimizer.zero_grad()

    y_pred = conv(x)
    y_pred = fullc(y_pred.view(-1, 4 * 4 * 50))

    loss = loss_fn(y_pred, y)
    loss_val[i] = loss.item()
    print(i, loss_val[i])

    loss.backward()
    optimizer.step()

torch.save(conv.state_dict(), 'pytorchtestnet-conv.pt')
torch.save(fullc.state_dict(), 'pytorchtestnet-fullc.pt')

#%%
plt.plot(loss_val[loss_val < 1])
plt.show()




