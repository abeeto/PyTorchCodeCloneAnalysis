from net import Net
from torchvision import datasets, transforms
import numpy as np
import torch
import matplotlib.pyplot as plt


dset = datasets.MNIST('../data', train=False, 
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		]))

ind = np.random.randint(len(dset))
x,y = dset[ind]


device = torch.device("cpu")
net = Net().to(device)
net.load_state_dict(torch.load('model.pt'))
torch.no_grad()
net.eval()
output = net(torch.tensor(np.array([x.numpy()]))).detach().numpy()[0]
output = (np.log(1- 1/output))
output = output/np.sum(output)
print(output)
plt.bar(range(len(output)),output)
#plt.imshow(x[0],cmap='gray')
plt.show()


