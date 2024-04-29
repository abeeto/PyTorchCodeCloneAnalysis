import torch
import torchvision
from torchvision import transforms,datasets

# transform -> convert the data to Tensor
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# iterating over the first batch of test data
for data in trainset:
    print(data)
    break

# In data at [0] are images and at [1] are labels
x,y = data[0][0], data[1][0]
print(y)

import matplotlib.pyplot as plt
# showing image
plt.imshow(data[0][0].view(28,28))
plt.show()
# shape is 1*28*28 it should be 28*28 that is why we made it 28*28 to show
print(data[0][0].shape)








