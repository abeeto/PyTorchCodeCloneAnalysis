# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:21:19 2020

@author: LUI8WX
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)




def num_flat_features( x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




from matplotlib import pyplot as plt

input = torch.randn(32, 32)

plt.imshow(input)

input = torch.randn(32, 32, 3)

plt.imshow(input)

input = torch.randn(1, 3, 32, 32)





#xnp = input.numpy()[0][0]
#xnp.shape
#plt.imshow(xnp, cmap ="gray")

tmp = conv1(input)

#xnp = tmp[0][0].detach().numpy()
#xnp.shape
#plt.imshow(xnp, cmap ="gray")


tmp = F.max_pool2d(F.relu(tmp), (2, 2))

xnp = tmp[0][0].detach().numpy()
xnp.shape
plt.imshow(xnp, cmap ="gray")


tmp = conv2(tmp)

xnp = tmp[0][0].detach().numpy()
xnp.shape
plt.imshow(xnp, cmap ="gray")


tmp = F.max_pool2d(F.relu(tmp), 2)

xnp = tmp[0][0].detach().numpy()
xnp.shape
plt.imshow(xnp, cmap ="gray")


tmp = tmp.view(-1, num_flat_features(tmp))

tmp = fc1(tmp)

xnp = tmp[0].detach().numpy()
xnp.shape
plt.imshow(xnp, cmap ="gray")


tmp = F.relu(tmp)


tmp = fc2(tmp)
tmp = F.relu(tmp)

tmp = fc3(tmp)

tmp = F.relu(tmp)




#######################  explore the parameters  ################################

params = list(net.parameters())
print(len(params))
print(params[0].size())

params[1]

params[2]

params[3]

params[4]

params[5]

params[6]

params[7]

params[8]

params[9]


#######################  clear gradients  ################################


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
net.zero_grad()

# because not a scalar
out.backward(torch.randn(1, 10))

out.backward()
#RuntimeError: grad can be implicitly created only for scalar outputs

# nSamples x nChannels x Height x Width

# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.






#######################  loss function  ################################

input = torch.randn(1, 1, 32, 32)

output = net(input)

target = torch.randn(10)  # a dummy target, for example [10]

target = target.view(1, -1)  # make it the same shape as output  [1,10]

criterion = nn.MSELoss()

loss = criterion(output, target)

print(loss)



print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


#######################  back prop  ################################



net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.


#######################  Update the weights  ################################


# weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


for f in net.parameters():
    print(f)
    
x = list(net.parameters())[0]    

import torch.optim as optim

# subtraction
x.data.sub_(0.1)

x.data.add_(0.1)




x.grad.data





import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

list(net.parameters())[0]



########################  try to load an image  #######################

#from PIL import Image
#import torchvision.transforms.functional as TF
#
#image = Image.open('C:/LV_CHAO_IMAGE/9sky/reshaped/00a716249b324e82bfb1286ded5c2435.jpg')
#x = TF.to_tensor(image)
#x.unsqueeze_(0)
#print(x.shape)
#
#tmp = conv1(x)
#
#F.max_pool2d(F.relu(conv1(x)), (2, 2))





