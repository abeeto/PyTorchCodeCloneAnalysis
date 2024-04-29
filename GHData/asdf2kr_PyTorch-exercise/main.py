import torch
import numpy as np

# ======================================= #
#         Table of Contents               #
# ======================================= #

# 1. Tensor
# 2. Autograd & Variable
# 3. Activation & Optimization & Loss
# 4. Neural network
# 5. Exercise (Using torchvision)
# 6. Save and load the model

# ======================================= #
#              1. Tensor                  #
# ======================================= #

print('####  1. Tensor  ####')

# Create tensors of shape (5) and (5, 2) and (5, 3)
w = torch.Tensor(5)
x = torch.FloatTensor(5)
y = torch.ones(5, 2) # you can also use 'zeros'.
z = torch.empty(5, 3, dtype = torch.float)

# Create random value of tensors.
x = torch.rand(5) # using uniform distribution random, range : 0 ~ 1
y = torch.randn(5, 2) # using normal distribution random, average is 0 and variance is 1

# Print out the Tensor value and shape.
print(x, x.size())
print(y, y.size())

# Featur of tensor. (In-place)
# x is filled with 5.
x.fill_(5)
# y is a new tensor filled with 8.
y = x.add(3)

# Create a numpy array.
x = np.array([1, 2, 3, 4, 5])

# Convert the numpy array to a torch tensor.
y = torch.Tensor(x)
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
x = y.numpy()

# Convert the shape of tensor.
x = torch.rand(5, 2)
y = x.view(1, 1, 5, 2)

# Combine the tensor with the other tensor.
x = torch.ones(5, 2)
y = torch.zeros(5, 2)
z = torch.cat((x, y), 0) #0 is the dimension.

# To use the GPU.
x = torch.ones(5, 1)
y = torch.ones(5, 1)
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    sum = x + y
print(sum) # If you print out the sum variable, you can see the device name.

# etc.
x = torch.rand(5, 1)
x.mean() # mean of Tensor a
x.sum() # sum of Tensor a



# ======================================= #
#         2. Autograd & Variable          #
# ======================================= #

print('#### 2. Autograd & Variable  ####')

# Create a Variable.
from torch.autograd import Variable
x = torch.ones(5, 2)
x = Variable(x, requires_grad = True)

# Print out the Variable.
# Variable is consist of three component (data, grad, grad_fn)
print(x.data)
print(x.grad)
print(x.grad_fn)

# Create tensors.
x = torch.tensor(1., requires_grad = True) # What is different Variable and Tensor?
y = torch.tensor(2., requires_grad = True)
a = torch.tensor(3., requires_grad = True)
b = torch.tensor(4., requires_grad = True)
c = torch.tensor(5., requires_grad = True)

# Build a computational graph.
y = a * x ** 2 + b * x + c

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad) # 2ax + b, So x.grad = 10
print(a.grad) # x^2, so a.gard = 1
print(b.grad) # x, so b.gard = 1
print(c.grad) # c.gard = 1



# ======================================= #
#   3. Activation & Optimization & Loss   #
# ======================================= #

print('#### 3. Activation & Optimization & Loss  ####')

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# Create a Tensor.
input = torch.randn(1, 1, 5, 2, requires_grad = True)

# Use the activation function called relu, sigmoid, tanh.
relu = F.relu(input)
sigmoid = F.sigmoid(input)
tanh = F.tanh(input)
print(relu)
print(sigmoid)
print(tanh)

#  Max pooling use the torch.nn.functional.
pooling = F.max_pool2d(relu, kernel_size = (2, 2))
print(pooling)

#  Max pooling use the torch.nn.
pool = nn.MaxPool2d(2, stride = 1)
pooling = pool(relu)
print(pooling)

# Use the SGD optimizer.
#optimizer = torch.optim.SGD(Net.parameters(), lr = 0.001, momentum = 0.9)

# Loss function for Multi-class problem.
# Use Softmax activation function and the CrossEntorpy.
#loss_function = nn.CrossEntropyLoss()

# How to train Network?
"""
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad() # Initialize
        inputs = inputs.cuda(), labels = labels.cuda()
        out = Net(input)
        loss = loss_function(out, labels)
        loss.backword()
        optim.step()
"""



# ======================================= #
#     4. Neural Network using PyTorch     #
# ======================================= #

print('#### 4. Neural Network using PyTorch  ####')

# Using torch.nn.functional
input = torch.ones(1, 1, 28, 28, requires_grad = True)
filter = torch.ones(1, 1, 5, 5)
output = F.conv2d(input, filter)
print(output)

# Using torch.nn
input = torch.ones(1, 1, 28, 28, requires_grad = True)
func = nn.Conv2d(1, 1, 5)
print(func.weight)
output = func(input)
print(output)

# If you want to use the received filter, using torch.nn
input = torch.ones(1, 1, 28, 28, requires_grad = True)
func = nn.Conv2d(1, 1, 5, bias = None)
func.weight = torch.nn.Parameter(torch.ones(1, 1, 5, 5) + 1)
print(func.weight)
output = func(input)



# ======================================= #
#              5. Exercise                #
# ======================================= #

print('#### 5. Exercise  ####')

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1):  # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # 입력을 받은 후,
        inputs, labels = data

        # Variable로 감싸고
        inputs, labels = inputs.cuda(), labels.cuda()#Variable(inputs.cuda()), Variable(labels.cuda())

        # 변화도 매개변수를 0으로 만든 후
        optimizer.zero_grad()

        # 학습 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        #print(loss.data)
        #print(loss.data[0])
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



# ======================================= #
#       6. Save and load the model        #
# ======================================= #

print('#### Save and load the model  ####')

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
