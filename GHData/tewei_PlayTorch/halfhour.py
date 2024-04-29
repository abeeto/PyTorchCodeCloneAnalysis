import torch
import numpy as np

def test():
    #Matrix
    x = torch.empty(4, 3)
    x = torch.rand(4, 3)
    x = torch.zeros(4, 3, dtype=torch.long)
    x = torch.tensor([8.7, 9.2])
    x = x.new_ones(4, 3, dtype=torch.double)
    x = torch.randn_like(x, dtype=torch.float)
    y = torch.rand(4, 3)
    z = torch.empty(4, 3)
    #print(x.size())
    #print(x, y, x+y)
    torch.add(x, y, out=z)
    #print(z)
    #print(y.add_(x), z.copy_(y))
    #print(x.t_(), x[:, 1])
    x - torch.randn(4, 3) #normal
    x = x.view(12)
    print(x)
    z = x.view(-1,2)
    print(z)
    w = torch.randn(1)
    print(w.item())
    x = torch.ones(4, 3)
    y = x.numpy()
    print(x, y)
    x.add_(1)
    print(y)
    z = torch.from_numpy(y)
    print(z)

    #Devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))

    #Gradient
    x = torch.tensor([-1, 0, 1], dtype=torch.float, requires_grad=True)
    y = 3*x**2+2
    print(y.grad_fn)
    y.backward(x)
    print(x.grad)


    print(x.requires_grad)
    with torch.no_grad():
        print((x**2).requires_grad)

#ConvNet

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

#imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net, list(net.parameters())[0].size())

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=0.0001)
#input.unsqueeze(0) for 1 sample
#Training
for epoch in range(5):
    #break
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #0 img0, 1 img0, 2 img0 
        inputs, labels = data
        #optimizer.zero_grad()
        net.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        #optimizer.step()
        #Just SGD
        lr = 0.01
        for p in net.parameters():
            p.data.sub_(p.grad.data*lr)

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')

#torch.save(net.state_dict(), 'cifar10.pt')
#net.load_state_dict(torch.load('cifar10.pt'))
#device = torch.device('cpu')
#model.load_state_dict(torch.load(PATH, map_location=device))

#Testing
dataiter = iter(testloader)
images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print(_)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0.
total = 0.
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() #Nice!
        c = (predicted == labels).squeeze()
        #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
        for i in range(4):
            class_correct[labels[i]] += c[i].item()
            class_total[labels[i]] += 1

print('Accuracy: %f' % (100*correct/total))
for i in range(10):
    print('Accuracy of %5s: %f' % (classes[i],  100*class_correct[i]/class_total[i]))


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
#inputs, labels = inputs.to(device), labels.to(device)



