import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5,),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root="C:\\Users\\marut\\Desktop\\Research\\C3D\\CIFAR10\\data\\", train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 0)

testset = torchvision.datasets.CIFAR10(root = "C:\\Users\\marut\\Desktop\\Research\\C3D\\CIFAR10\\data\\", train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 0)

classes = ('plane','car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# functions to show an image
def imshow(img):
    img = img/2 + 0.5 #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#Gather some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


###
#show images
###

# print labels


imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s'% classes[labels[j]] for j in range (4)))

###
#Defining a Convolutional Neural Network
###

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

'''
PATH = ".\\cifar_net.pth"
torch.save(net.state_dict(), PATH)
'''
###
# Defining a Loss function and optimizer
###

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)


###
#Training the Network
###

for epoch in range (2): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        #get the inputs; datais a list of [inputs, labels]
        inputs, labels = data

        #Zero the parameter gradients
        optimizer.zero_grad()

        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: #print every 2000 mini-bataches
            print('[%d, %5d] loss: %3f'% (epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0

print("Finished Training")


###
#Saving Trained Model
###


'''
net  = Net ()
net.load_state_dict(torch.load(PATH))
'''
###
#Test the Network on the test Data
###

dataiter = iter(testloader)
images, labels = dataiter.next()

#print images

imshow(torchvision.utils.make_grid(images))
print("Ground Truth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))


#Now we predict what the neural net thinks these examples are

outputs = net(images)

_, predicted = torch.max(outputs, 1)
print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))




###
#How does the model work on the entire dataset?
###

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the network on the 10,000 test images: %d %%"%(100 * correct / total))


###
# Classes that perform well and some that do not perform well
###

class_correct = list(0. for i in range (10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs,1)
        c = (predicted == labels).squeeze()

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print("Accuracy of %5s: %2d %%" % (classes[i], 100 *class_correct[i] / class_total[i]))


































