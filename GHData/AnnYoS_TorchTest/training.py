import torch
import torchvision
import torchvision.transforms as trans
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

PATH = './save_net.pth'
# define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    numpy_img = img.numpy()
    plt.imshow(np.transpose(numpy_img, (1, 2, 0)))
    plt.show()


def training_network(looptraining, train):
    net = Network()
    # using Classification Cross-Entropy loss
    lossfunc = nn.CrossEntropyLoss()
    # using some Stochastic Gradient Descent with momentum
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for loop in range(looptraining):  # loop multiple time one dataset
        running_loss = 0.0

        # iterator on 'train' begin each time 'loop' increase and 'i' simple interation operation
        for i, data in enumerate(train, 0):
            # get entry and label (data is list of [entry, labels])
            entry, trainlabels = data

            # zero the parameter gradient
            optimizer.zero_grad()

            # forward, backward and optimize
            out = net(entry)
            loss = lossfunc(out, trainlabels)
            loss.backward()
            optimizer.step()

            # print stat
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 iterations
                print("[", loop + 1, ",", i + 1, "]  loss:", (running_loss / 2000))
                running_loss = 0
    print("Training Finish !")
    # saving the current training model
    torch.save(net.state_dict(), PATH)


def testing_network(test):
    # loading our model
    net = Network()
    net.load_state_dict(torch.load(PATH))

    # let's see the perform of the model on all class
    # create 10 class, 1 for each class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        # try to predicted all testing set
        for data in test:
            entry, testlabel = data
            out = net(entry)
            _, predicted = torch.max(out.data, 1)
            c = (predicted == testlabel).squeeze()
            for i in range(4):  # a testing data it's 4 images
                label = testlabel[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool2d = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.f1 = nn.Linear(16 * 5 * 5, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 10)

    def forward(self, x):
        # convolution on images 'x'
        x = self.pool2d(func.relu(self.conv1(x)))
        x = self.pool2d(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # transform 'x' into a Tensor 1x(16*5*5)
        # apply functions on 'x'
        x = func.relu(self.f1(x))
        x = func.relu(self.f2(x))
        x = self.f3(x)
        return x


# transform a PILImage of torchvision dataset into Tensor of normalized range [-1, 1]
transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# load the training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

# load the testing set
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

show = False
training = True
testing = True

# just for testing training and testing sets are download
if show:
    # get some random image of the training
    data_iterator = iter(trainloader)
    showimages, showlabels = data_iterator.next()

    # show images
    imshow(torchvision.utils.make_grid(showimages))
    # print labels
    print(' '.join('%5s' % classes[showlabels[j]] for j in range(4)))

# make training at True to train the model
if training:
    training_network(7, trainloader)

# make testing at True to test your model (if you have any model save, you need to training before)
if testing:
    testing_network(testloader)
