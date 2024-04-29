import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from dataset import mnistDataSet
from net import network

classes = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))]
)

batchsize = 50

trainset = mnistDataSet('fashionmnist\\data\\fashion', "train")
testset = mnistDataSet('fashionmnist\\data\\fashion', "t10k")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

if __name__ == "__main__":
    net = network()

    epochs = 10
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for e in range(epochs):
        net.train()
        running_loss = 0.0

        for i, d in enumerate(trainloader, 0):
            inputs, labels = d
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = net(inputs)
            l = loss(outputs, labels.long())
            l.backward()
            optimizer.step()
            n = batchsize*(i+1)
            if n % 10000 == 0:
                print("TRAINING: Epoch {}, Batch {}/60000, Loss {}".format(e,n,l.item()))

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for i, d in enumerate(testloader, 0):
            inputs, labels = d
            inputs, labels = Variable(inputs), Variable(labels)
            labels = labels.long()

            outputs = net(inputs)
            l = loss(outputs, labels)
            pred = torch.argmax(outputs.data, 1)
            total += batchsize
            tru = (pred == labels).sum()
            correct += tru.item()

            test_loss += l.item()

            n = batchsize*(i+1)
            print("TESTING: Epoch {}, Batch {}/10000, Loss {:.4f}, ACC {:.8f}".format(e,n,l.item(), tru.item()/batchsize))
            

        test_loss /= len(testloader.dataset)
        print("TEST RESULT ON EPOCH {}, ACCURACY {:.8f}, AVG. LOSS {:.4f}".format(e,correct/(len(testloader.dataset)),test_loss))

    print("Finished Training")
