import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model import net
from data import MnistDataset

dataset = MnistDataset(train=True)
dataset.transform()  # turn into torch type

testdataset = MnistDataset(train=False)
testdataset.transform()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
testloader = DataLoader(testdataset, batch_size=1, shuffle=True)

dataiter = iter(trainloader)
testdataiter = iter(testloader)


def train():

    for epoch in range(6):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def evaluate():

    total = 0
    correct = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            prediction = torch.argmax(outputs)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

    accuracy = correct / total
    print('Accuracy: %f' % accuracy)
    path = os.getcwd() + '/saved_states/' + str(accuracy) + '.pt'
    torch.save(net.state_dict(), path)
    print('Saved State')


def show_next_prediction():
    inputs, labels = testdataiter.next()
    outputs = net(inputs)
    print('Label: ', labels.item(), '\t\tPrediction: ', outputs.argmax())


if __name__ == '__main__':
    train()
    evaluate()