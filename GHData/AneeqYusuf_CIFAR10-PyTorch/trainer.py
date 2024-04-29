# -*- coding: utf-8 -*-
from net import Net, data_load, nn
import torch
import torchvision.transforms as transforms
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

net = Net()

print (net)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = data_load(transform=transform, folder = '/home/aneeq/Downloads/train.rotfaces/train/', truth_file='/home/aneeq/Downloads/train.rotfaces/train.truth.csv')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)



for epoch in range(0, 4):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs = data[0]
        labels = data[1]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total+=labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the Train images: %d %%' % (
    100 * correct / total))
torch.save(net, 'model.sav')