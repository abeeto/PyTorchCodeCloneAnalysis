import torch
import torchvision
import torchvision.datasets as dsets
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import sklearn
import numpy as np
import os


# Import data

par_dir = './Datasets/hand'
path = os.listdir(par_dir)
image_data = []
labels = []
batch_size = 137

for folder in path:
    
    images = os.listdir(par_dir +'/'+ folder)
    
    for image in images:

        file = par_dir +'/'+ folder +'/'+ image
        
        if(os.path.getsize(file)>0):
            img = cv2.imread(file, 0)
            img = cv2.resize(img, (100, 100))
            image_data.append(img)
        labels.append(folder)

image_train = image_data[:1644]
label_train = labels[:1644]

image_train, label_train = sklearn.utils.shuffle(image_train, label_train)

image_test = image_data[1645: 2055]
label_test = labels[1645: 2055]

image_test, label_test = sklearn.utils.shuffle(image_test, label_test)

def get_train_data(input):

    batch_images = image_train[input*batch_size : (input+1)*batch_size]
    batch_labels = label_train[input*batch_size : (input+1)*batch_size]
    batch_labels = np.array(batch_labels, dtype = np.int)

    return batch_images, batch_labels

def get_test_data(input):

    batch_images = image_test[input*batch_size : (input+1)*batch_size]
    batch_labels = label_test[input*batch_size : (input+1)*batch_size]
    batch_labels = np.array(batch_labels, dtype = np.int)

    return batch_images, batch_labels


'''
train_data = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_data = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = True)

# Pipeline

train = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)
'''
# Hyper parameters

lr = 0.001
epochs = 10
input_size = 100*100
im_size = 100
channels = 1
num_classes = 10
loops = int(len(image_train)/batch_size)


# Model

class CNN(nn.Module):

    def __init__(self, num_classes = num_classes):

        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 0),
                                    nn.BatchNorm2d(16),
                                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 0),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                                    nn.ReLU())
        self.fc = nn.Linear(22*22*32, num_classes)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = lr)

for epoch in range(epochs):

    for i in range(loops):

        im, l = get_train_data(i)

        images = Variable(torch.Tensor(im))
        images = images.view(batch_size, channels, im_size, im_size)
        labels = Variable(torch.LongTensor(l))

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if(i % 10 == 0):
            print('Loss: {}'.format(loss.item()))

    print('---------------Epoch: {0}, Loss: {1} ------------------'.format(epoch, loss.item()))

print('Training Complete')

# Validation
'''
total = 0
correct = 0

for i in range(1):

    im, l = get_test_data(i)

    img = Variable(torch.Tensor(im))
    img = img.view(batch_size, channels, im_size, im_size)
    labels = Variable(torch.LongTensor(l))

    out = model(img)

    _, pred = torch.max(out.data, 1)

    for i, _ in enumerate(pred):

        if(pred[i].item() == labels[i].item()):

            correct += 1
        
        total += 1

print('Test samples: {0}, Score: {1}%'.format(total, (correct*100)/total))

'''







    








